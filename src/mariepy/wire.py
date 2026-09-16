"""Wire coils: closed loops of thin wire with a triangle basis.

Ported from MARIE 3.0's wire path: ``src_geometry/scoil_geometry/geo_wcoil.m``,
``mesh_geo/Mesh_Wire.m`` and ``triangle_geo/ProcessLoops.m`` for the geometry;
``src_integral_equations/src_wie`` (``Assembly_WIE_triangle_basis_vec.m``,
``assembly_wle.m``, ``excitation_wire.m``) for the wire's own system; and
``src_wvie/Cpp_Assembly/src/Assemble_tri_coupling_matrix_{N,K}_*.cpp`` for its
coupling to the body.

A wire is a polyline of segments. Each basis function is a triangle, a hat, on
two consecutive segments: it rises from the node before its own, peaks at its
own node and falls to the node after. The current follows the wire. A port or
a lumped element sits on the segment that leaves a node, and MARIE splits its
voltage or impedance evenly between the two basis functions that segment
carries.

Only closed loops are read: MARIE's open-wire branch of ``ProcessLoops.m``
assigns rows of mismatched size and cannot run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from mariepy.coil import Port
from mariepy.constants import COPPER_CONDUCTIVITY, VACUUM_PERMEABILITY, Medium
from mariepy.coupling import _cell_rule, _electric, _magnetic
from mariepy.mesh import _section
from mariepy.quadrature import gauss_legendre_1d
from mariepy.sie import CoilSystem

__all__ = [
    "WIRE_RADIUS",
    "WireCoil",
    "assemble",
    "coupling_k",
    "coupling_n",
    "impedance",
]

WIRE_RADIUS = 0.0005
"""MARIE's ``emc.thick_wire`` over two, in metres."""

_GMSH_POINT = 15
_GMSH_LINE = 1


@dataclass(frozen=True)
class WireCoil:
    """Closed wire loops with their triangle basis, ports and loads.

    Attributes
    ----------
    first, centre, last
        Each basis function's three nodes, MARIE's ``F_point``, ``S_point``
        and ``T_point``, shape ``(n_dof, 3)``.
    loops
        Each loop's basis functions, as ``(start, stop)`` ranges.
    ports
        Ports and lumped elements in file order, each spanning the two basis
        functions of the segment it sits on.
    radius
        Wire radius in metres.
    """

    first: torch.Tensor
    centre: torch.Tensor
    last: torch.Tensor
    loops: tuple[tuple[int, int], ...]
    ports: tuple[Port, ...] = ()
    radius: float = WIRE_RADIUS

    @property
    def n_dof(self) -> int:
        """Number of basis functions."""
        return int(self.centre.shape[0])

    @property
    def n_driven(self) -> int:
        """Number of driven ports."""
        return sum(1 for port in self.ports if port.kind == "port")

    @property
    def device(self) -> torch.device:
        """Device the geometry lives on."""
        return self.centre.device

    def points(self) -> torch.Tensor:
        """Every node the wire passes through, shape ``(n_dof, 3)``."""
        return self.centre

    def left_lengths(self) -> torch.Tensor:
        """Length of each basis function's rising segment, MARIE's ``Dl``."""
        return torch.linalg.vector_norm(self.centre - self.first, dim=-1)

    def right_lengths(self) -> torch.Tensor:
        """Length of each basis function's falling segment, MARIE's ``Dr``."""
        return torch.linalg.vector_norm(self.last - self.centre, dim=-1)

    def following(self, dof: int) -> int:
        """Return the basis function after ``dof`` along its own loop."""
        for start, stop in self.loops:
            if start <= dof < stop:
                return start + (dof - start + 1) % (stop - start)
        raise IndexError(f"basis function {dof} is on no loop")

    @classmethod
    def build(
        cls,
        nodes: torch.Tensor,
        segments: list[tuple[int, int]],
        port_nodes: list[int],
        elements: tuple[Port, ...] = (),
        *,
        radius: float = WIRE_RADIUS,
    ) -> WireCoil:
        """Build the basis of closed loops and place the ports on it.

        Parameters
        ----------
        nodes
            Node coordinates, shape ``(n_nodes, 3)``.
        segments
            Node pairs in file order; each loop is a run of consecutive
            segments whose last ends where its first begins.
        port_nodes
            The node each port or element sits at, in file order.
        elements
            The ports and lumped elements, in file order, as
            :func:`mariepy.coil.read_lumped_elements` reads them. The ``n``-th
            takes the ``n``-th port node.
        radius
            Wire radius in metres.

        Returns
        -------
        WireCoil
            The loops and their ports.

        Raises
        ------
        NotImplementedError
            If a run of segments does not close.
        ValueError
            If there are fewer port nodes than elements, or a port node is on
            no loop.
        """
        loops = []
        start = 0
        for k, (head, _) in enumerate(segments):
            if k > start and head != segments[k - 1][1]:
                raise NotImplementedError(
                    f"segments {start}..{k - 1} leave node {segments[start][0]} "
                    "without returning: open wires are not supported"
                )
            if segments[k][1] == segments[start][0]:
                loops.append((start, k + 1))
                start = k + 1
        if start != len(segments):
            raise NotImplementedError(
                f"segments from {start} on do not close: open wires are not supported"
            )

        previous = []
        for begin, end in loops:
            previous += [
                begin + (k - begin - 1) % (end - begin) for k in range(begin, end)
            ]
        heads = torch.tensor([s[0] for s in segments], device=nodes.device)
        tails = torch.tensor([s[1] for s in segments], device=nodes.device)
        first = nodes[heads[torch.tensor(previous, device=nodes.device)]]
        centre = nodes[heads]
        last = nodes[tails]

        if len(port_nodes) < len(elements):
            raise ValueError(
                f"{len(elements)} elements but only {len(port_nodes)} port nodes"
            )
        dof_of_node = {int(node): k for k, node in enumerate(heads.tolist())}
        coil = cls(
            first=first,
            centre=centre,
            last=last,
            loops=tuple(loops),
            radius=radius,
        )
        placed = []
        for element, node in zip(elements, port_nodes, strict=False):
            if node not in dof_of_node:
                raise ValueError(f"port node {node} is on no loop")
            dof = dof_of_node[node]
            placed.append(
                replace(
                    element,
                    dofs=torch.tensor(
                        [dof, coil.following(dof)],
                        dtype=torch.long,
                        device=nodes.device,
                    ),
                )
            )
        return replace(coil, ports=tuple(placed))

    @classmethod
    def read_gmsh22(
        cls,
        path: str | Path,
        elements: tuple[Port, ...] = (),
        *,
        radius: float = WIRE_RADIUS,
        device: torch.device | str = "cpu",
    ) -> WireCoil:
        """Read a GMSH 2.2 wire mesh, as ``Mesh_Wire.m`` does.

        Line elements are the segments, in file order; point elements are the
        port nodes, in file order.

        Parameters
        ----------
        path
            File to read.
        elements
            The ports and lumped elements, in file order.
        radius
            Wire radius in metres.
        device
            Device the geometry is built on.

        Returns
        -------
        WireCoil
            The loops and their ports.

        Raises
        ------
        ValueError
            If the file has no ``$Nodes`` or no ``$Elements`` section.
        """
        text = Path(path).read_text()
        node_block = _section(text, "Nodes")
        element_block = _section(text, "Elements")
        if node_block is None or element_block is None:
            raise ValueError(f"{path} has no $Nodes or no $Elements section")
        coordinates, position = [], {}
        for record in node_block[1:]:
            fields = record.split()
            position[int(fields[0])] = len(coordinates)
            coordinates.append([float(value) for value in fields[1:4]])
        segments, port_nodes = [], []
        for record in element_block[1:]:
            fields = [int(field) for field in record.split()]
            kind, n_tags = fields[1], fields[2]
            ends = fields[3 + n_tags :]
            if kind == _GMSH_POINT:
                port_nodes.append(position[ends[-1]])
            elif kind == _GMSH_LINE:
                segments.append((position[ends[-2]], position[ends[-1]]))
        nodes = torch.tensor(coordinates, dtype=torch.float64, device=device)
        return cls.build(nodes, segments, port_nodes, elements, radius=radius)

    @classmethod
    def loop(
        cls,
        loop_radius: float,
        n_segments: int,
        elements: tuple[Port, ...] = (),
        *,
        at: tuple[int, ...] | None = None,
        centre: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = WIRE_RADIUS,
        device: torch.device | str = "cpu",
    ) -> WireCoil:
        """Build a circular loop in the xy-plane.

        Parameters
        ----------
        loop_radius
            Radius of the loop in metres.
        n_segments
            Segments around it.
        elements
            Ports and lumped elements.
        at
            The node each element sits at; by default they are spread evenly
            from node 0.
        centre
            Centre of the loop.
        radius
            Wire radius in metres.
        device
            Device the geometry is built on.

        Returns
        -------
        WireCoil
            The loop and its ports.
        """
        angle = torch.arange(n_segments, dtype=torch.float64) * (
            2 * math.pi / n_segments
        )
        nodes = torch.stack(
            [
                centre[0] + loop_radius * torch.cos(angle),
                centre[1] + loop_radius * torch.sin(angle),
                torch.full_like(angle, centre[2]),
            ],
            dim=1,
        ).to(device)
        segments = [(k, (k + 1) % n_segments) for k in range(n_segments)]
        if at is None:
            at = tuple(
                k * n_segments // max(1, len(elements)) for k in range(len(elements))
            )
        return cls.build(nodes, segments, list(at), elements, radius=radius)


# --------------------------------------------------------------------------
# The wire's own system
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Segments:
    """Quadrature on every basis function's two segments."""

    points: torch.Tensor  # (n, 2, q, 3): rising then falling segment
    shape: torch.Tensor  # (2, q): the hat's value
    slope: torch.Tensor  # (n, 2): the hat's derivative along the wire
    tangent: torch.Tensor  # (n, 2, 3)
    length: torch.Tensor  # (n, 2)
    along: torch.Tensor  # (n, 2, q): distance from each segment's start
    weights: torch.Tensor  # (q,)


def _segments(coil: WireCoil, order: int) -> _Segments:
    weights, nodes = gauss_legendre_1d(order, device=coil.device, dtype=torch.float64)
    u = (nodes + 1) / 2
    left, right = coil.left_lengths(), coil.right_lengths()
    length = torch.stack([left, right], dim=1)
    start = torch.stack([coil.first, coil.centre], dim=1)
    stop = torch.stack([coil.centre, coil.last], dim=1)
    tangent = (stop - start) / length[..., None]
    points = (
        start[:, :, None, :] + u[None, None, :, None] * (stop - start)[:, :, None, :]
    )
    shape = torch.stack([u, 1 - u])
    slope = torch.stack([1 / left, -1 / right], dim=1)
    along = length[:, :, None] * u[None, None, :]
    return _Segments(points, shape, slope, tangent, length, along, weights)


def impedance(
    coil: WireCoil,
    medium: Medium,
    *,
    order: int = 6,
    conductivity: float = COPPER_CONDUCTIVITY,
    chunk: int = 1 << 14,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the wire's EFIE matrix and its copper loss.

    Ported from ``Assembly_WIE_triangle_basis_vec.m``. Segments are integrated
    with the reduced thin-wire kernel, ``exp(-jkR)/R`` with
    ``R = sqrt(|r - r'|^2 + a^2)``, on a Gauss rule of ``order`` points. Where
    a segment meets itself, in one basis function or in two neighbours, the
    inner integral is MARIE's closed form.

    Parameters
    ----------
    coil
        The wire and its basis.
    medium
        Free-space constants at the working frequency.
    order
        Gauss points per segment, MARIE's ``Quad_order_wie``.
    conductivity
        Conductivity of the wire in S/m.
    chunk
        Basis-function pairs integrated at a time.

    Returns
    -------
    matrix : torch.Tensor
        The EFIE matrix with the copper loss on its diagonal, shape
        ``(n_dof, n_dof)``: MARIE's ``Z`` before the lumped loads.
    copper : torch.Tensor
        The copper loss alone.
    """
    k = medium.wavenumber
    a = coil.radius
    s = _segments(coil, order)
    n = coil.n_dof
    device = coil.device
    lower = torch.zeros((n, n), dtype=torch.complex128, device=device)

    source, observer = torch.tril_indices(n, n, offset=-1, device=device).flip(0)
    # MARIE compares nodes exactly: a neighbour starts where a basis function
    # peaks (its falling segment is the neighbour's rising one), or peaks where
    # the basis function starts (the other way round).
    after = (coil.centre[source] == coil.first[observer]).all(dim=1)
    before = (coil.first[source] == coil.centre[observer]).all(dim=1) & ~after

    for begin in range(0, source.numel(), chunk):
        src = source[begin : begin + chunk]
        obs = observer[begin : begin + chunk]
        blocks = _regular(s, src, obs, k, a)  # (pairs, 2, 2)
        a_mask = after[begin : begin + chunk]
        b_mask = before[begin : begin + chunk]
        if a_mask.any():
            blocks[a_mask, 1, 0] = _neighbour_after(s, src[a_mask], obs[a_mask], k, a)
        if b_mask.any():
            blocks[b_mask, 0, 1] = _neighbour_before(s, src[b_mask], obs[b_mask], k, a)
        lower[obs, src] = blocks.sum(dim=(1, 2))

    every = torch.arange(n, device=device)
    diagonal = _regular(s, every, every, k, a)
    diagonal[:, 0, 0] = _self_rising(s, every, k, a)
    diagonal[:, 1, 1] = _self_falling(s, every, k, a)
    lower[every, every] = diagonal.sum(dim=(1, 2))

    scale = 1j * medium.angular_frequency * VACUUM_PERMEABILITY / (4 * math.pi)
    matrix = scale * (lower + lower.transpose(0, 1) - torch.diag(torch.diagonal(lower)))

    depth = math.sqrt(2.0) / math.sqrt(
        medium.angular_frequency * VACUUM_PERMEABILITY * conductivity
    )
    resistance = (1 / conductivity) / (math.pi * (2 * a - depth) * depth)
    left, right = coil.left_lengths(), coil.right_lengths()
    copper = torch.diag(resistance * (left + right).abs() / 3).to(torch.complex128)
    return matrix + copper, copper


def _regular(s: _Segments, src, obs, k, a):
    """Integrate every segment pair of two basis functions with the reduced kernel."""
    difference = (
        s.points[src][:, :, None, :, None, :] - s.points[obs][:, None, :, None, :, :]
    )
    distance = torch.sqrt((difference**2).sum(-1) + a * a)  # (p, 2, 2, q, q)
    green = torch.exp(-1j * k * distance) / distance
    dot = torch.einsum("psc,poc->pso", s.tangent[src], s.tangent[obs])
    shape = s.shape[None, :, None, :, None] * s.shape[None, None, :, None, :]
    slope = s.slope[src][:, :, None] * s.slope[obs][:, None, :]
    kernel = shape * dot[..., None, None] - slope[..., None, None] / k**2
    weight = s.weights[:, None] * s.weights[None, :]
    half = s.length[src][:, :, None] / 2 * s.length[obs][:, None, :] / 2
    return half * torch.einsum("qr,psoqr->pso", weight.to(green.dtype), kernel * green)


def _closed_form(length, along, a, k):
    """Return the pieces MARIE's closed forms share: the roots, the logarithm, ``jkL/2``."""
    root_start = torch.sqrt(a * a + along**2)
    root_end = torch.sqrt(a * a + (along - length) ** 2)
    logarithm = torch.log(
        (along + root_start).abs() / (along - length + root_end).abs()
    )
    term = 1j * k * length / 2
    return root_start, root_end, logarithm, term


def _self_rising(s: _Segments, dofs, k, a):
    """Integrate a rising segment against itself."""
    length = s.length[dofs, 0][:, None]
    along = s.along[dofs, 0]
    root_start, root_end, logarithm, term = _closed_form(length, along, a, k)
    s1 = (along / length) * logarithm + root_end / length - root_start / length - term
    s2 = (logarithm - 2 * term) / length
    integrand = s.shape[0] * s1 - (1 / k**2) * s.slope[dofs, 0][:, None] * s2
    return length[:, 0] / 2 * (s.weights * integrand).sum(-1)


def _self_falling(s: _Segments, dofs, k, a):
    """Integrate a falling segment against itself."""
    length = s.length[dofs, 1][:, None]
    along = s.along[dofs, 1]
    root_start, root_end, logarithm, term = _closed_form(length, along, a, k)
    s1 = (
        (1 - along / length) * logarithm
        - root_end / length
        + root_start / length
        - term
    )
    s2 = -(logarithm - 2 * term) / length
    integrand = s.shape[1] * s1 - (1 / k**2) * s.slope[dofs, 1][:, None] * s2
    return length[:, 0] / 2 * (s.weights * integrand).sum(-1)


def _neighbour_after(s: _Segments, src, obs, k, a):
    """Integrate a source's falling segment against the observer's rising one, the same segment."""
    length = s.length[src, 1][:, None]
    along = s.along[obs, 0]
    root_start, root_end, logarithm, term = _closed_form(length, along, a, k)
    s1 = (
        (1 - along / length) * logarithm
        - root_end / length
        + root_start / length
        - term
    )
    s2 = -(logarithm - 2 * term) / length
    integrand = s.shape[0] * s1 - (1 / k**2) * s.slope[obs, 0][:, None] * s2
    return s.length[obs, 0] / 2 * (s.weights * integrand).sum(-1)


def _neighbour_before(s: _Segments, src, obs, k, a):
    """Integrate a source's rising segment against the observer's falling one, the same segment."""
    length = s.length[src, 0][:, None]
    along = s.along[obs, 1]
    root_start, root_end, logarithm, term = _closed_form(length, along, a, k)
    s1 = (along / length) * logarithm + root_end / length - root_start / length - term
    s2 = (logarithm - 2 * term) / length
    integrand = s.shape[1] * s1 - (1 / k**2) * s.slope[obs, 1][:, None] * s2
    return s.length[obs, 1] / 2 * (s.weights * integrand).sum(-1)


def lumped_loads(
    matrix: torch.Tensor, coil: WireCoil, angular_frequency: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the lumped elements, as ``assembly_wle.m`` does.

    Half of each element's impedance joins the diagonal entry of each of its
    two basis functions; a mutual inductor adds half its mutual impedance from
    its own two basis functions to its partner's, in its own direction only,
    the partner adding the other.

    Parameters
    ----------
    matrix
        The EFIE matrix.
    coil
        The wire and its elements.
    angular_frequency
        Working frequency in rad/s.

    Returns
    -------
    loaded : torch.Tensor
        The matrix with every element added.
    loss : torch.Tensor
        The resistive part of what was added.
    """
    loaded = matrix.clone()
    loss = torch.zeros_like(matrix)
    by_tag = {port.tag: port for port in coil.ports}
    for port in coil.ports:
        if port.kind != "element":
            continue
        value, resistance = port.impedance(angular_frequency)
        first, second = port.dofs.tolist()
        for dof in (first, second):
            loaded[dof, dof] += 0.5 * value
            loss[dof, dof] += 0.5 * resistance
        if port.load == "mutual_inductor" and port.coupled_tag is not None:
            partner = by_tag[port.coupled_tag].dofs.tolist()
            mutual = 1j * angular_frequency * port.coupled_value
            loaded[first, partner[0]] += 0.5 * mutual
            loaded[second, partner[1]] += 0.5 * mutual
    return loaded, loss


def port_excitation(coil: WireCoil) -> torch.Tensor:
    """Drive each port with half its voltage on each of its two basis functions.

    Ported from ``excitation_wire.m``, for closed loops.

    Parameters
    ----------
    coil
        The wire and its ports.

    Returns
    -------
    torch.Tensor
        Shape ``(n_driven, n_dof)``, complex.
    """
    driven = [port for port in coil.ports if port.kind == "port"]
    excitation = torch.zeros(
        (len(driven), coil.n_dof), dtype=torch.complex128, device=coil.device
    )
    for row, port in enumerate(driven):
        excitation[row, port.dofs] = -0.5 * port.voltage
    return excitation


def assemble(coil: WireCoil, medium: Medium, *, order: int = 6) -> CoilSystem:
    """Build the wire's system matrix, its losses and its port drive.

    Ported from ``wie_assembly.m``.

    Parameters
    ----------
    coil
        The wire, its basis and its ports.
    medium
        Free-space constants at the working frequency.
    order
        Gauss points per segment, MARIE's ``Quad_order_wie``.

    Returns
    -------
    CoilSystem
        As :func:`mariepy.sie.assemble` gives it for a surface coil.
    """
    matrix, copper = impedance(coil, medium, order=order)
    loaded, lumped = lumped_loads(matrix, coil, medium.angular_frequency)
    return CoilSystem(
        impedance=-loaded,
        excitation=port_excitation(coil),
        copper_loss=copper,
        lumped_loss=lumped,
    )


# --------------------------------------------------------------------------
# Coupling to the body
# --------------------------------------------------------------------------


def coupling_n(
    coil: WireCoil,
    dofs: torch.Tensor,
    points: torch.Tensor,
    medium: Medium,
    *,
    order: int = 4,
    cell_size: float | None = None,
    cell_order: int = 2,
    basis_term: int = 0,
) -> torch.Tensor:
    """Give the electric field each wire basis function puts on its observer.

    Ported from ``Assemble_tri_coupling_matrix_N_*.cpp``, with the falling
    segment's current falling: MARIE's sources give it the rising ramp of the
    other segment, which is not the basis the wire's own matrix uses.

    Parameters
    ----------
    coil
        The wire and its basis.
    dofs
        Shape ``(n,)``: one basis function per observer.
    points
        Shape ``(n, 3)``: the observers.
    medium
        Supplies the wavenumber and ``j omega eps_0``.
    order
        Gauss points per segment, MARIE's ``Quad_order_wie_coup``.
    cell_size, cell_order, basis_term
        As in :func:`mariepy.coupling.coupling_n`.

    Returns
    -------
    torch.Tensor
        Shape ``(n, 3)``, complex.
    """
    return _couple(
        coil, dofs, points, medium, order, cell_size, cell_order, basis_term, _electric
    )


def coupling_k(
    coil: WireCoil,
    dofs: torch.Tensor,
    points: torch.Tensor,
    medium: Medium,
    *,
    order: int = 4,
    cell_size: float | None = None,
    cell_order: int = 2,
    basis_term: int = 0,
) -> torch.Tensor:
    """Give the magnetic field each wire basis function puts on its observer.

    Parameters
    ----------
    coil, dofs, points, medium, order, cell_size, cell_order, basis_term
        As in :func:`coupling_n`.

    Returns
    -------
    torch.Tensor
        Shape ``(n, 3)``, complex.
    """
    return _couple(
        coil, dofs, points, medium, order, cell_size, cell_order, basis_term, _magnetic
    )


_CHUNK = 1 << 20


def _couple(coil, dofs, points, medium, order, cell_size, cell_order, term, kernel):
    """Integrate a kernel over each basis function's two segments and its cell."""
    device = points.device
    s = _segments(coil, order)
    if cell_size is None:
        cell_weights = torch.ones(1, dtype=torch.float64, device=device)
        offsets = torch.zeros((1, 3), dtype=torch.float64, device=device)
    else:
        cell_weights, nodes, factors = _cell_rule(cell_order, device, torch.float64)
        cell_weights = cell_weights * factors[term]
        offsets = cell_size / 2 * nodes
    # Current per unit length at each point, times the segment's half length.
    current = s.shape[None, :, :, None] * s.tangent[:, :, None, :]  # (n_dof, 2, q, 3)
    weight = (s.length[:, :, None] / 2) * s.weights[None, None, :]  # (n_dof, 2, q)

    budget = max(1, _CHUNK // max(1, offsets.shape[0] * 2 * s.weights.numel()))
    pieces = []
    for begin in range(0, dofs.numel(), budget):
        chosen = dofs[begin : begin + budget]
        observer = points[begin : begin + budget, None, :] + offsets[None, :, :]
        separation = (
            observer[:, :, None, None, :] - s.points[chosen][:, None, :, :, :]
        )  # (p, cells, 2, q, 3)
        value = kernel(
            separation, current[chosen][:, None].to(torch.complex128), medium
        )
        pieces.append(
            torch.einsum(
                "g,psq,pgsqc->pc",
                cell_weights.to(torch.complex128),
                weight[chosen].to(torch.complex128),
                value,
            )
        )
    if not pieces:
        return torch.zeros((0, 3), dtype=torch.complex128, device=device)
    return torch.cat(pieces)
