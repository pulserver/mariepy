"""Wire coils: thin wires, closed or open, with a triangle basis.

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

MARIE's open-wire branch of ``ProcessLoops.m`` assigns rows of mismatched
size and cannot run. An open wire here carries a basis function at every
interior node, and none at its two ends, where the current vanishes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import COPPER_CONDUCTIVITY, VACUUM_PERMEABILITY, Medium
from mariepy.coupling import _cell_rule, _electric, _magnetic
from mariepy.mesh import _section
from mariepy.quadrature import gauss_legendre_1d, gauss_triangle
from mariepy.sie import CoilSystem, block_diagonal

__all__ = [
    "WIRE_RADIUS",
    "CombinedCoil",
    "WireCoil",
    "assemble",
    "assemble_combined",
    "coupling_k",
    "coupling_n",
    "impedance",
    "surface_coupling",
]

WIRE_RADIUS = 0.0005
"""MARIE's ``emc.thick_wire`` over two, in metres."""

_GMSH_POINT = 15
_GMSH_LINE = 1


@dataclass(frozen=True)
class WireCoil:
    """Wires, closed or open, with their triangle basis, ports and loads.

    Attributes
    ----------
    first, centre, last
        Each basis function's three nodes, MARIE's ``F_point``, ``S_point``
        and ``T_point``, shape ``(n_dof, 3)``.
    loops
        Each wire's basis functions, as ``(start, stop)`` ranges.
    ports
        Ports and lumped elements in file order, each spanning the basis
        functions of the segment it sits on.
    radius
        Wire radius in metres.
    closed
        Whether each wire closes on itself; all of them by default.
    ends
        The end nodes of the open wires, which carry no basis function.
    """

    first: torch.Tensor
    centre: torch.Tensor
    last: torch.Tensor
    loops: tuple[tuple[int, int], ...]
    ports: tuple[Port, ...] = ()
    radius: float = WIRE_RADIUS
    closed: tuple[bool, ...] | None = None
    ends: torch.Tensor | None = None

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
        """Return every node the wire passes through, shape ``(n_nodes, 3)``."""
        if self.ends is None or not self.ends.numel():
            return self.centre
        return torch.cat([self.centre, self.ends])

    def left_lengths(self) -> torch.Tensor:
        """Length of each basis function's rising segment, MARIE's ``Dl``."""
        return torch.linalg.vector_norm(self.centre - self.first, dim=-1)

    def right_lengths(self) -> torch.Tensor:
        """Length of each basis function's falling segment, MARIE's ``Dr``."""
        return torch.linalg.vector_norm(self.last - self.centre, dim=-1)

    def following(self, dof: int) -> int | None:
        """Return the basis function after ``dof`` along its wire, or None past an open end."""
        closed = self.closed or (True,) * len(self.loops)
        for (start, stop), shut in zip(self.loops, closed, strict=True):
            if start <= dof < stop:
                if shut:
                    return start + (dof - start + 1) % (stop - start)
                return dof + 1 if dof + 1 < stop else None
        raise IndexError(f"basis function {dof} is on no wire")

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
        """Build the basis of the wires and place the ports on it.

        A wire is a run of consecutive segments, each starting where the one
        before ended. A wire whose last segment ends where its first begins is
        closed and carries a basis function at every node, as
        ``ProcessLoops.m`` gives it; any other is open and carries one at
        every node but its two ends, the current vanishing there.

        Parameters
        ----------
        nodes
            Node coordinates, shape ``(n_nodes, 3)``.
        segments
            Node pairs in file order.
        port_nodes
            The node each port or element sits at, in file order. It sits on
            the segment that leaves that node.
        elements
            The ports and lumped elements, in file order, as
            :func:`mariepy.coil.read_lumped_elements` reads them. The ``n``-th
            takes the ``n``-th port node.
        radius
            Wire radius in metres.

        Returns
        -------
        WireCoil
            The wires and their ports.

        Raises
        ------
        ValueError
            If there are fewer port nodes than elements, a port node leaves no
            segment, or a port sits on an open wire's end segment, where the
            basis cannot carry a gap.
        """
        runs = []
        start = 0
        for k in range(1, len(segments) + 1):
            broken = k == len(segments) or segments[k][0] != segments[k - 1][1]
            closes = segments[k - 1][1] == segments[start][0]
            if broken or closes:
                runs.append((start, k))
                start = k

        first, centre, last, ranges, closed = [], [], [], [], []
        ends = []
        count = 0
        for begin, end in runs:
            heads = [segments[k][0] for k in range(begin, end)]
            tails = [segments[k][1] for k in range(begin, end)]
            shut = tails[-1] == heads[0]
            if shut:
                previous = heads[-1:] + heads[:-1]
                own, after = heads, tails
            else:
                previous, own, after = heads[:-1], heads[1:], tails[1:]
                ends += [heads[0], tails[-1]]
            first += previous
            centre += own
            last += after
            ranges.append((count, count + len(own)))
            closed.append(shut)
            count += len(own)
        dof_of_node = {}
        for index, node in enumerate(centre):
            dof_of_node.setdefault(int(node), index)

        device = nodes.device

        def take(index):
            return nodes[torch.tensor(index, dtype=torch.long, device=device)]

        coil = cls(
            first=take(first),
            centre=take(centre),
            last=take(last),
            loops=tuple(ranges),
            radius=radius,
            closed=tuple(closed),
            ends=take(ends) if ends else nodes.new_zeros((0, 3)),
        )

        if len(port_nodes) < len(elements):
            raise ValueError(
                f"{len(elements)} elements but only {len(port_nodes)} port nodes"
            )
        leaving = {int(s[0]): int(s[1]) for s in segments}
        placed = []
        for element, node in zip(elements, port_nodes, strict=False):
            node = int(node)
            if node not in leaving:
                raise ValueError(f"port node {node} leaves no segment")
            here, there = dof_of_node.get(node), dof_of_node.get(leaving[node])
            if here is None or there is None:
                raise ValueError(
                    f"element {element.tag} sits on an open wire's end segment"
                )
            placed.append(
                replace(
                    element,
                    dofs=torch.tensor([here, there], dtype=torch.long, device=device),
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
            The wires and their ports.

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


def _diagonal_sparse(diagonal: torch.Tensor) -> torch.Tensor:
    """Carry a diagonal as a sparse square matrix."""
    n = diagonal.shape[0]
    index = torch.arange(n, device=diagonal.device)
    return torch.sparse_coo_tensor(
        torch.stack([index, index]), diagonal, (n, n)
    ).coalesce()


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
        The copper loss alone, sparse: a wire basis function loses power only
        on its own two segments.
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
    diagonal = (resistance * (left + right).abs() / 3).to(torch.complex128)
    copper = _diagonal_sparse(diagonal)
    return matrix + torch.diag(diagonal), copper


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
        The resistive part of what was added, sparse.
    """
    loaded = matrix.clone()
    loss = torch.zeros(coil.n_dof, dtype=matrix.dtype, device=matrix.device)
    by_tag = {port.tag: port for port in coil.ports}
    for port in coil.ports:
        if port.kind != "element":
            continue
        value, resistance = port.impedance(angular_frequency)
        first, second = port.dofs.tolist()
        for dof in (first, second):
            loaded[dof, dof] += 0.5 * value
            loss[dof] += 0.5 * resistance
        if port.load == "mutual_inductor" and port.coupled_tag is not None:
            partner = by_tag[port.coupled_tag].dofs.tolist()
            mutual = 1j * angular_frequency * port.coupled_value
            loaded[first, partner[0]] += 0.5 * mutual
            loaded[second, partner[1]] += 0.5 * mutual
    return loaded, _diagonal_sparse(loss)


def port_excitation(coil: WireCoil) -> torch.Tensor:
    """Drive each port with half its voltage on each of its two basis functions.

    Ported from ``excitation_wire.m``.

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


# --------------------------------------------------------------------------
# A wire coil and a surface together
# --------------------------------------------------------------------------


def surface_coupling(
    coil: WireCoil,
    surface: SurfaceCoil,
    medium: Medium,
    *,
    order: int = 6,
    triangle_order: int = 4,
    chunk: int = 1 << 22,
) -> torch.Tensor:
    """Assemble the EFIE interaction of a wire with a surface it does not touch.

    Takes the place of MARIE's ``Assembly_WSIE_block_par.m``, which MARIE
    compresses by adaptive cross approximation. The integrand is the one both
    self-matrices use, ``jk f . f' + (1/jk) div f div f'`` against
    ``exp(-jkR)/R``, with the wire's triangle basis and the surface's RWG
    basis in the surface matrix's own scaling; MARIE's row assembly
    (``assembly_wire_surf_ns_row.m``) holds the hat at one half on both
    segments and gives both segments' charge the same sign, which neither
    self-matrix does.

    Parameters
    ----------
    coil
        The wire.
    surface
        The surface: a surface coil or a shield.
    medium
        Free-space constants at the working frequency.
    order
        Gauss points per wire segment, MARIE's ``Quad_order_wie``.
    triangle_order
        Points per axis of the Gauss rule on each triangle.
    chunk
        Kernel evaluations held at a time.

    Returns
    -------
    torch.Tensor
        Shape ``(coil.n_dof, surface.n_dof)``, complex, with the sign and scale
        of :attr:`mariepy.sie.CoilSystem.impedance`, as MARIE's ``Zcw`` has.
    """
    k = medium.wavenumber
    s = _segments(coil, order)
    device = coil.device
    weights, barycentric = gauss_triangle(triangle_order, device=device)
    weights = 0.5 * weights.to(torch.complex128)

    vertices = surface.mesh.vertices()  # (t, 3, 3)
    points = torch.einsum("pu,tuc->tpc", barycentric, vertices)  # (t, p, 3)
    arms = points[:, None, :, :] - vertices[:, :, None, :]  # (t, 3, p, 3)
    scale = (surface.mesh.edge_lengths() * surface.signs).to(torch.complex128)
    dof = surface.dof_of_triangle()

    # Wire side: every quadrature point of every basis function.
    wire_points = s.points.reshape(coil.n_dof, -1, 3)  # (w, 2q, 3)
    current = (s.shape[None, :, :, None] * s.tangent[:, :, None, :]).reshape(
        coil.n_dof, -1, 3
    )
    charge = (
        s.slope[:, :, None].expand(-1, -1, s.weights.numel()).reshape(coil.n_dof, -1)
    )
    line = ((s.length[:, :, None] / 2) * s.weights).reshape(coil.n_dof, -1)

    matrix = torch.zeros(
        (coil.n_dof, surface.n_dof), dtype=torch.complex128, device=device
    )
    n_triangles = surface.mesh.n_triangles
    per_block = max(
        1, chunk // max(1, points.shape[1] * wire_points.shape[1] * n_triangles)
    )
    for start in range(0, coil.n_dof, per_block):
        rows = slice(start, min(start + per_block, coil.n_dof))
        separation = (
            wire_points[rows, None, :, None, :] - points[None, :, None, :, :]
        )  # (w, t, 2q, p, 3)
        distance = torch.linalg.vector_norm(separation, dim=-1)
        green = torch.exp(-1j * k * distance) / distance * weights  # (w, t, 2q, p)
        vector = torch.einsum(
            "wtqp,wqc,tapc,wq->wta",
            green,
            current[rows].to(torch.complex128),
            arms.to(torch.complex128),
            line[rows].to(torch.complex128),
        )
        scalar = torch.einsum(
            "wtqp,wq->wt",
            green,
            (charge[rows] * line[rows]).to(torch.complex128),
        )
        block = (1j * k * vector + (2.0 / (1j * k)) * scalar[:, :, None]) * scale
        column = dof[None, :, :].expand(block.shape[0], -1, -1)
        row = torch.arange(start, rows.stop, device=device)[:, None, None].expand_as(
            column
        )
        keep = column >= 0
        matrix.index_put_((row[keep], column[keep]), block[keep], accumulate=True)
    return -(medium.impedance / (4.0 * math.pi)) * matrix


@dataclass(frozen=True)
class CombinedCoil:
    """A wire coil and a surface coil solved together, wire unknowns first.

    Attributes
    ----------
    wire
        The wire coil.
    surface
        The surface coil.
    """

    wire: WireCoil
    surface: SurfaceCoil

    @property
    def n_dof(self) -> int:
        """Number of unknowns, the wire's then the surface's."""
        return self.wire.n_dof + self.surface.n_dof

    @property
    def n_driven(self) -> int:
        """Number of driven ports, the wire's then the surface's."""
        return self.wire.n_driven + self.surface.n_driven

    @property
    def ports(self) -> tuple[Port, ...]:
        """The wire's ports, then the surface's."""
        return self.wire.ports + self.surface.ports

    @property
    def device(self) -> torch.device:
        """Device the geometry lives on."""
        return self.wire.device


def assemble_combined(
    coil: CombinedCoil,
    medium: Medium,
    *,
    order: int = 6,
    triangle_order: int = 4,
) -> CoilSystem:
    """Build the system of a wire coil and a surface coil together.

    Ported from the wire-and-surface branch of ``wsvie_assembly.m``: the two
    self-systems on the diagonal, the wire-surface interaction and its
    transpose off it, and the ports in MARIE's order, the wire's first.

    Parameters
    ----------
    coil
        The two coils.
    medium
        Free-space constants at the working frequency.
    order
        Gauss points per wire segment.
    triangle_order
        Points per axis of the Gauss rule on each triangle of the cross block.

    Returns
    -------
    CoilSystem
        Over the wire's unknowns, then the surface's.
    """
    from mariepy.sie import assemble as assemble_surface

    wire_system = assemble(coil.wire, medium, order=order)
    surface_system = assemble_surface(coil.surface, medium)
    cross = surface_coupling(
        coil.wire, coil.surface, medium, order=order, triangle_order=triangle_order
    )
    n_wire, n_surface = coil.wire.n_dof, coil.surface.n_dof

    def blocks(first, second, off):
        return torch.cat(
            [
                torch.cat([first, off], dim=1),
                torch.cat([off.transpose(0, 1), second], dim=1),
            ],
            dim=0,
        )

    excitation = torch.cat(
        [
            torch.nn.functional.pad(wire_system.excitation, (0, n_surface)),
            torch.nn.functional.pad(surface_system.excitation, (n_wire, 0)),
        ],
        dim=0,
    )
    return CoilSystem(
        impedance=blocks(wire_system.impedance, surface_system.impedance, cross),
        excitation=excitation,
        copper_loss=block_diagonal(wire_system.copper_loss, surface_system.copper_loss),
        lumped_loss=block_diagonal(wire_system.lumped_loss, surface_system.lumped_loss),
    )
