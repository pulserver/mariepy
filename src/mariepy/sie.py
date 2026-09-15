"""The coil impedance matrix: the EFIE in the RWG basis, its loads and its drive.

Ported from MARIE 3.0's ``src_integral_equations/src_sie/sie_assembly.m`` and
``src_operators_sie/Assembly_SIE_par.m``, ``assembly_ns_par.m``,
``assembly_{st,ea,va}_par.m``, ``assembly_le.m`` and ``excitation_coil.m``.

The Galerkin entry of the electric field integral equation for basis functions
``f_a`` and ``f_b`` is

``Z_ab = (eta0 / 4 pi) [ j k <f_a, G f_b> + (1 / j k) <div f_a, G div f_b> ]``

with ``G(R) = exp(-j k R) / R``. Four blocks build it, by how the two triangles
of a pair meet: disjoint triangles take a tensor-product Gauss rule on each,
and triangles that share a vertex, an edge or everything take the DIRECTFN
singular integrals. Each block is assembled for one ordering of every pair, and
the matrix is completed by its own transpose, so a block's reciprocity is a
statement about the quadrature rather than about the assembly.

Every block routine returns local three-by-three blocks indexed by
``(pair, observer local edge, source local edge)``, carrying the edge lengths
and the signs but not the ``eta0 / 4 pi`` in front.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mariepy._accelerators import require
from mariepy.coil import SurfaceCoil, _shared_vertices
from mariepy.constants import Medium
from mariepy.quadrature import gauss_legendre_1d, gauss_triangle

__all__ = [
    "CoilSystem",
    "assemble",
    "edge_block",
    "impedance",
    "lumped_loads",
    "near_block",
    "plane_wave_excitation",
    "port_excitation",
    "self_block",
    "vertex_block",
]

_DIRECTFN = "mariepy._directfn"

_ROW_CHUNK = 128
_CHUNK_ELEMENTS = 1 << 22

# The moments of the barycentric coordinates over a triangle: the integral of
# lambda_u lambda_v over a triangle of area A is A (1 + delta_uv) / 12.
_MOMENT_DIAGONAL = 2.0
_MOMENT_OFF = 1.0


@dataclass(frozen=True)
class CoilSystem:
    """The coil's method-of-moments system at one frequency.

    Attributes
    ----------
    impedance
        MARIE's ``SIE.Z``: minus the EFIE matrix with the lumped loads added,
        shape ``(n_dof, n_dof)``, complex.
    excitation
        Delta-gap drive of each port, shape ``(n_driven, n_dof)``, complex.
    copper_loss
        The part of the self block that the conductor's surface resistance
        contributes, same shape as ``impedance``.
    lumped_loss
        The resistive part of the lumped loads, same shape.
    """

    impedance: torch.Tensor
    excitation: torch.Tensor
    copper_loss: torch.Tensor
    lumped_loss: torch.Tensor


def assemble(
    coil: SurfaceCoil,
    medium: Medium,
    *,
    near_order: int = 4,
    self_order: int = 10,
    edge_orders: tuple[int, int] = (10, 10),
    vertex_orders: tuple[int, int, int] = (6, 6, 6),
    surface_resistance: float | None = None,
) -> CoilSystem:
    """Build the coil's system matrix, its losses and its port drive.

    Parameters
    ----------
    coil
        The coil, its basis and its ports.
    medium
        Free-space constants at the working frequency.
    near_order
        Points per axis of the Gauss rule on each triangle of a disjoint pair.
    self_order
        Points of the DIRECTFN rule for a triangle against itself.
    edge_orders
        Points of the theta and psi rules for an edge-adjacent pair.
    vertex_orders
        Points of the two theta rules and the psi rule for a vertex-adjacent
        pair.
    surface_resistance
        Surface resistance of the conductor in ohm per square. Defaults to
        copper at this frequency; pass 0 for a perfect conductor.

    Returns
    -------
    CoilSystem
        The system MARIE's ``sie_assembly.m`` hands to the solver.
    """
    resistance = (
        medium.surface_resistance if surface_resistance is None else surface_resistance
    )
    matrix, copper = impedance(
        coil,
        medium,
        near_order=near_order,
        self_order=self_order,
        edge_orders=edge_orders,
        vertex_orders=vertex_orders,
        surface_resistance=resistance,
    )
    loaded, lumped = lumped_loads(matrix, coil, medium.angular_frequency)
    return CoilSystem(
        impedance=-loaded,
        excitation=port_excitation(coil),
        copper_loss=copper,
        lumped_loss=lumped,
    )


def impedance(
    coil: SurfaceCoil,
    medium: Medium,
    *,
    near_order: int = 4,
    self_order: int = 10,
    edge_orders: tuple[int, int] = (10, 10),
    vertex_orders: tuple[int, int, int] = (6, 6, 6),
    surface_resistance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the EFIE matrix of the coil.

    Parameters
    ----------
    coil
        The coil and its basis.
    medium
        Free-space constants at the working frequency.
    near_order, self_order, edge_orders, vertex_orders
        Quadrature orders of the four blocks, as in :func:`assemble`.
    surface_resistance
        Surface resistance of the conductor in ohm per square.

    Returns
    -------
    matrix : torch.Tensor
        Shape ``(n_dof, n_dof)``, complex.
    copper_loss : torch.Tensor
        The surface-resistance part of it, same shape.
    """
    wavenumber = medium.wavenumber
    device = coil.mesh.device
    matrix = torch.zeros(
        (coil.n_dof, coil.n_dof), dtype=torch.complex128, device=device
    )

    for pairs in _disjoint_pairs(coil, near_order):
        _scatter(
            matrix, coil, pairs, near_block(coil, wavenumber, pairs, order=near_order)
        )
    _scatter(
        matrix,
        coil,
        coil.adjacency.edge,
        edge_block(coil, wavenumber, coil.adjacency.edge, orders=edge_orders),
    )
    _scatter(
        matrix,
        coil,
        coil.adjacency.vertex,
        vertex_block(coil, wavenumber, coil.adjacency.vertex, orders=vertex_orders),
    )
    matrix = matrix + matrix.T

    triangles = torch.arange(coil.mesh.n_triangles, device=device)
    same = torch.stack([triangles, triangles], dim=1)
    _scatter(matrix, coil, same, self_block(coil, wavenumber, order=self_order))

    matrix = matrix * (medium.impedance / (4.0 * torch.pi))
    copper = torch.zeros_like(matrix)
    if surface_resistance:
        block = surface_block(coil, surface_resistance)
        _scatter(matrix, coil, same, block)
        _scatter(copper, coil, same, block)
    return matrix, copper


def near_block(
    coil: SurfaceCoil, wavenumber: float, pairs: torch.Tensor, *, order: int = 4
) -> torch.Tensor:
    """Integrate disjoint triangle pairs with a tensor-product Gauss rule.

    Parameters
    ----------
    coil
        The coil and its basis.
    wavenumber
        Free-space wavenumber in rad/m.
    pairs
        Triangle index pairs, shape ``(n_pairs, 2)``, observer then source.
    order
        Points per axis; the rule on each triangle has ``order ** 2`` points.

    Returns
    -------
    torch.Tensor
        Shape ``(n_pairs, 3, 3)``, complex.
    """
    device = coil.mesh.device
    weights, points = gauss_triangle(order, device=device)
    weights = 0.5 * weights.to(torch.complex128)

    vertices = coil.mesh.vertices()
    observer = vertices[pairs[:, 0]]
    source = vertices[pairs[:, 1]]
    quadrature_observer = torch.einsum("pu,nuc->npc", points, observer)
    quadrature_source = torch.einsum("qv,nvc->nqc", points, source)

    separation = quadrature_observer[:, :, None, :] - quadrature_source[:, None, :, :]
    distance = torch.linalg.vector_norm(separation, dim=-1)
    green = torch.exp(-1j * wavenumber * distance) / distance

    weighted = weights[None, :, None] * green * weights[None, None, :]
    scalar = weighted.sum(dim=(1, 2))
    moments = torch.einsum(
        "npq,pu,qv->nuv",
        weighted,
        points.to(torch.complex128),
        points.to(torch.complex128),
    )

    arms_observer = observer[:, None, :, :] - observer[:, :, None, :]
    arms_source = source[:, None, :, :] - source[:, :, None, :]
    vector = torch.einsum(
        "nuv,nauc,nbvc->nab",
        moments,
        arms_observer.to(torch.complex128),
        arms_source.to(torch.complex128),
    )

    kernel = (
        1j * wavenumber * vector + (4.0 / (1j * wavenumber)) * scalar[:, None, None]
    )
    return _weigh(coil, pairs, kernel)


def edge_block(
    coil: SurfaceCoil,
    wavenumber: float,
    pairs: torch.Tensor,
    *,
    orders: tuple[int, int] = (6, 6),
) -> torch.Tensor:
    """Integrate edge-adjacent triangle pairs with DIRECTFN.

    Parameters
    ----------
    coil
        The coil and its basis.
    wavenumber
        Free-space wavenumber in rad/m.
    pairs
        Triangle index pairs that share two vertices, shape ``(n_pairs, 2)``.
    orders
        Points of the theta and psi rules.

    Returns
    -------
    torch.Tensor
        Shape ``(n_pairs, 3, 3)``, complex.
    """
    directfn = require(module=_DIRECTFN)
    rules = [gauss_legendre_1d(count) for count in orders]
    arguments = [part.numpy() for rule in rules for part in rule]

    def integrate(observer, source, permutation_observer, permutation_source):
        vertices = torch.cat(
            [observer[permutation_observer], source[permutation_source[2]][None]]
        )
        return directfn.triangle_edge(vertices.numpy(), wavenumber, *arguments)

    return _singular(coil, pairs, _edge_permutations, integrate)


def vertex_block(
    coil: SurfaceCoil,
    wavenumber: float,
    pairs: torch.Tensor,
    *,
    orders: tuple[int, int, int] = (6, 6, 6),
) -> torch.Tensor:
    """Integrate vertex-adjacent triangle pairs with DIRECTFN.

    Parameters
    ----------
    coil
        The coil and its basis.
    wavenumber
        Free-space wavenumber in rad/m.
    pairs
        Triangle index pairs that share one vertex, shape ``(n_pairs, 2)``.
    orders
        Points of the two theta rules and the psi rule.

    Returns
    -------
    torch.Tensor
        Shape ``(n_pairs, 3, 3)``, complex.
    """
    directfn = require(module=_DIRECTFN)
    rules = [gauss_legendre_1d(count) for count in orders]
    arguments = [part.numpy() for rule in rules for part in rule]

    def integrate(observer, source, permutation_observer, permutation_source):
        vertices = torch.cat(
            [observer[permutation_observer], source[permutation_source[1:]]]
        )
        return directfn.triangle_vertex(vertices.numpy(), wavenumber, *arguments)

    return _singular(coil, pairs, _vertex_permutations, integrate)


def self_block(
    coil: SurfaceCoil, wavenumber: float, *, order: int = 10
) -> torch.Tensor:
    """Integrate each triangle against itself with DIRECTFN.

    Parameters
    ----------
    coil
        The coil and its basis.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Points of the DIRECTFN psi rule.

    Returns
    -------
    torch.Tensor
        Shape ``(n_triangles, 3, 3)``, complex.
    """
    directfn = require(module=_DIRECTFN)
    weights, nodes = gauss_legendre_1d(order)
    vertices = coil.mesh.vertices().cpu()

    values = torch.empty((coil.mesh.n_triangles, 3, 3), dtype=torch.complex128)
    for index in range(coil.mesh.n_triangles):
        raw = directfn.triangle_self(
            vertices[index].numpy(), wavenumber, weights.numpy(), nodes.numpy()
        )
        values[index] = torch.from_numpy(raw).reshape(3, 3)

    triangles = torch.arange(coil.mesh.n_triangles, device=coil.mesh.device)
    pairs = torch.stack([triangles, triangles], dim=1)
    signs = coil.signs.to(torch.complex128)
    return (
        values.to(coil.mesh.device)
        * signs[pairs[:, 0]][:, :, None]
        * signs[pairs[:, 1]][:, None, :]
    )


def surface_block(coil: SurfaceCoil, surface_resistance: float) -> torch.Tensor:
    """Integrate the conductor's surface impedance over each triangle.

    The surface impedance contributes ``rho_s <f_a, f_b>`` over the triangle the
    two basis functions share. The overlap is exact: the integral of
    ``lambda_u lambda_v`` over a triangle of area ``A`` is ``A (1 + delta_uv) / 12``.

    Parameters
    ----------
    coil
        The coil and its basis.
    surface_resistance
        Surface resistance of the conductor in ohm per square.

    Returns
    -------
    torch.Tensor
        Shape ``(n_triangles, 3, 3)``, complex, already in ohms.
    """
    device = coil.mesh.device
    vertices = coil.mesh.vertices()
    arms = vertices[:, None, :, :] - vertices[:, :, None, :]
    moments = torch.full((3, 3), _MOMENT_OFF, dtype=torch.float64, device=device)
    moments.fill_diagonal_(_MOMENT_DIAGONAL)
    overlap = torch.einsum("uv,nauc,nbvc->nab", moments, arms, arms) / 12.0

    areas = coil.mesh.areas()
    scaling = surface_resistance / (4.0 * areas)
    triangles = torch.arange(coil.mesh.n_triangles, device=device)
    pairs = torch.stack([triangles, triangles], dim=1)
    return _weigh(coil, pairs, (scaling[:, None, None] * overlap).to(torch.complex128))


def lumped_loads(
    matrix: torch.Tensor, coil: SurfaceCoil, angular_frequency: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the lumped elements to the coil matrix.

    A lumped element spans the degrees of freedom of its own edges, and the
    current through it is the sum of theirs weighted by edge length, which is
    why its impedance enters as an outer product of those lengths.

    Parameters
    ----------
    matrix
        The EFIE matrix, shape ``(n_dof, n_dof)``.
    coil
        The coil and its ports.
    angular_frequency
        Working frequency in rad/s.

    Returns
    -------
    loaded : torch.Tensor
        The matrix with every lumped element added.
    loss : torch.Tensor
        The resistive part of what was added.
    """
    loaded = matrix.clone()
    loss = torch.zeros_like(matrix)
    lengths = coil.dof_lengths().to(torch.complex128)
    by_tag = {port.tag: port for port in coil.ports}

    for port in coil.ports:
        if port.kind != "element":
            continue
        value, resistance = port.impedance(angular_frequency)
        here = lengths[port.dofs]
        rows, columns = torch.meshgrid(port.dofs, port.dofs, indexing="ij")
        loaded[rows, columns] += value * here[:, None] * here[None, :]
        loss[rows, columns] += resistance * here[:, None] * here[None, :]

        if port.load == "mutual_inductor" and port.coupled_tag is not None:
            partner = by_tag[port.coupled_tag]
            there = lengths[partner.dofs]
            rows, columns = torch.meshgrid(port.dofs, partner.dofs, indexing="ij")
            loaded[rows, columns] += (
                1j
                * angular_frequency
                * port.coupled_value
                * here[:, None]
                * there[None, :]
            )
    return loaded, loss


def port_excitation(coil: SurfaceCoil) -> torch.Tensor:
    """Build the delta-gap drive of each driven port.

    Parameters
    ----------
    coil
        The coil and its ports.

    Returns
    -------
    torch.Tensor
        Shape ``(n_driven, n_dof)``, complex: each row drives one port's edges
        with minus its own length times the port voltage.
    """
    lengths = coil.dof_lengths().to(torch.complex128)
    driven = [port for port in coil.ports if port.kind == "port"]
    excitation = torch.zeros(
        (len(driven), coil.n_dof), dtype=torch.complex128, device=lengths.device
    )
    for row, port in enumerate(driven):
        excitation[row, port.dofs] = -lengths[port.dofs] * port.voltage
    return excitation


def plane_wave_excitation(
    coil: SurfaceCoil,
    medium: Medium,
    *,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    polarisation: tuple[float, float, float] = (1.0, 0.0, 0.0),
    amplitude: complex = 1.0,
    order: int = 6,
) -> torch.Tensor:
    """Test a plane wave against the basis, in the convention of the port drive.

    A plane wave drives no port, so it is not how a coil is used; it is how the
    coil matrix is checked, because a closed conductor in a plane wave has an
    analytic scattering cross section.

    Parameters
    ----------
    coil
        The coil and its basis.
    medium
        Supplies the wavenumber.
    direction
        Propagation direction; normalised here.
    polarisation
        Electric field direction. Its component along ``direction`` is removed,
        since a plane wave is transverse.
    amplitude
        Field strength in V/m.
    order
        Points per axis of the Gauss rule on each triangle.

    Returns
    -------
    torch.Tensor
        Shape ``(n_dof,)``, complex: minus the tested incident field, so that
        solving with :attr:`CoilSystem.impedance` gives the induced current.

    Raises
    ------
    ValueError
        The direction is zero, or the polarisation is parallel to it.
    """
    device = coil.mesh.device
    unit = torch.tensor(direction, dtype=torch.float64, device=device)
    norm = torch.linalg.vector_norm(unit)
    if norm == 0:
        raise ValueError("a plane wave needs a direction of propagation")
    unit = unit / norm

    field = torch.tensor(polarisation, dtype=torch.float64, device=device)
    field = field - (field @ unit) * unit
    strength = torch.linalg.vector_norm(field)
    if strength == 0:
        raise ValueError(
            "a plane wave is transverse, so it needs a polarisation across it"
        )
    field = amplitude * (field / strength).to(torch.complex128)

    weights, points = gauss_triangle(order, device=device)
    weights = (0.5 * weights).to(torch.complex128)
    vertices = coil.mesh.vertices()
    quadrature = torch.einsum("pu,nuc->npc", points, vertices)
    phase = torch.exp(-1j * medium.wavenumber * (quadrature @ unit))
    arms = (quadrature[:, :, None, :] - vertices[:, None, :, :]).to(torch.complex128)

    tested = torch.einsum("p,npac,np,c->na", weights, arms, phase, field)
    tested = tested * (coil.mesh.edge_lengths() * coil.signs).to(torch.complex128)

    dof = coil.dof_of_triangle()
    keep = dof >= 0
    excitation = torch.zeros(coil.n_dof, dtype=torch.complex128, device=device)
    excitation.index_put_((dof[keep],), -tested[keep], accumulate=True)
    return excitation


def _weigh(
    coil: SurfaceCoil, pairs: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor:
    """Multiply a local block by the edge lengths and signs of its two triangles."""
    lengths = coil.mesh.edge_lengths().to(torch.complex128)
    signs = coil.signs.to(torch.complex128)
    observer = (lengths * signs)[pairs[:, 0]]
    source = (lengths * signs)[pairs[:, 1]]
    return kernel * observer[:, :, None] * source[:, None, :]


def _edge_permutations(
    observer: list[int], source: list[int]
) -> tuple[list[int], list[int]]:
    """Order both triangles of an edge-adjacent pair as DIRECTFN expects them."""
    shared = [slot for slot, node in enumerate(observer) if node in source]
    free = [slot for slot, node in enumerate(observer) if node not in source]
    order_observer = shared + free
    at = {node: slot for slot, node in enumerate(source)}
    first = at[observer[order_observer[1]]]
    second = at[observer[order_observer[0]]]
    return order_observer, [first, second, 3 - first - second]


def _vertex_permutations(
    observer: list[int], source: list[int]
) -> tuple[list[int], list[int]]:
    """Order both triangles of a vertex-adjacent pair as DIRECTFN expects them."""
    order_observer = [slot for slot, node in enumerate(observer) if node in source]
    order_observer += [slot for slot, node in enumerate(observer) if node not in source]
    order_source = [slot for slot, node in enumerate(source) if node in observer]
    order_source += [slot for slot, node in enumerate(source) if node not in observer]
    return order_observer, order_source


def _singular(coil, pairs, permutations, integrate) -> torch.Tensor:
    """Run a DIRECTFN routine over triangle pairs and undo its vertex ordering."""
    vertices = coil.mesh.vertices().cpu()
    triangles = coil.mesh.triangles.cpu().tolist()
    signs = coil.signs.to(torch.complex128).cpu()

    values = torch.zeros((pairs.shape[0], 3, 3), dtype=torch.complex128)
    for row, (here, there) in enumerate(pairs.cpu().tolist()):
        order_observer, order_source = permutations(triangles[here], triangles[there])
        raw = integrate(
            vertices[here],
            vertices[there],
            torch.tensor(order_observer),
            torch.tensor(order_source),
        )
        block = torch.from_numpy(raw).reshape(3, 3)
        rows = torch.tensor(order_observer)
        columns = torch.tensor(order_source)
        values[row, rows[:, None], columns[None, :]] = block
        values[row] *= signs[here][:, None] * signs[there][None, :]
    return values.to(coil.mesh.device)


def _disjoint_pairs(coil: SurfaceCoil, order: int):
    """Yield the triangle pairs that do not touch, in chunks a Gauss rule fits in."""
    n_triangles = coil.mesh.n_triangles
    device = coil.mesh.device
    budget = max(1, _CHUNK_ELEMENTS // order**4)
    for start in range(0, n_triangles, _ROW_CHUNK):
        rows = torch.arange(start, min(start + _ROW_CHUNK, n_triangles), device=device)
        shared = _shared_vertices(coil.mesh.triangles, rows)
        below = torch.arange(n_triangles, device=device)[None, :] < rows[:, None]
        here, there = torch.nonzero((shared == 0) & below, as_tuple=True)
        pairs = torch.stack([rows[here], there], dim=1)
        for begin in range(0, pairs.shape[0], budget):
            yield pairs[begin : begin + budget]


def _scatter(
    matrix: torch.Tensor, coil: SurfaceCoil, pairs: torch.Tensor, blocks: torch.Tensor
) -> None:
    """Accumulate local blocks into the matrix, dropping edges that carry no basis."""
    if pairs.numel() == 0:
        return
    dof = coil.dof_of_triangle()
    rows = dof[pairs[:, 0]][:, :, None].expand(-1, 3, 3)
    columns = dof[pairs[:, 1]][:, None, :].expand(-1, 3, 3)
    keep = (rows >= 0) & (columns >= 0)
    matrix.index_put_((rows[keep], columns[keep]), blocks[keep], accumulate=True)
