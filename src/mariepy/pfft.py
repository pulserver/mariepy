"""Precorrected FFT coupling between a coil and a voxel body.

Ported from MARIE 3.0's ``src_integral_equations/src_wsvie/src_pfft``:
``src_svie_pfft/pfft_surface_domain.m``,
``pfft_proj_surface_create_near_lists.m``,
``pfft_projection_surface_assembly.m``, ``pfft_surface_assemble_direct_bc.m``,
``src_pfft_coil/pfft_assemble_voxel_bc.m``, ``pfft_assemble_voxel_cc.m`` and
the supporting files under ``src_pfft_supporting``; for a wire coil, from
``src_wvie_pfft/pfft_wire_domain.m``, ``pfft_proj_wire_create_near_lists.m``,
``pfft_projection_wire_assembly.m`` and ``pfft_wire_assemble_direct_bc.m``,
which differ from the surface files only in where a basis function sits, how
wide it is and which kernel gives its field.

The coil and the body are never coupled by a dense matrix. Every coil basis
function is replaced by cell currents on a three-by-three-by-three block of an
extended grid that carries the body as well, chosen so that those cell currents
radiate the basis function's own field on a sphere of collocation points around
it. One convolution of the whole grid with the body kernel then carries every
interaction at once. Close in, where a block of cells is a poor stand-in for a
surface current, the projected interaction is subtracted and the true one added
back -- one sparse correction per operator, which is what the method's name
refers to.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass

import torch

from mariepy import coupling, vie
from mariepy import wire as wire_module
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.quadrature import lebedev_26_directions
from mariepy.tucker import circulant_tucker

__all__ = [
    "Coupling",
    "ExtendedGrid",
    "NearLists",
    "assemble",
    "coil_precorrection",
    "coupling_rows",
    "direct_coupling",
    "expansion_response",
    "extended_domain",
    "near_lists",
    "projected_coupling",
    "projection",
    "projection_matrix",
    "restrict",
    "scatter_matrix",
    "widest_basis",
]

# MARIE fixes the expansion block at three cells a side in
# ``pfft_surface_domain.m``; the code below carries it as a parameter but the
# collocation sphere and the near distance are both sized from it.
EXPANSION = 3

# A basis function is replaced by currents on the expansion block, which reaches
# one cell either side of its centre. Past this many cells across, the widest
# basis function no longer sits on its block, and the coupled system takes
# markedly more GMRES iterations: on a head at 4 mm the same coil took 58
# iterations meshed at 2.8 cells and 132 at 5.2.
WIDEST_BASIS = 3.0

Coil = SurfaceCoil | wire_module.WireCoil | wire_module.CombinedCoil


def _nodes(coil: Coil) -> torch.Tensor:
    """Every point the coil reaches."""
    if isinstance(coil, wire_module.CombinedCoil):
        return torch.cat([_nodes(coil.wire), _nodes(coil.surface)])
    if isinstance(coil, wire_module.WireCoil):
        return coil.points()
    return coil.mesh.nodes


def _anchors(coil: Coil) -> torch.Tensor:
    """Where each basis function sits: its edge's midpoint, or its wire node."""
    if isinstance(coil, wire_module.CombinedCoil):
        return torch.cat([_anchors(coil.wire), _anchors(coil.surface)])
    if isinstance(coil, wire_module.WireCoil):
        return coil.centre
    nodes = coil.mesh.nodes
    edges = coil.edges[coil.edge_of_dof]
    return 0.5 * (nodes[edges[:, 0]] + nodes[edges[:, 1]])


def _width_in_cells(coil: Coil, resolution: float) -> int:
    """Return the basis functions' mean width in cells, as MARIE sizes the near cube.

    A surface basis function is as wide as its widest triangle edge
    (``pfft_surface_domain.m``); a wire one as its rising segment, rounded up
    before the mean (``pfft_wire_domain.m``); for both together, the wider of
    the two (``pfft_wire_surface_domain.m``).
    """
    if isinstance(coil, wire_module.CombinedCoil):
        return max(
            _width_in_cells(coil.wire, resolution),
            _width_in_cells(coil.surface, resolution),
        )
    if isinstance(coil, wire_module.WireCoil):
        return math.ceil(float(torch.ceil(coil.left_lengths() / resolution).mean()))
    lengths = coil.mesh.edge_lengths()
    carried = coil.dof_of_triangle() >= 0
    widest = torch.zeros(coil.n_dof, dtype=lengths.dtype, device=lengths.device)
    triangle, local = torch.nonzero(carried, as_tuple=True)
    widest.scatter_reduce_(
        0,
        coil.dof_of_triangle()[triangle, local],
        lengths[triangle].max(dim=1).values,
        reduce="amax",
    )
    return math.ceil(float((widest / resolution).mean()))


def widest_basis(coil: Coil, resolution: float) -> float:
    """Return the widest basis function's span, in cells of a grid.

    Parameters
    ----------
    coil
        The coil and its basis.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    float
        The longest triangle edge of a surface coil, or the longest hat of a
        wire one, divided by the pitch.
    """
    if isinstance(coil, wire_module.CombinedCoil):
        return max(
            widest_basis(coil.wire, resolution),
            widest_basis(coil.surface, resolution),
        )
    if isinstance(coil, wire_module.WireCoil):
        widest = float((coil.left_lengths() + coil.right_lengths()).max())
    else:
        widest = float(coil.mesh.edge_lengths().max())
    return widest / resolution


def _field(
    coil: Coil,
    dofs: torch.Tensor,
    points: torch.Tensor,
    medium: Medium,
    *,
    magnetic: bool,
    order: int,
    **arguments,
) -> torch.Tensor:
    """Return the field each named basis function puts on its observer."""
    if isinstance(coil, wire_module.CombinedCoil):
        n_wire = coil.wire.n_dof
        on_wire = dofs < n_wire
        out = torch.zeros(
            (dofs.numel(), 3), dtype=torch.complex128, device=points.device
        )
        for part, chosen, shift in (
            (coil.wire, on_wire, 0),
            (coil.surface, ~on_wire, n_wire),
        ):
            if bool(chosen.any()):
                out[chosen] = _field(
                    part,
                    dofs[chosen] - shift,
                    points[chosen],
                    medium,
                    magnetic=magnetic,
                    order=order,
                    **arguments,
                )
        return out
    if isinstance(coil, wire_module.WireCoil):
        kernel = wire_module.coupling_k if magnetic else wire_module.coupling_n
        return kernel(coil, dofs, points, medium, order=order, **arguments)
    kernel = coupling.coupling_k if magnetic else coupling.coupling_n
    corners = coil.rwg_vertices()
    return kernel(corners[dofs], points, medium, triangle_order=order, **arguments)


@dataclass(frozen=True)
class ExtendedGrid:
    """A uniform grid covering both the body and the coil.

    Attributes
    ----------
    shape
        Cells along each axis.
    resolution
        Cell pitch in metres.
    origin
        Coordinates of cell ``(0, 0, 0)``.
    mask
        Where the body sits in this grid, shape ``shape``, boolean.
    body_origin
        Index of the body grid's cell ``(0, 0, 0)`` in this grid.
    """

    shape: tuple[int, int, int]
    resolution: float
    origin: tuple[float, float, float]
    mask: torch.Tensor
    body_origin: tuple[int, int, int]

    @property
    def n_cells(self) -> int:
        """Number of cells in the grid."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def device(self) -> torch.device:
        """Device the grid lives on."""
        return self.mask.device

    def centres(self, index: torch.Tensor) -> torch.Tensor:
        """Give the centre of each cell named by a triple of indices.

        Parameters
        ----------
        index
            Shape ``(..., 3)``, integer.

        Returns
        -------
        torch.Tensor
            Shape ``(..., 3)``, in metres.
        """
        origin = torch.tensor(self.origin, dtype=torch.float64, device=index.device)
        return origin + self.resolution * index.to(torch.float64)

    def flatten(self, index: torch.Tensor) -> torch.Tensor:
        """Turn a triple of indices into one index into the flattened grid.

        Parameters
        ----------
        index
            Shape ``(..., 3)``, integer.

        Returns
        -------
        torch.Tensor
            Shape ``(...,)``.
        """
        second, third = self.shape[1], self.shape[2]
        return (index[..., 0] * second + index[..., 1]) * third + index[..., 2]

    def body_cells(self) -> torch.Tensor:
        """Flat indices of the cells the body occupies, in the body's own order."""
        return torch.nonzero(self.mask.reshape(-1), as_tuple=False).flatten()


@dataclass(frozen=True)
class NearLists:
    """Which cells and which basis functions each basis function reaches.

    Attributes
    ----------
    centre
        Index triple of the cell each basis function sits in, shape
        ``(n_dof, 3)``.
    expansion
        Index triples of each basis function's expansion block, shape
        ``(n_dof, expansion ** 3, 3)``.
    span
        Cells a side of the cube the near corrections are computed on.
    distance
        Half-width, in cells, of the block of basis functions a basis function
        is precorrected against.
    """

    centre: torch.Tensor
    expansion: torch.Tensor
    span: int
    distance: int


@dataclass(frozen=True)
class Coupling:
    """The coupled operator's pieces, in the layout the matrix-vector product wants.

    Attributes
    ----------
    grid
        The extended grid.
    near
        The near lists.
    project
        Coil currents onto the extended grid, sparse ``(c * n_cells, n_dof)``,
        with ``c`` the unknowns per cell: 3 for the constant basis, 12 for the
        linear one.
    scatter
        Body currents onto the extended grid, sparse
        ``(c * n_cells, c * n_voxels)``.
    electric
        The near correction of the electric coupling, sparse
        ``(c * n_voxels, n_dof)``.
    magnetic
        The same for the magnetic coupling.
    coil
        The coil block's near correction, sparse ``(n_dof, n_dof)``.
    symbols_n, symbols_k
        The compressed kernels on the extended grid.
    linear
        Whether the body carries the piecewise-linear basis.
    """

    grid: ExtendedGrid
    near: NearLists
    project: torch.Tensor
    scatter: torch.Tensor
    electric: torch.Tensor
    magnetic: torch.Tensor
    coil: torch.Tensor
    symbols_n: tuple
    symbols_k: tuple
    linear: bool = False

    @property
    def n_components(self) -> int:
        """Unknowns per cell: 3 for the constant basis, 12 for the linear one."""
        return 12 if self.linear else 3


def extended_domain(
    body: VoxelBody, coil: Coil, *, expansion: int = EXPANSION
) -> ExtendedGrid:
    """Grow the body grid until it holds every basis function's expansion block.

    Parameters
    ----------
    body
        The body and its grid.
    coil
        The coil, whose nodes set how far the grid must reach.
    expansion
        Cells a side of the expansion block.

    Returns
    -------
    ExtendedGrid
        The grown grid, with the body's mask placed inside it.
    """
    half = (expansion - 1) // 2
    nodes = _nodes(coil)
    origin = torch.tensor(body.origin, dtype=torch.float64, device=nodes.device)
    reach = (nodes - origin) / body.resolution

    low, high, start = [], [], []
    for axis in range(3):
        first = min(0, math.floor(float(reach[:, axis].min())) - half)
        last = max(body.shape[axis] - 1, math.ceil(float(reach[:, axis].max())) + half)
        low.append(first)
        high.append(last)
        start.append(-first)

    shape = tuple(high[axis] - low[axis] + 1 for axis in range(3))
    mask = torch.zeros(shape, dtype=torch.bool, device=body.device)
    mask[
        start[0] : start[0] + body.shape[0],
        start[1] : start[1] + body.shape[1],
        start[2] : start[2] + body.shape[2],
    ] = body.mask
    return ExtendedGrid(
        shape=shape,
        resolution=body.resolution,
        origin=tuple(
            body.origin[axis] + low[axis] * body.resolution for axis in range(3)
        ),
        mask=mask,
        body_origin=(start[0], start[1], start[2]),
    )


def near_lists(
    grid: ExtendedGrid,
    coil: Coil,
    *,
    expansion: int = EXPANSION,
    distance: int | None = None,
) -> NearLists:
    """Place every basis function on the grid and size the near corrections.

    Parameters
    ----------
    grid
        The extended grid.
    coil
        The coil and its basis.
    expansion
        Cells a side of the expansion block.
    distance
        Half-width, in cells, of the near corrections. The default is MARIE's:
        1.6 times whichever is larger, the expansion block and a half or the
        mean basis function's own width in cells.

    Returns
    -------
    NearLists
        Where each basis function sits and how far its corrections reach.

    Raises
    ------
    ValueError
        If a basis function's expansion block would leave the grid.
    """
    centre_point = _anchors(coil)
    nodes = centre_point
    origin = torch.tensor(grid.origin, dtype=torch.float64, device=nodes.device)
    centre = torch.round((centre_point - origin) / grid.resolution).to(torch.int64)

    half = (expansion - 1) // 2
    steps = torch.arange(-half, half + 1, device=nodes.device)
    block = torch.cartesian_prod(steps, steps, steps)
    expansion_index = centre[:, None, :] + block[None, :, :]

    limits = torch.tensor(grid.shape, device=nodes.device)
    if bool((expansion_index < 0).any()) or bool((expansion_index >= limits).any()):
        raise ValueError("a basis function's expansion block leaves the extended grid")

    if distance is None:
        by_basis = _width_in_cells(coil, grid.resolution)
        by_cell = expansion + expansion // 2
        distance = math.ceil(1.6 * max(by_cell, by_basis))
    span = 2 * distance + 2 * (expansion // 2) + expansion % 2
    return NearLists(
        centre=centre, expansion=expansion_index, span=span, distance=distance
    )


def projection(
    grid: ExtendedGrid,
    coil: Coil,
    medium: Medium,
    near: NearLists,
    *,
    triangle_order: int = 4,
    cell_order: int = 2,
    linear: bool = False,
) -> torch.Tensor:
    """Replace each basis function by cell currents that radiate its own field.

    The cell currents of the expansion block are fitted, in the least-squares
    sense, to the field the basis function itself puts on a sphere of 26
    collocation points around the block.

    Parameters
    ----------
    grid
        The extended grid.
    coil
        The coil and its basis.
    medium
        Supplies the wavenumber.
    near
        Where each basis function sits.
    triangle_order
        Degree of the Dunavant rule on each triangle, or Gauss points per
        segment of a wire.
    cell_order
        Points per axis of the Gauss rule over each cell.
    linear
        Fit the cells' piecewise-linear currents rather than constant ones.

    Returns
    -------
    torch.Tensor
        Shape ``(n_dof, c, expansion ** 3)``, complex, with ``c`` 3 or 12: the
        current each expansion cell carries for a unit coefficient of the basis
        function.
    """
    device = grid.device
    offsets = (near.expansion[0] - near.centre[0]).to(torch.float64) * grid.resolution
    reach = ((near.span - 1) / 2 + 1) * grid.resolution
    collocation = reach * lebedev_26_directions(device=device)

    n_basis = 4 if linear else 1
    matrix = coupling.collocation_matrix(
        offsets,
        collocation,
        medium,
        cell_size=grid.resolution,
        cell_order=cell_order,
        n_basis=n_basis,
    )

    centres = grid.centres(near.centre)
    points = centres[:, None, :] + collocation[None, :, :]
    n_dof, n_points = coil.n_dof, collocation.shape[0]
    field = _field(
        coil,
        torch.arange(n_dof, device=device).repeat_interleave(n_points),
        points.reshape(-1, 3),
        medium,
        magnetic=False,
        order=triangle_order,
    ).reshape(n_dof, n_points, 3)

    # The collocation sphere gives 78 equations for 81 cell currents, or 324
    # with the linear basis, so the
    # system is underdetermined and a pseudo-inverse takes the smallest
    # solution. Which solution it is does not reach the answer: every one of
    # them radiates the same field, and the near correction subtracts back
    # exactly the one that was used. It is a pseudo-inverse rather than a
    # least-squares solve because CUDA's driver needs at least as many
    # equations as unknowns and this has fewer.
    right = field.permute(0, 2, 1).reshape(n_dof, 3 * n_points).transpose(0, 1)
    weights = torch.linalg.pinv(matrix) @ right
    n_block = near.expansion.shape[1]
    return weights.reshape(3 * n_basis, n_block, n_dof).permute(2, 0, 1).contiguous()


def projection_matrix(
    grid: ExtendedGrid, near: NearLists, weights: torch.Tensor
) -> torch.Tensor:
    """Spread the projection weights over the whole extended grid.

    Parameters
    ----------
    grid
        The extended grid.
    near
        Where each basis function's expansion block sits.
    weights
        As :func:`projection` returns them.

    Returns
    -------
    torch.Tensor
        Sparse ``(c * n_cells, n_dof)``, complex.
    """
    device = grid.device
    n_dof, n_components, n_block = weights.shape
    cells = grid.flatten(near.expansion)
    component = torch.arange(n_components, device=device)
    rows = (component[None, :, None] * grid.n_cells + cells[:, None, :]).reshape(-1)
    columns = torch.arange(n_dof, device=device).repeat_interleave(
        n_components * n_block
    )
    return torch.sparse_coo_tensor(
        torch.stack([rows, columns]),
        weights.reshape(-1),
        (n_components * grid.n_cells, n_dof),
        check_invariants=False,
    ).coalesce()


def scatter_matrix(grid: ExtendedGrid, n_components: int = 3) -> torch.Tensor:
    """Place the body's degrees of freedom on the extended grid.

    Parameters
    ----------
    grid
        The extended grid, whose mask fixes the order of the body's own
        degrees of freedom.
    n_components
        Unknowns per cell: 3 for the constant basis, 12 for the linear one.

    Returns
    -------
    torch.Tensor
        Sparse ``(c * n_cells, c * n_voxels)``, complex.
    """
    device = grid.device
    cells = grid.body_cells()
    n_voxels = cells.numel()
    component = torch.arange(n_components, device=device)
    rows = (component[:, None] * grid.n_cells + cells[None, :]).reshape(-1)
    columns = torch.arange(n_components * n_voxels, device=device)
    values = torch.ones(rows.numel(), dtype=torch.complex128, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([rows, columns]),
        values,
        (n_components * grid.n_cells, n_components * n_voxels),
        check_invariants=False,
    ).coalesce()


def direct_coupling(
    grid: ExtendedGrid,
    coil: Coil,
    medium: Medium,
    near: NearLists,
    *,
    triangle_order: int = 4,
    cell_order: int = 2,
    chunk: int = 4096,
    linear: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate each basis function against the body cells it comes close to.

    Ported from ``pfft_surface_assemble_direct_bc.m``. With the linear basis
    each cell takes the four basis terms of every component, ordered
    component-major, as MARIE stacks its twelve coupling sources.

    Parameters
    ----------
    grid
        The extended grid.
    coil
        The coil and its basis.
    medium
        Supplies the wavenumber.
    near
        Where each basis function sits and how far it reaches.
    triangle_order
        Degree of the Dunavant rule on each triangle.
    cell_order
        Points per axis of the Gauss rule over each cell.
    chunk
        Basis-function-and-cell pairs taken at a time.
    linear
        Couple to the cells' piecewise-linear basis.

    Returns
    -------
    electric : torch.Tensor
        Sparse ``(c * n_voxels, n_dof)``, complex.
    magnetic : torch.Tensor
        The same for the magnetic coupling.
    """
    volume = grid.resolution**3
    body_dof = body_numbering(grid)
    n_voxels = int(grid.mask.sum())
    terms = range(4) if linear else range(1)
    n_components = 3 * len(terms)
    component = torch.arange(n_components, device=grid.device)

    rows, columns, electric, magnetic_values = [], [], [], []
    for dof, cells in near_body_pairs(grid, near, chunk=chunk):
        points = grid.centres(unflatten(grid, cells))
        arguments = {
            "order": triangle_order,
            "cell_size": grid.resolution,
            "cell_order": cell_order,
        }
        for magnetic, out in ((False, electric), (True, magnetic_values)):
            stacked = torch.stack(
                [
                    _field(
                        coil,
                        dof,
                        points,
                        medium,
                        magnetic=magnetic,
                        basis_term=term,
                        **arguments,
                    )
                    for term in terms
                ],
                dim=-1,
            )
            out.append(volume * stacked.reshape(-1))
        rows.append(
            (component[None, :] * n_voxels + body_dof[cells][:, None]).reshape(-1)
        )
        columns.append(dof.repeat_interleave(n_components))

    shape = (n_components * n_voxels, coil.n_dof)
    index = torch.stack([torch.cat(rows), torch.cat(columns)])
    return (
        torch.sparse_coo_tensor(
            index, torch.cat(electric), shape, check_invariants=False
        ).coalesce(),
        torch.sparse_coo_tensor(
            index, torch.cat(magnetic_values), shape, check_invariants=False
        ).coalesce(),
    )


def coupling_rows(
    grid: ExtendedGrid,
    coil: Coil,
    medium: Medium,
    cells: torch.Tensor,
    *,
    dofs: torch.Tensor | None = None,
    triangle_order: int = 4,
    cell_order: int = 2,
    linear: bool = False,
    chunk: int = 1 << 20,
) -> torch.Tensor:
    """Integrate the coupling to named cells of a grid, densely, for every basis function.

    These are rows of ``Zbc``, taken as :func:`direct_coupling` takes its near
    entries and by the same quadrature, at whatever distance the cells lie: no
    projection and no convolution enter, so a row costs one kernel evaluation
    per basis function.

    Parameters
    ----------
    grid
        The extended grid.
    coil
        The coil and its basis.
    medium
        Supplies the wavenumber.
    cells
        Flat indices of the cells to take, as :func:`body_numbering` numbers
        the grid.
    dofs
        Which basis functions to take, or None for every one.
    triangle_order, cell_order
        Quadrature orders, as :func:`direct_coupling` takes them.
    linear
        Couple to the cells' piecewise-linear basis.
    chunk
        Basis-function-and-cell pairs taken at a time.

    Returns
    -------
    torch.Tensor
        Shape ``(c * len(cells), len(dofs))`` with ``c`` 3 or 12, complex, the
        rows running component-major over ``cells`` as the body's unknowns run
        over its own.
    """
    volume = grid.resolution**3
    terms = range(4) if linear else range(1)
    n_components = 3 * len(terms)
    points = grid.centres(unflatten(grid, cells))
    n_cells = points.shape[0]
    taken = torch.arange(coil.n_dof, device=grid.device) if dofs is None else dofs
    n_taken = int(taken.numel())
    out = torch.zeros(
        (3, len(terms), n_cells, n_taken),
        dtype=torch.complex128,
        device=grid.device,
    )
    arguments = {
        "order": triangle_order,
        "cell_size": grid.resolution,
        "cell_order": cell_order,
    }
    stride = max(1, chunk // max(1, n_cells))
    for start in range(0, n_taken, stride):
        width = min(stride, n_taken - start)
        paired = taken[start : start + width].repeat_interleave(n_cells)
        seen = points.repeat(width, 1)
        for index, term in enumerate(terms):
            field = _field(
                coil, paired, seen, medium, magnetic=False, basis_term=term, **arguments
            )
            field = field.reshape(width, n_cells, 3).permute(2, 1, 0)
            out[:, index, :, start : start + width] = volume * field
    return out.reshape(n_components * n_cells, n_taken)


def projected_coupling(
    grid: ExtendedGrid,
    coil: Coil,
    medium: Medium,
    near: NearLists,
    weights: torch.Tensor,
    response: tuple[torch.Tensor, torch.Tensor],
    *,
    block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the same interactions as the projection delivers them, to be subtracted.

    Parameters
    ----------
    grid
        The extended grid.
    coil
        The coil and its basis.
    medium
        Supplies nothing here; kept so the call reads beside
        :func:`direct_coupling`.
    near
        Where each basis function sits and how far it reaches.
    weights
        The projection weights.
    response
        The electric and magnetic response of one expansion block, from
        :func:`expansion_response`.
    block
        Basis functions taken at a time. Each one's field over its near cube is
        held while its pairs are read, so this sets the working memory: the
        piecewise-linear basis carries twelve components of ``span ** 3`` cells
        per basis function.

    Returns
    -------
    electric : torch.Tensor
        Sparse ``(c * n_voxels, n_dof)``, complex.
    magnetic : torch.Tensor
        The same for the magnetic coupling.
    """
    del medium
    responses = response
    n_components = responses[0].shape[1]
    flat_weights = weights.reshape(weights.shape[0], -1)
    body_dof = body_numbering(grid)
    n_voxels = int(grid.mask.sum())
    component = torch.arange(n_components, device=grid.device)
    reach = (near.span - 1) // 2
    n_cube = near.span**3

    rows, columns, electric, magnetic = [], [], [], []
    for begin, dof, cells in _near_body_blocks(grid, near, block):
        local = unflatten(grid, cells) - near.centre[dof] + reach
        where = (local[:, 0] * near.span + local[:, 1]) * near.span + local[:, 2]
        flat = (dof - begin) * n_cube + where
        run = flat_weights[begin : begin + block]
        for store, each in ((electric, responses[0]), (magnetic, responses[1])):
            field = (
                _projected_field(run, each).permute(0, 2, 1).reshape(-1, n_components)
            )
            store.append(field[flat].reshape(-1))
        rows.append(
            (component[None, :] * n_voxels + body_dof[cells][:, None]).reshape(-1)
        )
        columns.append(dof.repeat_interleave(n_components))

    shape = (n_components * n_voxels, coil.n_dof)
    index = torch.stack([torch.cat(rows), torch.cat(columns)])
    return (
        torch.sparse_coo_tensor(
            index, torch.cat(electric), shape, check_invariants=False
        ).coalesce(),
        torch.sparse_coo_tensor(
            index, torch.cat(magnetic), shape, check_invariants=False
        ).coalesce(),
    )


def coil_precorrection(
    coil: Coil,
    impedance: torch.Tensor,
    near: NearLists,
    weights: torch.Tensor,
    response: tuple[torch.Tensor, torch.Tensor],
    *,
    chunk: int = 4096,
    block: int = 256,
) -> torch.Tensor:
    """Undo the projection's share of the coil block, and put the true one back.

    Parameters
    ----------
    coil
        The coil and its basis.
    impedance
        The coil's own matrix, MARIE's ``SIE.Z``.
    near
        Where each basis function sits and how far it reaches.
    weights
        The projection weights.
    response
        As in :func:`projected_coupling`; only the electric part is used.
    chunk
        Pairs of basis functions taken at a time.
    block
        Basis functions whose field over the near cube is held at a time, as in
        :func:`projected_coupling`.

    Returns
    -------
    torch.Tensor
        Sparse ``(n_dof, n_dof)``, complex.
    """
    electric, _ = response
    n_components = electric.shape[1]
    flat_weights = weights.reshape(weights.shape[0], -1)
    reach = (near.span - 1) // 2
    n_cube = near.span**3
    n_dof = coil.n_dof

    separation = (
        (near.centre[:, None, :] - near.centre[None, :, :]).abs().max(dim=-1).values
    )
    observer, source = torch.nonzero(separation < near.distance, as_tuple=True)
    # Sorted by source, so each run of pairs needs only its own sources' fields.
    order = torch.argsort(source, stable=True)
    observer, source = observer[order], source[order]
    starts = list(range(0, n_dof, block))
    edges = torch.searchsorted(
        source, torch.tensor([*starts, n_dof], device=source.device)
    )

    projected = torch.empty(
        observer.shape[0], dtype=torch.complex128, device=weights.device
    )
    for begin, first, last in zip(starts, edges[:-1], edges[1:], strict=True):
        field = _projected_field(flat_weights[begin : begin + block], electric)
        field = field.permute(0, 2, 1).reshape(-1, n_components)
        for start in range(int(first), int(last), chunk):
            stop = min(start + chunk, int(last))
            o = observer[start:stop]
            q = source[start:stop]
            local = near.expansion[o] - near.centre[q][:, None, :] + reach
            where = (local[..., 0] * near.span + local[..., 1]) * near.span + local[
                ..., 2
            ]
            seen = field[(q - begin)[:, None] * n_cube + where]
            projected[start:stop] = torch.einsum("pbc,pcb->p", seen, weights[o])

    values = impedance[observer, source] - projected
    return torch.sparse_coo_tensor(
        torch.stack([observer, source]), values, (n_dof, n_dof), check_invariants=False
    ).coalesce()


def _projected_field(
    weights: torch.Tensor, response: torch.Tensor, *, chunk: int = 256
) -> torch.Tensor:
    """Give the field each basis function's expansion block puts on its near cube.

    Parameters
    ----------
    weights
        Shape ``(n_dof, c * expansion ** 3)``, the projection weights flattened.
    response
        Shape ``(c * expansion ** 3, c, span ** 3)``, one of the pair
        :func:`expansion_response` returns.
    chunk
        Basis functions taken at a time.

    Returns
    -------
    torch.Tensor
        Shape ``(n_dof, c, span ** 3)``.
    """
    return torch.cat(
        [
            torch.einsum("dj,jcn->dcn", weights[start : start + chunk], response)
            for start in range(0, weights.shape[0], chunk)
        ]
    )


def kernels(
    grid: ExtendedGrid,
    near: NearLists,
    medium: Medium,
    *,
    tol: float = 1e-7,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
    linear: bool = False,
) -> tuple:
    """Build the body kernels once and compress them for both grids they act on.

    The near correction subtracts what the convolution over the whole grid puts
    in, so the two must be the same kernel. One table, built at the larger of
    the two shapes and sliced, is what makes that so by construction.

    Parameters
    ----------
    grid
        The extended grid.
    near
        Fixes the near cube's size.
    medium
        Supplies the wavenumber.
    tol
        Relative tolerance of the Tucker compressions.
    far_order, medium_order, near_order
        Quadrature orders of the body kernels.
    linear
        Build the kernels of the piecewise-linear basis.

    Returns
    -------
    tuple
        Four sets of symbols: N and K on the extended grid, then N and K on the
        near cube.
    """
    shape = tuple(max(grid.shape[axis], near.span) for axis in range(3))
    orders = {
        "far_order": far_order,
        "medium_order": medium_order,
        "near_order": near_order,
        "linear": linear,
    }
    span = near.span
    pieces = []
    for build in (vie.kernel_n, vie.kernel_k):
        table = build(shape, grid.resolution, medium.wavenumber, **orders).to(
            grid.device
        )
        pieces.append(
            (
                circulant_tucker(
                    table[: grid.shape[0], : grid.shape[1], : grid.shape[2]], tol
                ),
                circulant_tucker(table[:span, :span, :span], tol),
            )
        )
    return pieces[0][0], pieces[1][0], pieces[0][1], pieces[1][1]


def expansion_response(
    grid: ExtendedGrid,
    near: NearLists,
    medium: Medium,
    symbols_n: tuple,
    symbols_k: tuple,
    *,
    n_components: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give the field one expansion block puts on every cell of its near cube.

    The kernels are translation-invariant, so this is computed once and reused
    by every basis function.

    Parameters
    ----------
    grid
        The extended grid, which fixes the cell pitch.
    near
        Fixes the cube's size and the block's place in it.
    medium
        Supplies ``j omega eps_0``.
    symbols_n, symbols_k
        The kernels compressed for the near cube, from :func:`kernels`.
    n_components
        Unknowns per cell: 3 for the constant basis, 12 for the linear one.

    Returns
    -------
    electric : torch.Tensor
        Shape ``(c * expansion ** 3, c, span ** 3)``, complex.
    magnetic : torch.Tensor
        The same for the magnetic kernel.
    """
    span = near.span
    shape = (span, span, span)
    reach = (span - 1) // 2
    offsets = near.expansion[0] - near.centre[0] + reach
    n_block = offsets.shape[0]

    electric = vie.unit_responses(symbols_n, offsets, shape)
    magnetic = vie.unit_responses(symbols_k, offsets, shape, curl=True)

    # The Galerkin mass term of each unit current sits on its own cell.
    weights = vie.mass(n_components, grid.resolution).to(electric.device)
    component = torch.arange(n_components, device=grid.device)
    place = torch.arange(n_block, device=grid.device)
    cell = (offsets[:, 0] * span + offsets[:, 1]) * span + offsets[:, 2]
    source = (component[:, None] * n_block + place[None, :]).reshape(-1)
    electric[
        source, component.repeat_interleave(n_block), cell.repeat(n_components)
    ] -= weights.repeat_interleave(n_block).to(electric.dtype)
    return electric / medium.electric_scaling, magnetic


def assemble(
    body: VoxelBody,
    coil: Coil,
    impedance: torch.Tensor,
    medium: Medium,
    *,
    expansion: int = EXPANSION,
    distance: int | None = None,
    tol: float = 1e-7,
    triangle_order: int = 4,
    cell_order: int = 2,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
    linear: bool = False,
) -> Coupling:
    """Build every piece of the coupled operator.

    Parameters
    ----------
    body
        The body and its grid.
    coil
        The coil and its basis.
    impedance
        The coil's own matrix, MARIE's ``SIE.Z``.
    medium
        Supplies the wavenumber and ``j omega eps_0``.
    expansion
        Cells a side of the expansion block.
    distance
        Half-width, in cells, of the near corrections.
    tol
        Relative tolerance of the Tucker compressions.
    triangle_order, cell_order
        Quadrature orders of the coupling kernels.
    far_order, medium_order, near_order
        Quadrature orders of the body kernels.
    linear
        Give the body the piecewise-linear basis.

    Returns
    -------
    Coupling
        Ready for the coupled matrix-vector product.
    """
    span = widest_basis(coil, body.resolution)
    if span > WIDEST_BASIS:
        warnings.warn(
            f"the coil's widest basis function spans {span:.1f} voxels, beyond "
            f"{WIDEST_BASIS:.0f}: the coupled solve will take more GMRES "
            "iterations. Refine the coil mesh, or coarsen the body grid.",
            stacklevel=2,
        )
    grid = extended_domain(body, coil, expansion=expansion)
    near = near_lists(grid, coil, expansion=expansion, distance=distance)
    symbols_n, symbols_k, cube_n, cube_k = kernels(
        grid,
        near,
        medium,
        tol=tol,
        far_order=far_order,
        medium_order=medium_order,
        near_order=near_order,
        linear=linear,
    )
    n_components = 12 if linear else 3
    weights = projection(
        grid,
        coil,
        medium,
        near,
        triangle_order=triangle_order,
        cell_order=cell_order,
        linear=linear,
    )
    response = expansion_response(
        grid, near, medium, cube_n, cube_k, n_components=n_components
    )

    direct = direct_coupling(
        grid,
        coil,
        medium,
        near,
        triangle_order=triangle_order,
        cell_order=cell_order,
        linear=linear,
    )
    projected = projected_coupling(grid, coil, medium, near, weights, response)

    return Coupling(
        grid=grid,
        near=near,
        project=projection_matrix(grid, near, weights),
        scatter=scatter_matrix(grid, n_components),
        electric=(direct[0] - projected[0]).coalesce(),
        magnetic=(direct[1] - projected[1]).coalesce(),
        coil=coil_precorrection(coil, impedance, near, weights, response),
        symbols_n=symbols_n,
        symbols_k=symbols_k,
        linear=linear,
    )


def restrict(coupling: Coupling, mask: torch.Tensor) -> Coupling:
    """Take a coupling assembled over a region down to a body inside it.

    Every entry of the near corrections belongs to one basis function and one
    cell, whatever else the grid holds, so a coupling assembled once with a
    region of the body grid counted as body serves any body within that region:
    its rows are selected and the scatter rebuilt.

    Parameters
    ----------
    coupling
        Assembled by :func:`assemble` for a body whose mask is the region.
    mask
        The body to keep, on the same body grid, shape ``(n1, n2, n3)``,
        boolean, inside the region.

    Returns
    -------
    Coupling
        The coupling :func:`assemble` gives for a body with ``mask`` on the same
        grid.

    Raises
    ------
    ValueError
        If ``mask`` is not the body grid's shape, or reaches outside the region.
    """
    grid = coupling.grid
    start = grid.body_origin
    window = (
        slice(start[0], start[0] + mask.shape[0]),
        slice(start[1], start[1] + mask.shape[1]),
        slice(start[2], start[2] + mask.shape[2]),
    )
    region = grid.mask[window]
    if tuple(region.shape) != tuple(mask.shape) or int(region.sum()) != int(
        grid.mask.sum()
    ):
        raise ValueError(
            f"the coupling's body grid is not of shape {tuple(mask.shape)}"
        )
    mask = mask.to(grid.mask.device)
    if bool((mask & ~region).any()):
        raise ValueError("the body reaches outside the region the coupling covers")
    placed = torch.zeros_like(grid.mask)
    placed[window] = mask
    kept = dataclasses.replace(grid, mask=placed)

    n_region = int(region.sum())
    n_components = coupling.n_components
    numbering = torch.full((n_region,), -1, dtype=torch.int64, device=grid.device)
    inside = mask[region]
    n_voxels = int(inside.sum())
    numbering[inside] = torch.arange(n_voxels, device=grid.device)

    def rows_of(matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.coalesce()
        row, column = matrix.indices()
        component, cell = row // n_region, row % n_region
        new = numbering[cell]
        keep = new >= 0
        return torch.sparse_coo_tensor(
            torch.stack([component[keep] * n_voxels + new[keep], column[keep]]),
            matrix.values()[keep],
            (n_components * n_voxels, matrix.shape[1]),
            check_invariants=False,
        ).coalesce()

    return dataclasses.replace(
        coupling,
        grid=kept,
        scatter=scatter_matrix(kept, n_components),
        electric=rows_of(coupling.electric),
        magnetic=rows_of(coupling.magnetic),
    )


def body_numbering(grid: ExtendedGrid) -> torch.Tensor:
    """Map each flat grid cell to its body degree of freedom, or to -1.

    Parameters
    ----------
    grid
        The extended grid.

    Returns
    -------
    torch.Tensor
        Shape ``(n_cells,)``, ``-1`` outside the body.
    """
    numbering = torch.full((grid.n_cells,), -1, dtype=torch.int64, device=grid.device)
    cells = grid.body_cells()
    numbering[cells] = torch.arange(cells.numel(), device=grid.device)
    return numbering


def unflatten(grid: ExtendedGrid, flat: torch.Tensor) -> torch.Tensor:
    """Turn flat grid indices back into triples.

    Parameters
    ----------
    grid
        The extended grid.
    flat
        Shape ``(...,)``, indices into the flattened grid.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)``.
    """
    second, third = grid.shape[1], grid.shape[2]
    return torch.stack(
        [flat // (second * third), (flat // third) % second, flat % third], dim=-1
    )


def near_body_pairs(grid: ExtendedGrid, near: NearLists, *, chunk: int = 4096):
    """Yield the basis-function-and-body-cell pairs the near correction covers.

    Parameters
    ----------
    grid
        The extended grid.
    near
        Where each basis function sits and how far it reaches.
    chunk
        Pairs yielded at a time.

    Yields
    ------
    tuple of torch.Tensor
        The basis function of each pair and the flat grid cell of each pair.
    """
    dofs, cells = [], []
    for _, block_dofs, block_cells in _near_body_blocks(
        grid, near, near.centre.shape[0]
    ):
        dofs.append(block_dofs)
        cells.append(block_cells)
    if not cells:
        return
    dofs, cells = torch.cat(dofs), torch.cat(cells)
    for start in range(0, cells.numel(), chunk):
        yield dofs[start : start + chunk], cells[start : start + chunk]


def _near_body_blocks(grid: ExtendedGrid, near: NearLists, block: int):
    """Yield each run of ``block`` basis functions with the body cells it reaches.

    Yields
    ------
    tuple
        The run's first basis function, the basis function of each pair, and the
        flat grid cell of each pair.
    """
    reach = (near.span - 1) // 2
    steps = torch.arange(-reach, reach + 1, device=grid.device)
    cube = torch.cartesian_prod(steps, steps, steps)
    inside = grid.mask.reshape(-1)
    limits = torch.tensor(grid.shape, device=grid.device)
    n_dof = near.centre.shape[0]

    for begin in range(0, n_dof, block):
        dofs, cells = [], []
        for dof in range(begin, min(begin + block, n_dof)):
            index = near.centre[dof][None, :] + cube
            held = ((index >= 0) & (index < limits)).all(dim=1)
            flat = grid.flatten(index[held])
            flat = flat[inside[flat]]
            if flat.numel() == 0:
                continue
            dofs.append(torch.full_like(flat, dof))
            cells.append(flat)
        if cells:
            yield begin, torch.cat(dofs), torch.cat(cells)
