"""Precorrected FFT coupling between a surface coil and a voxel body.

Ported from MARIE 3.0's ``src_integral_equations/src_wsvie/src_pfft``:
``src_svie_pfft/pfft_surface_domain.m``,
``pfft_proj_surface_create_near_lists.m``,
``pfft_projection_surface_assembly.m``, ``pfft_surface_assemble_direct_bc.m``,
``src_pfft_coil/pfft_assemble_voxel_bc.m``, ``pfft_assemble_voxel_cc.m`` and
the supporting files under ``src_pfft_supporting``.

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

import math
from dataclasses import dataclass

import torch

from mariepy import coupling, vie
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.quadrature import lebedev_26_directions
from mariepy.tucker import CirculantSymbol, circulant_tucker

__all__ = [
    "Coupling",
    "ExtendedGrid",
    "NearLists",
    "assemble",
    "coil_precorrection",
    "direct_coupling",
    "expansion_response",
    "extended_domain",
    "near_lists",
    "projected_coupling",
    "projection",
    "projection_matrix",
    "scatter_matrix",
]

# MARIE fixes the expansion block at three cells a side in
# ``pfft_surface_domain.m``; the code below carries it as a parameter but the
# collocation sphere and the near distance are both sized from it.
EXPANSION = 3


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
        Coil currents onto the extended grid, sparse ``(3 * n_cells, n_dof)``.
    scatter
        Body currents onto the extended grid, sparse ``(3 * n_cells, 3 * n_voxels)``.
    electric
        The near correction of the electric coupling, sparse
        ``(3 * n_voxels, n_dof)``.
    magnetic
        The same for the magnetic coupling.
    coil
        The coil block's near correction, sparse ``(n_dof, n_dof)``.
    symbols_n, symbols_k
        The compressed kernels on the extended grid.
    """

    grid: ExtendedGrid
    near: NearLists
    project: torch.Tensor
    scatter: torch.Tensor
    electric: torch.Tensor
    magnetic: torch.Tensor
    coil: torch.Tensor
    symbols_n: tuple[CirculantSymbol, ...]
    symbols_k: tuple[CirculantSymbol, ...]


def extended_domain(
    body: VoxelBody, coil: SurfaceCoil, *, expansion: int = EXPANSION
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
    nodes = coil.mesh.nodes
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
    coil: SurfaceCoil,
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
    nodes = coil.mesh.nodes
    edges = coil.edges[coil.edge_of_dof]
    centre_point = 0.5 * (nodes[edges[:, 0]] + nodes[edges[:, 1]])
    origin = torch.tensor(grid.origin, dtype=torch.float64, device=nodes.device)
    centre = torch.round((centre_point - origin) / grid.resolution).to(torch.int64)

    half = (expansion - 1) // 2
    steps = torch.arange(-half, half + 1, device=nodes.device)
    block = torch.cartesian_prod(steps, steps, steps)
    expansion_index = centre[:, None, :] + block[None, :, :]

    limits = torch.tensor(grid.shape, device=nodes.device)
    if bool((expansion_index < 0).any()) or bool((expansion_index >= limits).any()):
        raise ValueError("a basis function's expansion block leaves the extended grid")

    lengths = coil.mesh.edge_lengths()
    carried = coil.dof_of_triangle() >= 0
    widest = torch.zeros(coil.n_dof, dtype=lengths.dtype, device=nodes.device)
    triangle, local = torch.nonzero(carried, as_tuple=True)
    widest.scatter_reduce_(
        0,
        coil.dof_of_triangle()[triangle, local],
        lengths[triangle].max(dim=1).values,
        reduce="amax",
    )
    if distance is None:
        by_triangle = math.ceil(float((widest / grid.resolution).mean()))
        by_cell = expansion + expansion // 2
        distance = math.ceil(1.6 * max(by_cell, by_triangle))
    span = 2 * distance + 2 * (expansion // 2) + expansion % 2
    return NearLists(
        centre=centre, expansion=expansion_index, span=span, distance=distance
    )


def projection(
    grid: ExtendedGrid,
    coil: SurfaceCoil,
    medium: Medium,
    near: NearLists,
    *,
    triangle_order: int = 4,
    cell_order: int = 2,
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
        Degree of the Dunavant rule on each triangle.
    cell_order
        Points per axis of the Gauss rule over each cell.

    Returns
    -------
    torch.Tensor
        Shape ``(n_dof, 3, expansion ** 3)``, complex: the current each
        expansion cell carries for a unit coefficient of the basis function.
    """
    device = grid.device
    offsets = (near.expansion[0] - near.centre[0]).to(torch.float64) * grid.resolution
    reach = ((near.span - 1) / 2 + 1) * grid.resolution
    collocation = reach * lebedev_26_directions(device=device)

    matrix = coupling.collocation_matrix(
        offsets, collocation, medium, cell_size=grid.resolution, cell_order=cell_order
    )

    corners = coil.rwg_vertices()
    centres = grid.centres(near.centre)
    points = centres[:, None, :] + collocation[None, :, :]
    n_dof, n_points = coil.n_dof, collocation.shape[0]
    field = coupling.coupling_n(
        corners.repeat_interleave(n_points, dim=0),
        points.reshape(-1, 3),
        medium,
        triangle_order=triangle_order,
    ).reshape(n_dof, n_points, 3)

    # The collocation sphere gives 78 equations for 81 cell currents, so the
    # system is underdetermined and a pseudo-inverse takes the smallest
    # solution. Which solution it is does not reach the answer: every one of
    # them radiates the same field, and the near correction subtracts back
    # exactly the one that was used. It is a pseudo-inverse rather than a
    # least-squares solve because CUDA's driver needs at least as many
    # equations as unknowns and this has fewer.
    right = field.permute(0, 2, 1).reshape(n_dof, 3 * n_points).transpose(0, 1)
    weights = torch.linalg.pinv(matrix) @ right
    n_block = near.expansion.shape[1]
    return weights.reshape(3, n_block, n_dof).permute(2, 0, 1).contiguous()


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
        Sparse ``(3 * n_cells, n_dof)``, complex.
    """
    device = grid.device
    n_dof, _, n_block = weights.shape
    cells = grid.flatten(near.expansion)
    component = torch.arange(3, device=device)
    rows = (component[None, :, None] * grid.n_cells + cells[:, None, :]).reshape(-1)
    columns = torch.arange(n_dof, device=device).repeat_interleave(3 * n_block)
    return torch.sparse_coo_tensor(
        torch.stack([rows, columns]),
        weights.reshape(-1),
        (3 * grid.n_cells, n_dof),
        check_invariants=False,
    ).coalesce()


def scatter_matrix(grid: ExtendedGrid) -> torch.Tensor:
    """Place the body's degrees of freedom on the extended grid.

    Parameters
    ----------
    grid
        The extended grid, whose mask fixes the order of the body's own
        degrees of freedom.

    Returns
    -------
    torch.Tensor
        Sparse ``(3 * n_cells, 3 * n_voxels)``, complex.
    """
    device = grid.device
    cells = grid.body_cells()
    n_voxels = cells.numel()
    component = torch.arange(3, device=device)
    rows = (component[:, None] * grid.n_cells + cells[None, :]).reshape(-1)
    columns = torch.arange(3 * n_voxels, device=device)
    values = torch.ones(rows.numel(), dtype=torch.complex128, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([rows, columns]),
        values,
        (3 * grid.n_cells, 3 * n_voxels),
        check_invariants=False,
    ).coalesce()


def direct_coupling(
    grid: ExtendedGrid,
    coil: SurfaceCoil,
    medium: Medium,
    near: NearLists,
    *,
    triangle_order: int = 4,
    cell_order: int = 2,
    chunk: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate each basis function against the body cells it comes close to.

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

    Returns
    -------
    electric : torch.Tensor
        Sparse ``(3 * n_voxels, n_dof)``, complex.
    magnetic : torch.Tensor
        The same for the magnetic coupling.
    """
    corners = coil.rwg_vertices()
    volume = grid.resolution**3
    body_dof = body_numbering(grid)
    n_voxels = int(grid.mask.sum())
    component = torch.arange(3, device=grid.device)

    rows, columns, electric, magnetic = [], [], [], []
    for dof, cells in near_body_pairs(grid, near, chunk=chunk):
        points = grid.centres(unflatten(grid, cells))
        arguments = {
            "triangle_order": triangle_order,
            "cell_size": grid.resolution,
            "cell_order": cell_order,
        }
        electric.append(
            volume
            * coupling.coupling_n(corners[dof], points, medium, **arguments).reshape(-1)
        )
        magnetic.append(
            volume
            * coupling.coupling_k(corners[dof], points, medium, **arguments).reshape(-1)
        )
        rows.append(
            (component[None, :] * n_voxels + body_dof[cells][:, None]).reshape(-1)
        )
        columns.append(dof.repeat_interleave(3))

    shape = (3 * n_voxels, coil.n_dof)
    index = torch.stack([torch.cat(rows), torch.cat(columns)])
    return (
        torch.sparse_coo_tensor(
            index, torch.cat(electric), shape, check_invariants=False
        ).coalesce(),
        torch.sparse_coo_tensor(
            index, torch.cat(magnetic), shape, check_invariants=False
        ).coalesce(),
    )


def projected_coupling(
    grid: ExtendedGrid,
    coil: SurfaceCoil,
    medium: Medium,
    near: NearLists,
    weights: torch.Tensor,
    response: tuple[torch.Tensor, torch.Tensor],
    *,
    chunk: int = 4096,
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
    chunk
        Basis-function-and-cell pairs taken at a time.

    Returns
    -------
    electric : torch.Tensor
        Sparse ``(3 * n_voxels, n_dof)``, complex.
    magnetic : torch.Tensor
        The same for the magnetic coupling.
    """
    del medium
    block, curl = response
    flat_weights = weights.reshape(weights.shape[0], -1)
    body_dof = body_numbering(grid)
    n_voxels = int(grid.mask.sum())
    component = torch.arange(3, device=grid.device)
    reach = (near.span - 1) // 2

    rows, columns, electric, magnetic = [], [], [], []
    for dof, cells in near_body_pairs(grid, near, chunk=chunk):
        local = unflatten(grid, cells) - near.centre[dof] + reach
        where = (local[:, 0] * near.span + local[:, 1]) * near.span + local[:, 2]
        here = flat_weights[dof]
        electric.append(torch.einsum("jcn,nj->nc", block[..., where], here).reshape(-1))
        magnetic.append(torch.einsum("jcn,nj->nc", curl[..., where], here).reshape(-1))
        rows.append(
            (component[None, :] * n_voxels + body_dof[cells][:, None]).reshape(-1)
        )
        columns.append(dof.repeat_interleave(3))

    shape = (3 * n_voxels, coil.n_dof)
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
    coil: SurfaceCoil,
    impedance: torch.Tensor,
    near: NearLists,
    weights: torch.Tensor,
    response: tuple[torch.Tensor, torch.Tensor],
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

    Returns
    -------
    torch.Tensor
        Sparse ``(n_dof, n_dof)``, complex.
    """
    block, _ = response
    flat_weights = weights.reshape(weights.shape[0], -1)
    reach = (near.span - 1) // 2
    n_dof = coil.n_dof

    separation = (
        (near.centre[:, None, :] - near.centre[None, :, :]).abs().max(dim=-1).values
    )
    observer, source = torch.nonzero(separation < near.distance, as_tuple=True)

    local = near.expansion[observer] - near.centre[source][:, None, :] + reach
    where = (local[..., 0] * near.span + local[..., 1]) * near.span + local[..., 2]
    picked = block[..., where.reshape(-1)].reshape(
        block.shape[0], 3, where.shape[0], where.shape[1]
    )
    projected = torch.einsum(
        "jcpb,pj,pcb->p", picked, flat_weights[source], weights[observer]
    )

    values = impedance[observer, source] - projected
    return torch.sparse_coo_tensor(
        torch.stack([observer, source]), values, (n_dof, n_dof), check_invariants=False
    ).coalesce()


def kernels(
    grid: ExtendedGrid,
    near: NearLists,
    medium: Medium,
    *,
    tol: float = 1e-7,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
) -> tuple[tuple[CirculantSymbol, ...], ...]:
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
    symbols_n: tuple[CirculantSymbol, ...],
    symbols_k: tuple[CirculantSymbol, ...],
    *,
    chunk: int = 8,
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
    chunk
        Unit sources taken at a time.

    Returns
    -------
    electric : torch.Tensor
        Shape ``(3 * expansion ** 3, 3, span ** 3)``, complex.
    magnetic : torch.Tensor
        The same for the magnetic kernel.
    """
    span = near.span
    shape = (span, span, span)
    reach = (span - 1) // 2
    offsets = near.expansion[0] - near.centre[0] + reach
    n_block = offsets.shape[0]

    sources = torch.zeros(
        (3, n_block, 3, *shape), dtype=torch.complex128, device=grid.device
    )
    place = torch.arange(n_block, device=grid.device)
    for component in range(3):
        sources[
            component, place, component, offsets[:, 0], offsets[:, 1], offsets[:, 2]
        ] = 1.0
    sources = sources.reshape(3 * n_block, 3, *shape)

    electric, magnetic = [], []
    for start in range(0, sources.shape[0], chunk):
        piece = sources[start : start + chunk]
        applied = vie.apply_n(symbols_n, piece) - vie.apply_g(piece, grid.resolution)
        electric.append(applied / medium.electric_scaling)
        magnetic.append(vie.apply_k(symbols_k, piece))
    return (
        torch.cat(electric).reshape(3 * n_block, 3, -1),
        torch.cat(magnetic).reshape(3 * n_block, 3, -1),
    )


def assemble(
    body: VoxelBody,
    coil: SurfaceCoil,
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

    Returns
    -------
    Coupling
        Ready for the coupled matrix-vector product.
    """
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
    )
    weights = projection(
        grid, coil, medium, near, triangle_order=triangle_order, cell_order=cell_order
    )
    response = expansion_response(grid, near, medium, cube_n, cube_k)

    direct = direct_coupling(
        grid, coil, medium, near, triangle_order=triangle_order, cell_order=cell_order
    )
    projected = projected_coupling(grid, coil, medium, near, weights, response)

    return Coupling(
        grid=grid,
        near=near,
        project=projection_matrix(grid, near, weights),
        scatter=scatter_matrix(grid),
        electric=(direct[0] - projected[0]).coalesce(),
        magnetic=(direct[1] - projected[1]).coalesce(),
        coil=coil_precorrection(coil, impedance, near, weights, response),
        symbols_n=symbols_n,
        symbols_k=symbols_k,
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
    reach = (near.span - 1) // 2
    steps = torch.arange(-reach, reach + 1, device=grid.device)
    cube = torch.cartesian_prod(steps, steps, steps)
    inside = grid.mask.reshape(-1)
    limits = torch.tensor(grid.shape, device=grid.device)

    dofs, cells = [], []
    for dof in range(near.centre.shape[0]):
        index = near.centre[dof][None, :] + cube
        held = ((index >= 0) & (index < limits)).all(dim=1)
        flat = grid.flatten(index[held])
        flat = flat[inside[flat]]
        if flat.numel() == 0:
            continue
        dofs.append(torch.full_like(flat, dof))
        cells.append(flat)
    if not cells:
        return
    dofs, cells = torch.cat(dofs), torch.cat(cells)
    for start in range(0, cells.numel(), chunk):
        yield dofs[start : start + chunk], cells[start : start + chunk]
