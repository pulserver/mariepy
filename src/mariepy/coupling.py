"""The field an RWG basis function puts on the body, and the body's on a point.

Ported from MARIE 3.0's 24 sources under
``src_integral_equations/src_svie/Cpp_Assembly/src/``,
``Assemble_rwg_coupling_matrix_{N,K}_{x,y,z}{,1,2,3}.cpp``, and from
``src_wsvie/src_pfft/src_pfft_supporting/pfft_proj_pwx_to_collocation.m``.

MARIE's 24 sources are one file with 23 of its 24 inner lines commented out.
The variation is exactly two-dimensional -- operator N or K, field component x,
y or z, basis term constant or linear in one coordinate of the cell -- and all
of it is carried here by arguments. The kernels themselves are the ones the
body operator already uses: the coupling of an RWG basis function ``f`` to a
point is

``E(r) = 1 / (j omega eps_0) integral_S N(r - r') f(r') dS'``

for the electric field and ``H(r) = integral_S K(r - r') x f(r') dS'`` for the
magnetic, with ``N`` and ``K`` from :mod:`mariepy.vie`. Nothing else is needed:
the surface quadrature carries the basis function, and the cell quadrature
carries the observer.
"""

from __future__ import annotations

import torch

from mariepy.constants import Medium
from mariepy.quadrature import dunavant, gauss_legendre_1d
from mariepy.vie import _DYADIC_INDEX, green_k, green_n

__all__ = ["collocation_matrix", "coupling_k", "coupling_n"]

_CHUNK_ELEMENTS = 1 << 21


def coupling_n(
    corners: torch.Tensor,
    points: torch.Tensor,
    medium: Medium,
    *,
    triangle_order: int = 4,
    cell_size: float | None = None,
    cell_order: int = 2,
    basis_term: int = 0,
) -> torch.Tensor:
    """Give the electric field each RWG basis function puts on its observer.

    Parameters
    ----------
    corners
        Shape ``(n, 4, 3)``: the free vertex of the positive triangle, the free
        vertex of the negative triangle, and the two vertices of the shared
        edge, as :meth:`mariepy.coil.SurfaceCoil.rwg_vertices` returns them.
    points
        Shape ``(n, 3)``: one observer per basis function. The caller pairs
        them, so a near list is a flat list of pairs.
    medium
        Supplies the wavenumber and ``j omega eps_0``.
    triangle_order
        Degree of the Dunavant rule on each triangle.
    cell_size
        Cell pitch in metres, to average the field over a cell, or None to take
        it at the point. The cell average is not multiplied by the cell volume:
        the caller does that when it wants the integral, as
        ``pfft_surface_assemble_direct_bc.m`` does and
        ``pfft_projection_surface_assembly.m`` does not.
    cell_order
        Points per axis of the Gauss rule over the cell.
    basis_term
        ``0`` for the piecewise-constant cell basis, or ``1``, ``2``, ``3`` for
        the term linear in the cell's own x, y or z. Milestone 1 uses ``0``.

    Returns
    -------
    torch.Tensor
        Shape ``(n, 3)``, complex: the three field components.
    """
    return _couple(
        corners,
        points,
        medium,
        triangle_order,
        cell_size,
        cell_order,
        basis_term,
        _electric,
    )


def coupling_k(
    corners: torch.Tensor,
    points: torch.Tensor,
    medium: Medium,
    *,
    triangle_order: int = 4,
    cell_size: float | None = None,
    cell_order: int = 2,
    basis_term: int = 0,
) -> torch.Tensor:
    """Give the magnetic field each RWG basis function puts on its observer.

    Parameters
    ----------
    corners, points, medium, triangle_order, cell_size, cell_order, basis_term
        As in :func:`coupling_n`.

    Returns
    -------
    torch.Tensor
        Shape ``(n, 3)``, complex.
    """
    return _couple(
        corners,
        points,
        medium,
        triangle_order,
        cell_size,
        cell_order,
        basis_term,
        _magnetic,
    )


def collocation_matrix(
    centres: torch.Tensor,
    points: torch.Tensor,
    medium: Medium,
    *,
    cell_size: float,
    cell_order: int = 2,
    n_basis: int = 1,
) -> torch.Tensor:
    """Build the matrix from cell currents to the electric field at points.

    The least-squares solve of this matrix against the field of one basis
    function gives that function's projection onto the expansion cells.

    Parameters
    ----------
    centres
        Cell centres, shape ``(n_cells, 3)``.
    points
        Collocation points, shape ``(n_points, 3)``.
    medium
        Supplies the wavenumber and ``j omega eps_0``.
    cell_size
        Cell pitch in metres.
    cell_order
        Points per axis of the Gauss rule over each cell.
    n_basis
        ``1`` for the piecewise-constant cell basis, ``4`` for the
        piecewise-linear one.

    Returns
    -------
    torch.Tensor
        Shape ``(3 * n_points, 3 * n_basis * n_cells)``, complex, in volts per
        metre for one ampere per square metre in the cell: the cell volume is
        carried here. Rows run component-major over the points; columns run
        component-major, then basis term, then cell, as
        ``pfft_proj_pwx_to_collocation.m`` orders them.
    """
    weights, nodes, factors = _cell_rule(cell_order, centres.device, centres.dtype)
    offsets = cell_size / 2.0 * nodes
    separation = points[:, None, None, :] - (
        centres[None, :, None, :] + offsets[None, None, :, :]
    )
    dyadic = green_n(separation, medium.wavenumber) / medium.electric_scaling

    n_points, n_cells = points.shape[0], centres.shape[0]
    matrix = torch.zeros(
        (3, n_points, 3, n_basis, n_cells),
        dtype=torch.complex128,
        device=centres.device,
    )
    volume = cell_size**3
    for term in range(n_basis):
        weighted = volume * torch.einsum(
            "g,pcgd->pcd", (weights * factors[term]).to(torch.complex128), dyadic
        )
        for row in range(3):
            for column in range(3):
                matrix[row, :, column, term, :] = weighted[
                    ..., _DYADIC_INDEX[row][column]
                ]
    return matrix.reshape(3 * n_points, 3 * n_basis * n_cells)


def _electric(
    separation: torch.Tensor, rho: torch.Tensor, medium: Medium
) -> torch.Tensor:
    """Contract the double-curl kernel with the surface current."""
    dyadic = green_n(separation, medium.wavenumber)
    return (
        torch.stack(
            [
                sum(
                    dyadic[..., _DYADIC_INDEX[row][column]] * rho[..., column]
                    for column in range(3)
                )
                for row in range(3)
            ],
            dim=-1,
        )
        / medium.electric_scaling
    )


def _magnetic(
    separation: torch.Tensor, rho: torch.Tensor, medium: Medium
) -> torch.Tensor:
    """Cross the curl kernel with the surface current."""
    return torch.linalg.cross(green_k(separation, medium.wavenumber), rho, dim=-1)


def _cell_rule(order, device, dtype):
    """Return the cell cubature: its weights, its nodes and one factor per basis term."""
    line_weights, line_nodes = gauss_legendre_1d(order, device=device, dtype=dtype)
    grid = torch.cartesian_prod(line_nodes, line_nodes, line_nodes)
    weights = (
        line_weights[:, None, None]
        * line_weights[None, :, None]
        * line_weights[None, None, :]
    ).reshape(-1) / 8.0
    factors = [torch.ones_like(weights)]
    factors += [grid[:, axis] / 2.0 for axis in range(3)]
    return weights, grid, factors


def _couple(
    corners, points, medium, triangle_order, cell_size, cell_order, basis_term, kernel
):
    """Integrate one kernel over each basis function's two triangles and its observer's cell."""
    device = corners.device
    triangle_weights, barycentric = dunavant(triangle_order, device=device)
    # MARIE's dunavant_rule.m halves Burkardt's weights, so the surface rule
    # carries the RWG basis function's own 1 / (2 A).
    triangle_weights = 0.5 * triangle_weights

    if cell_size is None:
        cell_weights = torch.ones(1, dtype=corners.dtype, device=device)
        offsets = torch.zeros((1, 3), dtype=corners.dtype, device=device)
        factor = torch.ones(1, dtype=corners.dtype, device=device)
    else:
        cell_weights, nodes, factors = _cell_rule(cell_order, device, corners.dtype)
        offsets = cell_size / 2.0 * nodes
        factor = factors[basis_term]

    free = corners[:, :2]
    shared = corners[:, 2:]
    length = torch.linalg.vector_norm(shared[:, 1] - shared[:, 0], dim=-1)
    sign = torch.tensor([1.0, -1.0], dtype=corners.dtype, device=device)

    source = (
        barycentric[None, None, :, 0, None] * free[:, :, None, :]
        + barycentric[None, None, :, 1, None] * shared[:, None, None, 0, :]
        + barycentric[None, None, :, 2, None] * shared[:, None, None, 1, :]
    )
    rho = sign[None, :, None, None] * (source - free[:, :, None, :])
    weight = ((cell_weights * factor)[:, None] * triangle_weights[None, :]).to(
        torch.complex128
    )

    rows = int(corners.shape[0])
    budget = max(
        1, _CHUNK_ELEMENTS // max(1, offsets.shape[0] * 2 * barycentric.shape[0])
    )
    pieces = []
    for start in range(0, rows, budget):
        stop = min(start + budget, rows)
        observer = points[start:stop, None, :] + offsets[None, :, :]
        separation = observer[:, :, None, None, :] - source[start:stop, None, :, :, :]
        value = kernel(separation, rho[start:stop, None].to(torch.complex128), medium)
        pieces.append(
            torch.einsum("gq,ngtqc->nc", weight, value)
            * length[start:stop, None].to(torch.complex128)
        )
    return (
        torch.cat(pieces)
        if pieces
        else torch.zeros((0, 3), dtype=torch.complex128, device=device)
    )
