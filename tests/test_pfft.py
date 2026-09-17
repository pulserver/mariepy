"""The precorrected FFT coupling: its grid, its projection and its corrections."""

import pytest
import torch

from mariepy import coupling, pfft, sie, vie
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.quadrature import lebedev_26_directions
from mariepy.vie import _DYADIC_INDEX, green_n

FIELD_STRENGTH = 3.0
RESOLUTION = 0.01
BODY_RADIUS = 0.02
COIL_RADIUS = 0.05
DISTANCE = 3

# MARIE's tol, which PLAN.md makes the default for every other tolerance.
TOLERANCE = 1e-5

# What the three-cell expansion block reproduces of the coil matrix beyond the
# collocation sphere, on the grid the tests below build.
PROJECTION_ERROR = 1e-3


def _medium():
    return Medium(FIELD_STRENGTH)


def _body(device, radius=BODY_RADIUS):
    return VoxelBody.sphere(radius, RESOLUTION, 52.0, 0.55, padding=1, device=device)


def _coil(device, radius=COIL_RADIUS, n_around=8):
    mesh = SurfaceMesh.loop(
        radius=radius, width=0.01, n_around=n_around, n_across=1, device=device
    )
    port = Port(tag=1, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
    return SurfaceCoil.build(mesh, (port,))


def _placed(device, **coil_arguments):
    body = _body(device)
    coil = _coil(device, **coil_arguments)
    grid = pfft.extended_domain(body, coil)
    return body, coil, grid, pfft.near_lists(grid, coil, distance=DISTANCE)


def test_the_extended_grid_holds_every_expansion_block(device):
    _, _, grid, near = _placed(device)
    limits = torch.tensor(grid.shape, device=grid.device)
    assert bool((near.expansion >= 0).all())
    assert bool((near.expansion < limits).all())


def test_the_extended_grid_carries_the_body_where_the_body_grid_put_it(device):
    body, _, grid, _ = _placed(device)
    assert int(grid.mask.sum()) == body.n_voxels
    start = grid.body_origin
    inside = grid.mask[
        start[0] : start[0] + body.shape[0],
        start[1] : start[1] + body.shape[1],
        start[2] : start[2] + body.shape[2],
    ]
    assert bool((inside == body.mask).all())
    corner = grid.centres(torch.tensor(start, device=grid.device))
    torch.testing.assert_close(
        corner, torch.tensor(body.origin, dtype=torch.float64, device=grid.device)
    )


def test_a_coil_that_leaves_the_grid_is_refused(device):
    body = _body(device)
    coil = _coil(device)
    grid = pfft.extended_domain(body, _coil(device, radius=0.02))
    with pytest.raises(ValueError, match="leaves the extended grid"):
        pfft.near_lists(grid, coil, distance=DISTANCE)


def test_the_projection_reproduces_the_field_of_a_basis_function_at_the_collocation_sphere(
    device,
):
    medium = _medium()
    _, coil, grid, near = _placed(device)
    weights = pfft.projection(grid, coil, medium, near)

    offsets = (near.expansion[0] - near.centre[0]).to(torch.float64) * grid.resolution
    reach = ((near.span - 1) / 2 + 1) * grid.resolution
    collocation = reach * lebedev_26_directions(device=grid.device)
    matrix = coupling.collocation_matrix(
        offsets, collocation, medium, cell_size=grid.resolution
    )
    points = grid.centres(near.centre)[:, None, :] + collocation[None, :, :]
    field = coupling.coupling_n(
        coil.rwg_vertices().repeat_interleave(collocation.shape[0], dim=0),
        points.reshape(-1, 3),
        medium,
    ).reshape(coil.n_dof, collocation.shape[0], 3)
    expected = field.permute(0, 2, 1).reshape(coil.n_dof, -1).transpose(0, 1)

    residual = matrix @ weights.reshape(coil.n_dof, -1).transpose(0, 1) - expected
    assert float(residual.abs().max() / expected.abs().max()) <= TOLERANCE


def test_the_projection_matrix_places_each_basis_function_on_its_own_expansion_block(
    device,
):
    medium = _medium()
    _, coil, grid, near = _placed(device)
    weights = pfft.projection(grid, coil, medium, near)
    matrix = pfft.projection_matrix(grid, near, weights)
    dense = matrix.to_dense().reshape(3, grid.n_cells, coil.n_dof)
    for dof in range(coil.n_dof):
        carried = torch.nonzero(dense[0, :, dof] != 0, as_tuple=False).flatten()
        block = grid.flatten(near.expansion[dof])
        assert set(carried.tolist()) <= set(block.tolist())


def test_the_scatter_matrix_is_the_body_s_own_degree_of_freedom_map(device):
    body, _, grid, _ = _placed(device)
    matrix = pfft.scatter_matrix(grid).to_dense()
    assert matrix.shape == (3 * grid.n_cells, 3 * body.n_voxels)
    identity = matrix.transpose(0, 1) @ matrix
    torch.testing.assert_close(
        identity,
        torch.eye(3 * body.n_voxels, dtype=identity.dtype, device=identity.device),
    )


def test_the_expansion_response_is_the_body_kernel_at_a_cell_the_block_does_not_touch(
    device,
):
    medium = _medium()
    _, _, grid, near = _placed(device)
    _, _, cube_n, cube_k = pfft.kernels(grid, near, medium, medium_order=4)
    electric, _ = pfft.expansion_response(grid, near, medium, cube_n, cube_k)

    reach = (near.span - 1) // 2
    step = torch.tensor([3, 1, 0], device=grid.device)
    cell = torch.tensor([reach, reach, reach], device=grid.device) + step
    where = int((cell[0] * near.span + cell[1]) * near.span + cell[2])

    separation = step.to(torch.float64) * grid.resolution
    dyadic = (
        grid.resolution**6
        * green_n(separation, medium.wavenumber)
        / medium.electric_scaling
    )
    centre = near.expansion.shape[1] // 2
    for component in range(3):
        got = electric[component * near.expansion.shape[1] + centre, :, where]
        want = torch.stack([dyadic[_DYADIC_INDEX[row][component]] for row in range(3)])
        torch.testing.assert_close(
            got, want, rtol=0.02, atol=1e-3 * float(want.abs().max())
        )


def test_the_projected_coil_block_reproduces_the_coil_matrix_beyond_the_collocation_sphere(
    device,
):
    medium = _medium()
    _, coil, grid, near = _placed(device, n_around=32, radius=0.08)
    symbols_n, _, _, _ = pfft.kernels(grid, near, medium, medium_order=4)
    weights = pfft.projection(grid, coil, medium, near)
    matrix = pfft.projection_matrix(grid, near, weights)

    field = (
        matrix.to_dense()
        .reshape(3, grid.n_cells, coil.n_dof)
        .permute(2, 0, 1)
        .reshape(coil.n_dof, 3, *grid.shape)
    )
    applied = (
        vie.apply_n(symbols_n, field) - vie.apply_g(field, grid.resolution)
    ) / medium.electric_scaling
    through = torch.einsum(
        "ncg,mcg->mn",
        applied.reshape(coil.n_dof, 3, -1),
        field.reshape(coil.n_dof, 3, -1),
    )

    system = sie.assemble(coil, medium)
    separation = (
        (near.centre[:, None, :] - near.centre[None, :, :]).abs().max(dim=-1).values
    )
    beyond = separation > (near.span - 1) // 2 + 1
    assert int(beyond.sum()) > 0
    error = (through[beyond] - system.impedance[beyond]).abs().max()
    assert float(error / system.impedance[beyond].abs().max()) <= PROJECTION_ERROR


def test_the_corrected_coupling_reproduces_the_direct_integral_where_it_corrects(
    device,
):
    medium = _medium()
    body, coil, grid, near = _placed(device)
    _, _, cube_n, cube_k = pfft.kernels(grid, near, medium, medium_order=4)
    weights = pfft.projection(grid, coil, medium, near)
    response = pfft.expansion_response(grid, near, medium, cube_n, cube_k)
    direct = pfft.direct_coupling(grid, coil, medium, near)
    projected = pfft.projected_coupling(grid, coil, medium, near, weights, response)

    corrected = (direct[0] - projected[0]).to_dense() + _through_the_grid(
        grid, coil, medium, near, weights, cube_n
    )
    expected = _direct_dense(grid, coil, medium, near, body)
    covered = expected != 0
    error = (corrected[covered] - expected[covered]).abs().max()
    assert float(error / expected[covered].abs().max()) <= 1e-9


def _through_the_grid(grid, coil, medium, near, weights, symbols):
    """Apply the body kernel over the near cube to every basis function's projection."""
    span = near.span
    reach = (span - 1) // 2
    sources = torch.zeros(
        (coil.n_dof, 3, span, span, span), dtype=torch.complex128, device=grid.device
    )
    local = near.expansion - near.centre[:, None, :] + reach
    for dof in range(coil.n_dof):
        sources[dof, :, local[dof, :, 0], local[dof, :, 1], local[dof, :, 2]] = weights[
            dof
        ]
    applied = (
        vie.apply_n(symbols, sources) - vie.apply_g(sources, grid.resolution)
    ) / medium.electric_scaling

    n_voxels = int(grid.mask.sum())
    out = torch.zeros(
        (3 * n_voxels, coil.n_dof), dtype=torch.complex128, device=grid.device
    )
    numbering = pfft.body_numbering(grid)
    for dof, cells in pfft.near_body_pairs(grid, near, chunk=1 << 30):
        index = pfft.unflatten(grid, cells) - near.centre[dof] + reach
        for component in range(3):
            out[component * n_voxels + numbering[cells], dof] = applied[
                dof, component, index[:, 0], index[:, 1], index[:, 2]
            ]
    return out


def _direct_dense(grid, coil, medium, near, body):
    """Integrate each basis function against the body cells its correction covers."""
    n_voxels = int(grid.mask.sum())
    out = torch.zeros(
        (3 * n_voxels, coil.n_dof), dtype=torch.complex128, device=grid.device
    )
    numbering = pfft.body_numbering(grid)
    corners = coil.rwg_vertices()
    for dof, cells in pfft.near_body_pairs(grid, near, chunk=1 << 30):
        points = grid.centres(pfft.unflatten(grid, cells))
        value = body.resolution**3 * coupling.coupling_n(
            corners[dof], points, medium, cell_size=grid.resolution
        )
        for component in range(3):
            out[component * n_voxels + numbering[cells], dof] = value[:, component]
    return out


def _whole_grid(body):
    """The body's grid with every cell counted as body."""
    return VoxelBody(
        permittivity=torch.ones_like(body.permittivity),
        conductivity=torch.zeros_like(body.conductivity),
        mask=torch.ones_like(body.mask),
        resolution=body.resolution,
        origin=body.origin,
    )


@pytest.mark.parametrize("linear", [False, True], ids=["constant", "linear"])
def test_a_coupling_over_a_region_restricts_to_the_body_s_own(linear, device):
    body = _body(device)
    coil = _coil(device)
    medium = _medium()
    impedance = sie.assemble(coil, medium).impedance
    orders = {"far_order": 2, "medium_order": 2, "near_order": 4, "linear": linear}
    own = pfft.assemble(body, coil, impedance, medium, distance=DISTANCE, **orders)
    whole = pfft.assemble(
        _whole_grid(body), coil, impedance, medium, distance=DISTANCE, **orders
    )
    restricted = pfft.restrict(whole, body.mask)

    assert bool((restricted.grid.mask == own.grid.mask).all())
    for name in ("project", "scatter", "electric", "magnetic", "coil"):
        torch.testing.assert_close(
            getattr(restricted, name).to_dense(),
            getattr(own, name).to_dense(),
            rtol=1e-12,
            atol=0.0,
        )


def test_a_body_reaching_outside_the_region_is_refused(device):
    body = _body(device)
    coil = _coil(device)
    medium = _medium()
    impedance = sie.assemble(coil, medium).impedance
    own = pfft.assemble(
        body,
        coil,
        impedance,
        medium,
        distance=DISTANCE,
        far_order=2,
        medium_order=2,
        near_order=4,
    )
    with pytest.raises(ValueError, match="outside the region"):
        pfft.restrict(own, torch.ones_like(body.mask))
