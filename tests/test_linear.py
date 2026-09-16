"""The piecewise-linear body basis: its kernel, its products, its right-hand side."""

import itertools

import pytest
import torch

from mariepy import tucker, vie
from mariepy.body import VoxelBody
from mariepy.constants import Medium
from mariepy.incident import plane_wave
from mariepy.preconditioner import body_diagonal
from mariepy.quadrature import gauss_legendre_1d
from mariepy.solver import BodyOperator, solve_body

RESOLUTION = 0.01
WAVENUMBER = 30.0

# MARIE's bt_n1, bt_n2, bt_n3 in assembly_fft_circ_tucker_pwl.m, one row per
# axis, one column per stored pair.
MARIE_PAIR_PARITY = (
    (+1, +1, +1, +1, -1, +1, +1, -1, -1, +1),
    (+1, +1, +1, +1, +1, -1, +1, -1, +1, -1),
    (+1, +1, +1, +1, +1, +1, -1, +1, -1, -1),
)

# MARIE's llp table in mvp_N_pwl_tucker.m, one-based and signed.
MARIE_PAIR_OF = ((1, 5, 6, 7), (-5, 2, 8, 9), (-6, 8, 3, 10), (-7, 9, 10, 4))

REDUCTIONS = {
    "N": (vie.surface_surface_n, vie.volume_volume_n, vie._N_REDUCTION),
    "K": (vie.surface_surface_k, vie.volume_volume_k, vie._K_REDUCTION),
}


def test_the_pair_parity_is_marie_s_table():
    for column, pair in enumerate(vie.PAIRS):
        derived = tucker.pair_parity(*pair)
        assert derived == tuple(MARIE_PAIR_PARITY[axis][column] for axis in range(3))


def test_the_pair_lookup_is_marie_s_table():
    for test, basis in itertools.product(range(4), repeat=2):
        pair, sign = vie.PAIR_OF[test][basis]
        expected = MARIE_PAIR_OF[test][basis]
        assert (pair + 1, sign) == (abs(expected), 1 if expected > 0 else -1)


def test_every_stored_pair_names_the_pair_it_stores():
    for pair, (test, basis) in enumerate(vie.PAIRS):
        assert vie.PAIR_OF[test][basis] == (pair, +1)


@pytest.mark.parametrize("operator", ["N", "K"])
@pytest.mark.parametrize("offset", [(2, 0, 0), (2, 1, 0), (3, 2, 1), (0, 2, 1)])
def test_the_linear_surface_reduction_agrees_with_the_volume_rule_where_both_hold(
    operator, offset
):
    """Two independent forms of every pair, away from the singularity.

    The reduction carries all four face kernels and their constants; the volume
    rule carries none of them, so agreement checks every constant ported.
    """
    reduced_by, direct_by, _ = REDUCTIONS[operator]
    reduced = reduced_by(offset, RESOLUTION, WAVENUMBER, order=12, linear=True)
    direct = direct_by(
        RESOLUTION * torch.tensor([offset], dtype=torch.float64),
        RESOLUTION,
        WAVENUMBER,
        order=10,
        linear=True,
    )[0]
    # Relative to the whole kernel: a pair that vanishes by symmetry has nothing
    # of its own to be relative to.
    assert (reduced - direct).abs().max() <= 1e-10 * direct.abs().max()


@pytest.mark.parametrize("operator", ["N", "K"])
def test_the_constant_pair_of_the_linear_kernel_is_the_constant_kernel(operator):
    assemble = vie.kernel_n if operator == "N" else vie.kernel_k
    orders = {"far_order": 2, "medium_order": 2, "near_order": 4}
    linear = assemble((3, 2, 2), RESOLUTION, WAVENUMBER, linear=True, **orders)
    constant = assemble((3, 2, 2), RESOLUTION, WAVENUMBER, **orders)
    # Against the kernel's scale: some entries vanish by symmetry.
    assert (linear[..., 0, :] - constant).abs().max() <= 1e-13 * constant.abs().max()


def _volume_every_pair(offset, kernel, n_components, order):
    """Integrate a kernel over two cells for all sixteen pairs, without any table."""
    weights, nodes = gauss_legendre_1d(order)
    grids = torch.meshgrid(*([nodes] * 6), indexing="ij")
    weight = torch.ones_like(grids[0])
    for axis in range(6):
        shape = [1] * 6
        shape[axis] = order
        weight = weight * weights.reshape(shape)
    observer = [torch.ones_like(grids[0])] + [grids[a] / 2 for a in range(3)]
    source = [torch.ones_like(grids[0])] + [grids[a] / 2 for a in range(3, 6)]
    separation = RESOLUTION * torch.tensor(offset, dtype=torch.float64) + (
        RESOLUTION
        / 2
        * torch.stack(
            [grids[0] - grids[3], grids[1] - grids[4], grids[2] - grids[5]], -1
        )
    )
    values = kernel(separation.reshape(-1, 3), WAVENUMBER)
    out = torch.zeros(4, 4, n_components, dtype=torch.complex128)
    for test, basis in itertools.product(range(4), repeat=2):
        pair_weight = (weight * observer[test] * source[basis]).reshape(-1, 1)
        out[test, basis] = (RESOLUTION / 2) ** 6 * (pair_weight * values).sum(0)
    return out


def _as_matrix(stored, index, sign):
    """Spread stored components over the full three-by-three."""
    out = torch.zeros(*stored.shape[:-1], 3, 3, dtype=stored.dtype)
    for p, q in itertools.product(range(3), repeat=2):
        if index[p][q] is not None:
            scale = 1.0 if sign is None else sign[p][q]
            out[..., p, q] = scale * stored[..., index[p][q]]
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    ("operator", "tolerance"),
    [("N", 1e-5), ("K", 1e-4)],
)
def test_the_compressed_linear_operator_is_the_one_built_pair_by_pair(
    operator, tolerance
):
    """Every signed offset and every one of the sixteen pairs, computed directly.

    The compressed product reaches a negative offset through each pair's parity
    and a swapped pair through the sign of :data:`mariepy.vie.PAIR_OF`. Neither
    enters here, so agreement checks both. What remains between the two is the
    singular quadrature's own asymmetry at the order used.
    """
    shape = (3, 2, 2)
    order, near_order = 4, 6
    kernel_of = {"N": (vie.green_n, 6), "K": (vie.green_k, 3)}[operator]
    index, sign = (
        (vie._DYADIC_INDEX, None)
        if operator == "N"
        else (vie._CURL_INDEX, vie._CURL_SIGN)
    )
    reduction = REDUCTIONS[operator][2]
    voxels = list(itertools.product(*(range(n) for n in shape)))

    blocks = {}
    dense = torch.zeros(len(voxels), 12, len(voxels), 12, dtype=torch.complex128)
    for row, observer in enumerate(voxels):
        for column, source in enumerate(voxels):
            offset = tuple(a - b for a, b in zip(observer, source, strict=True))
            if offset not in blocks:
                if max(abs(value) for value in offset) <= 1:
                    pairs = vie._surface_surface(
                        offset,
                        RESOLUTION,
                        WAVENUMBER,
                        near_order,
                        True,
                        reduction,
                        every_pair=True,
                    )
                else:
                    pairs = _volume_every_pair(offset, *kernel_of, order)
                blocks[offset] = _as_matrix(pairs, index, sign)
            block = blocks[offset]
            for p, q, test, basis in itertools.product(
                range(3), range(3), range(4), range(4)
            ):
                dense[row, 4 * p + test, column, 4 * q + basis] = block[
                    test, basis, p, q
                ]

    generator = torch.Generator().manual_seed(0)
    current = torch.randn(12, *shape, dtype=torch.complex128, generator=generator)
    expected = torch.einsum("iajb,jb->ia", dense, current.reshape(12, -1).T).T.reshape(
        12, *shape
    )

    assemble, apply = (
        (vie.kernel_n, vie.apply_n) if operator == "N" else (vie.kernel_k, vie.apply_k)
    )
    kernel = assemble(
        shape,
        RESOLUTION,
        WAVENUMBER,
        far_order=order,
        medium_order=order,
        near_order=near_order,
        linear=True,
    )
    got = apply(tucker.circulant_tucker(kernel, 1e-12), current)
    assert (got - expected).abs().max() <= tolerance * expected.abs().max()


@pytest.mark.parametrize("operator", ["N", "K"])
def test_a_mirrored_offset_carries_each_pair_s_parity_exactly(operator):
    """The compressed product reaches negative offsets through these signs."""
    reduction = REDUCTIONS[operator][2]
    parity = tucker._PARITY_XX_TO_ZZ if operator == "N" else tucker._PARITY_X_TO_Z
    offset = (1, 1, 0)
    here = vie._surface_surface(
        offset, RESOLUTION, WAVENUMBER, 6, True, reduction, every_pair=True
    )
    there = vie._surface_surface(
        (-1, -1, 0), RESOLUTION, WAVENUMBER, 6, True, reduction, every_pair=True
    )
    expected = torch.empty_like(here)
    for test, basis in itertools.product(range(4), repeat=2):
        pair_sign = tucker.pair_parity(test, basis)
        for component in range(here.shape[-1]):
            sign = 1
            for axis in (0, 1):
                sign *= pair_sign[axis] * parity[axis][component]
            expected[test, basis, component] = sign * there[test, basis, component]
    assert (here - expected).abs().max() <= 1e-13 * here.abs().max()


@pytest.mark.slow
@pytest.mark.parametrize("operator", ["N", "K"])
def test_a_swapped_pair_approaches_its_stored_sign_as_the_order_rises(operator):
    """What a swapped pair differs by is the singular quadrature, not its sign."""
    reduction = REDUCTIONS[operator][2]
    defects = []
    for order in (6, 9, 12):
        pairs = vie._surface_surface(
            (1, 1, 0), RESOLUTION, WAVENUMBER, order, True, reduction, every_pair=True
        )
        defect = 0.0
        for test, basis in itertools.product(range(4), repeat=2):
            pair, sign = vie.PAIR_OF[test][basis]
            stored = vie.PAIRS[pair]
            defect = max(
                defect, float((pairs[test, basis] - sign * pairs[stored]).abs().max())
            )
        defects.append(defect / float(pairs.abs().max()))
    assert defects[0] > defects[1] > defects[2]
    assert defects[2] <= 1e-10


def test_the_linear_products_refuse_symbols_built_for_the_constant_basis():
    current = torch.zeros(12, 2, 2, 2, dtype=torch.complex128)
    constant = tucker.circulant_tucker(torch.ones(2, 2, 2, 6, dtype=torch.complex128))
    with pytest.raises(ValueError, match="linear basis"):
        vie.apply_n(constant, current)


def test_the_mass_of_each_linear_function_is_its_square_integrated():
    weights, nodes = gauss_legendre_1d(3)
    cube = torch.cartesian_prod(nodes, nodes, nodes) / 2
    weight = (
        weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    ).reshape(-1) / 8
    functions = torch.cat([torch.ones_like(cube[:, :1]), cube], dim=1)
    integrated = RESOLUTION**3 * (weight[:, None] * functions**2).sum(0)
    torch.testing.assert_close(
        vie.mass(12, RESOLUTION)[:4], integrated, rtol=1e-14, atol=0.0
    )
    torch.testing.assert_close(
        vie.mass(12, RESOLUTION).reshape(3, 4), integrated.expand(3, 4)
    )


def test_the_mass_matrix_and_its_inverse_undo_each_other_in_the_linear_basis():
    generator = torch.Generator().manual_seed(1)
    current = torch.randn(2, 12, 3, 2, 2, dtype=torch.complex128, generator=generator)
    back = vie.apply_inverse_g(vie.apply_g(current, RESOLUTION), RESOLUTION)
    torch.testing.assert_close(back, current)


def test_a_linear_current_goes_to_the_solution_vector_and_back():
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    generator = torch.Generator().manual_seed(2)
    field = torch.randn(12, *body.shape, dtype=torch.complex128, generator=generator)
    field = field * body.mask
    vector = body.to_dof(field)
    assert vector.shape == (12 * body.n_voxels,)
    torch.testing.assert_close(body.from_dof(vector), field)


def test_the_linear_preconditioner_divides_by_each_function_s_mass():
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    medium = Medium(3.0)
    constant = body_diagonal(body, medium)
    linear = body_diagonal(body, medium, linear=True).reshape(12, -1)
    per_voxel = constant.reshape(3, -1)[0]
    for component in range(12):
        factor = 1.0 if component % 4 == 0 else 12.0
        torch.testing.assert_close(linear[component], factor * per_voxel)


def test_a_projected_plane_wave_rebuilds_the_wave_inside_each_voxel():
    """Inside a voxel the wave is a phase ramp; the linear terms carry its slope."""
    medium = Medium(7.0)
    body = VoxelBody.sphere(0.02, 0.005, 52.0, 0.55, padding=0)
    direction = (0.3, -0.2, 0.93)
    projected = plane_wave(body, medium, direction=direction, linear=True)
    sampled = plane_wave(body, medium, direction=direction)
    coefficients = projected.reshape(3, 4, *body.shape)

    torch.testing.assert_close(
        coefficients[:, 0], sampled, rtol=0.0, atol=1e-3 * sampled.abs().max()
    )

    # The slope of exp(-j k u.r) across a voxel is -j k dx u times the field.
    unit = torch.tensor(direction, dtype=torch.float64)
    unit = unit / torch.linalg.vector_norm(unit)
    slope = -1j * medium.wavenumber * body.resolution * unit
    for axis in range(3):
        torch.testing.assert_close(
            coefficients[:, axis + 1],
            slope[axis] * sampled,
            rtol=0.0,
            atol=1e-2 * abs(slope[axis]) * sampled.abs().max(),
        )


def test_the_linear_basis_solves_in_the_steps_the_constant_basis_takes(device):
    """Refining the current inside a voxel adds unknowns, not ill-conditioning.

    In either basis the body equation is the identity minus a compact term, so
    the same sphere at the same tolerance is reached in a comparable number of
    GMRES steps. A left scaling that varies from one basis function to the next
    breaks that, and shows up here as several times the steps.
    """
    body = VoxelBody.sphere(0.03, 0.006, 52.0, 0.55, padding=1, device=device)
    medium = Medium(3.0)
    orders = {"far_order": 4, "medium_order": 6, "near_order": 6}
    steps = []
    for linear in (False, True):
        operator = BodyOperator.build(body, medium, linear=linear, **orders)
        incident = plane_wave(body, medium, linear=linear)
        solution = solve_body(operator, incident, tol=1e-6, maxit=400)
        assert solution.converged
        steps.append(len(solution.residuals) - 1)
    assert steps[1] <= 2 * steps[0]
