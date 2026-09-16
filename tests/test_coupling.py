"""The coupling of a coil basis function to the body, against MARIE and against physics."""

import itertools

import numpy as np
import pytest
import torch

from mariepy import coupling
from mariepy.constants import Medium
from mariepy.quadrature import dunavant, gauss_legendre_1d
from mariepy.vie import _DYADIC_INDEX, green_k, green_n
from tests import parity

FIELD_STRENGTH = 3.0
RESOLUTION = 0.004
TRIANGLE_ORDER = 4
CELL_ORDER = 2


def _medium():
    return Medium(FIELD_STRENGTH)


def _geometry(device, n_rwg=5, n_points=7, seed=11):
    generator = np.random.default_rng(seed)
    corners = torch.tensor(
        generator.normal(scale=0.02, size=(n_rwg, 4, 3)),
        dtype=torch.float64,
        device=device,
    )
    points = torch.tensor(
        generator.normal(scale=0.05, size=(n_points, 3)) + np.array([0.2, 0.0, 0.0]),
        dtype=torch.float64,
        device=device,
    )
    return corners, points


def _pairs(corners, points):
    n_rwg, n_points = corners.shape[0], points.shape[0]
    rows = torch.arange(n_rwg, device=corners.device).repeat_interleave(n_points)
    columns = torch.arange(n_points, device=points.device).repeat(n_rwg)
    return corners[rows], points[columns]


def _basis(corners, order, device):
    """Sample the RWG basis function itself, from its own definition."""
    weights, barycentric = dunavant(order, device=device)
    free = corners[:, :2]
    shared = corners[:, 2:]
    length = torch.linalg.vector_norm(shared[:, 1] - shared[:, 0], dim=-1)
    sign = torch.tensor([1.0, -1.0], dtype=corners.dtype, device=device)

    vertices = torch.stack([free[:, 0], shared[:, 0], shared[:, 1]], dim=1)
    negative = torch.stack([free[:, 1], shared[:, 0], shared[:, 1]], dim=1)
    triangles = torch.stack([vertices, negative], dim=1)
    quadrature = torch.einsum("qu,ntuc->ntqc", barycentric, triangles)
    edge = torch.stack(
        [
            torch.linalg.cross(
                triangles[:, side, 1] - triangles[:, side, 0],
                triangles[:, side, 2] - triangles[:, side, 0],
                dim=-1,
            )
            for side in range(2)
        ],
        dim=1,
    )
    areas = 0.5 * torch.linalg.vector_norm(edge, dim=-1)
    arms = quadrature - free[:, :, None, :]
    current = (
        sign[None, :, None, None]
        * length[:, None, None, None]
        / (2.0 * areas[:, :, None, None])
        * arms
    )
    measure = areas[:, :, None] * weights[None, None, :]
    return quadrature, current, measure


def test_the_electric_coupling_is_the_field_the_basis_function_itself_radiates(device):
    medium = _medium()
    corners, points = _geometry(device, n_rwg=3, n_points=4)
    flat_corners, flat_points = _pairs(corners, points)
    got = coupling.coupling_n(
        flat_corners, flat_points, medium, triangle_order=TRIANGLE_ORDER
    )

    quadrature, current, measure = _basis(flat_corners, TRIANGLE_ORDER, corners.device)
    separation = flat_points[:, None, None, :] - quadrature
    dyadic = green_n(separation, medium.wavenumber)
    expected = torch.stack(
        [
            sum(
                dyadic[..., _DYADIC_INDEX[row][column]] * current[..., column]
                for column in range(3)
            )
            for row in range(3)
        ],
        dim=-1,
    )
    expected = torch.einsum("ntq,ntqc->nc", measure.to(torch.complex128), expected)
    torch.testing.assert_close(got, expected / medium.electric_scaling)


def test_the_magnetic_coupling_is_the_field_the_basis_function_itself_radiates(device):
    medium = _medium()
    corners, points = _geometry(device, n_rwg=3, n_points=4)
    flat_corners, flat_points = _pairs(corners, points)
    got = coupling.coupling_k(
        flat_corners, flat_points, medium, triangle_order=TRIANGLE_ORDER
    )

    quadrature, current, measure = _basis(flat_corners, TRIANGLE_ORDER, corners.device)
    separation = flat_points[:, None, None, :] - quadrature
    expected = torch.linalg.cross(
        green_k(separation, medium.wavenumber), current.to(torch.complex128), dim=-1
    )
    torch.testing.assert_close(
        got, torch.einsum("ntq,ntqc->nc", measure.to(torch.complex128), expected)
    )


@pytest.mark.parametrize("operator", ["n", "k"])
def test_a_cell_average_returns_to_the_value_at_its_centre_as_the_cell_shrinks(
    device, operator
):
    medium = _medium()
    corners, points = _geometry(device, n_rwg=2, n_points=2)
    flat_corners, flat_points = _pairs(corners, points)
    kernel = coupling.coupling_n if operator == "n" else coupling.coupling_k
    at_point = kernel(flat_corners, flat_points, medium)
    errors = []
    for size in (0.02, 0.005, 0.00125):
        averaged = kernel(
            flat_corners, flat_points, medium, cell_size=size, cell_order=CELL_ORDER
        )
        errors.append(float((averaged - at_point).abs().max() / at_point.abs().max()))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] <= 1e-5


@pytest.mark.parametrize("term", [1, 2, 3])
def test_a_linear_cell_basis_weighs_the_field_by_its_first_moment_across_the_cell(
    device, term
):
    medium = _medium()
    corners, points = _geometry(device, n_rwg=2, n_points=2)
    flat_corners, flat_points = _pairs(corners, points)
    shares = []
    for size in (0.008, 0.002, 0.0005):
        constant = coupling.coupling_n(
            flat_corners, flat_points, medium, cell_size=size, cell_order=CELL_ORDER
        )
        linear = coupling.coupling_n(
            flat_corners,
            flat_points,
            medium,
            cell_size=size,
            cell_order=CELL_ORDER,
            basis_term=term,
        )
        shares.append(float(linear.abs().max() / constant.abs().max()))
    for coarse, fine in itertools.pairwise(shares):
        assert fine == pytest.approx(coarse / 4.0, rel=0.02)


def test_the_collocation_matrix_is_the_body_kernel_over_a_cell_small_against_its_distance(
    device,
):
    medium = _medium()
    centres = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    points = torch.tensor(
        [[0.3, 0.1, 0.2], [0.2, 0.3, 0.1]], dtype=torch.float64, device=device
    )
    size = 1e-5
    matrix = coupling.collocation_matrix(centres, points, medium, cell_size=size)

    separation = points[:, None, :] - centres[None, :, :]
    dyadic = size**3 * green_n(separation, medium.wavenumber) / medium.electric_scaling
    for row in range(3):
        for column in range(3):
            block = matrix[row * 2 : (row + 1) * 2, column * 2 : (column + 1) * 2]
            torch.testing.assert_close(
                block, dyadic[..., _DYADIC_INDEX[row][column]], rtol=1e-8, atol=0.0
            )


def test_the_collocation_matrix_carries_one_column_block_for_each_cell_basis_term(
    device,
):
    medium = _medium()
    centres = torch.tensor(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=torch.float64, device=device
    )
    points = torch.tensor([[0.3, 0.1, 0.2]], dtype=torch.float64, device=device)
    constant = coupling.collocation_matrix(
        centres, points, medium, cell_size=RESOLUTION
    )
    linear = coupling.collocation_matrix(
        centres, points, medium, cell_size=RESOLUTION, n_basis=4
    )
    assert constant.shape == (3, 6)
    assert linear.shape == (3, 24)
    for component in range(3):
        torch.testing.assert_close(
            constant[:, component * 2 : component * 2 + 2],
            linear[:, component * 8 : component * 8 + 2],
        )


@pytest.mark.skipif(parity.reason() is not None, reason=parity.reason() or "")
@pytest.mark.parametrize(
    ("index", "variant"),
    list(enumerate(parity.variants())),
    ids=lambda value: str(value),
)
def test_each_coupling_kernel_reproduces_marie_s_own_source(index, variant):
    operator, component, term = variant
    medium = _medium()
    corners, points = _geometry("cpu")
    flat_corners, flat_points = _pairs(corners, points)

    triangle_weights, barycentric = dunavant(TRIANGLE_ORDER)
    cell_weights, cell_nodes = gauss_legendre_1d(CELL_ORDER)
    triangle_rule, cell_rule = parity.packed_rules(
        triangle_weights, barycentric, cell_weights, cell_nodes
    )
    expected = parity.build()(
        index,
        corners.numpy(),
        points.numpy(),
        triangle_rule,
        cell_rule,
        RESOLUTION,
        medium.wavenumber,
    )

    kernel = coupling.coupling_n if operator == "N" else coupling.coupling_k
    got = kernel(
        flat_corners,
        flat_points,
        medium,
        triangle_order=TRIANGLE_ORDER,
        cell_size=RESOLUTION,
        cell_order=CELL_ORDER,
        basis_term=term,
    )[:, component].reshape(corners.shape[0], points.shape[0])
    error = np.abs(got.numpy() - expected).max() / np.abs(expected).max()
    assert error <= 1e-12, error
