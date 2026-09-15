"""The quadrature rules integrate what they claim to integrate."""

import itertools
import math

import pytest
import torch

from mariepy import quadrature

LINE_ORDERS = [1, 2, 3, 4, 6, 8, 15]
DUNAVANT_DEGREES = [1, 2, 3, 4, 5, 6]
TRIANGLE_ORDERS = [2, 3, 4, 5]


def _exact_line_moment(degree):
    """Return the integral of x**degree over [-1, 1]."""
    if degree % 2:
        return 0.0
    return 2.0 / (degree + 1)


def _exact_triangle_moment(p, q):
    """Return the integral of x**p * y**q over the unit reference triangle."""
    return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)


@pytest.mark.parametrize("order", LINE_ORDERS)
def test_gauss_legendre_weights_sum_to_the_length_of_the_interval(order, device):
    weights, _ = quadrature.gauss_legendre_1d(order, device=device)
    assert weights.sum().item() == pytest.approx(2.0, abs=1e-13)


@pytest.mark.parametrize("order", LINE_ORDERS)
def test_gauss_legendre_integrates_polynomials_of_degree_2n_minus_1_exactly(
    order, device
):
    weights, nodes = quadrature.gauss_legendre_1d(order, device=device)
    for degree in range(2 * order):
        got = (weights * nodes**degree).sum().item()
        assert got == pytest.approx(_exact_line_moment(degree), abs=1e-12)


@pytest.mark.parametrize("order", LINE_ORDERS)
def test_gauss_legendre_nodes_lie_inside_the_interval(order, device):
    _, nodes = quadrature.gauss_legendre_1d(order, device=device)
    assert torch.all(nodes > -1.0)
    assert torch.all(nodes < 1.0)
    assert torch.all(nodes[1:] > nodes[:-1])


def test_gauss_legendre_rejects_an_order_below_one():
    with pytest.raises(ValueError, match="at least one point"):
        quadrature.gauss_legendre_1d(0)


@pytest.mark.parametrize("degree", DUNAVANT_DEGREES)
def test_dunavant_weights_sum_to_one(degree, device):
    weights, _ = quadrature.dunavant(degree, device=device)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("degree", DUNAVANT_DEGREES)
def test_dunavant_points_are_barycentric_coordinates_of_interior_points(degree, device):
    _, points = quadrature.dunavant(degree, device=device)
    assert torch.allclose(points.sum(dim=1), torch.ones_like(points[:, 0]), atol=1e-14)
    assert torch.all(points > 0.0)


@pytest.mark.parametrize("degree", DUNAVANT_DEGREES)
def test_dunavant_integrates_monomials_up_to_its_degree_exactly(degree, device):
    weights, points = quadrature.dunavant(degree, device=device)
    x, y = points[:, 1], points[:, 2]
    for p in range(degree + 1):
        for q in range(degree + 1 - p):
            got = 0.5 * (weights * x**p * y**q).sum().item()
            assert got == pytest.approx(_exact_triangle_moment(p, q), abs=1e-14)


@pytest.mark.parametrize("degree", DUNAVANT_DEGREES)
def test_dunavant_is_invariant_under_relabelling_the_triangle_vertices(degree, device):
    """A symmetric rule keeps the same point set when the vertices are permuted."""
    weights, points = quadrature.dunavant(degree, device=device)
    reference = sorted(
        (round(w, 12), *(round(value, 12) for value in sorted(point)))
        for w, point in zip(weights.tolist(), points.tolist(), strict=True)
    )
    for permutation in itertools.permutations(range(3)):
        permuted = points[:, list(permutation)]
        got = sorted(
            (round(w, 12), *(round(value, 12) for value in sorted(point)))
            for w, point in zip(weights.tolist(), permuted.tolist(), strict=True)
        )
        assert got == reference


def test_dunavant_rejects_a_degree_it_does_not_tabulate():
    with pytest.raises(ValueError, match="no Dunavant rule of degree 7"):
        quadrature.dunavant(7)


@pytest.mark.parametrize("order", TRIANGLE_ORDERS)
def test_gauss_triangle_weights_sum_to_one(order, device):
    weights, _ = quadrature.gauss_triangle(order, device=device)
    assert weights.sum().item() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("order", TRIANGLE_ORDERS)
def test_gauss_triangle_points_are_barycentric_coordinates(order, device):
    _, points = quadrature.gauss_triangle(order, device=device)
    assert torch.allclose(points.sum(dim=1), torch.ones_like(points[:, 0]), atol=1e-14)
    assert torch.all(points >= 0.0)


@pytest.mark.parametrize("order", TRIANGLE_ORDERS)
def test_gauss_triangle_integrates_monomials_of_total_degree_2n_minus_2_exactly(
    order, device
):
    """The Duffy Jacobian raises the degree in one axis by one, costing one order."""
    weights, points = quadrature.gauss_triangle(order, device=device)
    x, y = points[:, 1], points[:, 2]
    for total in range(2 * order - 1):
        for p in range(total + 1):
            q = total - p
            got = 0.5 * (weights * x**p * y**q).sum().item()
            assert got == pytest.approx(_exact_triangle_moment(p, q), abs=1e-13)


def test_lebedev_supplies_twenty_six_distinct_directions(device):
    directions = quadrature.lebedev_26_directions(device=device)
    assert directions.shape == (26, 3)
    assert len({tuple(round(v, 12) for v in row) for row in directions.tolist()}) == 26


def test_lebedev_directions_are_unit_vectors(device):
    directions = quadrature.lebedev_26_directions(device=device)
    norms = torch.linalg.vector_norm(directions, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-15)


def test_lebedev_directions_are_invariant_under_the_octahedral_group(device):
    directions = quadrature.lebedev_26_directions(device=device)
    reference = {tuple(round(v, 12) for v in row) for row in directions.tolist()}
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            mapped = directions[:, list(permutation)] * torch.tensor(
                signs, device=directions.device, dtype=directions.dtype
            )
            got = {tuple(round(v, 12) for v in row) for row in mapped.tolist()}
            assert got == reference


def test_lebedev_directions_sum_to_zero(device):
    """Every orbit comes in antipodal pairs, so the rule has no net direction."""
    directions = quadrature.lebedev_26_directions(device=device)
    assert torch.allclose(
        directions.sum(dim=0), torch.zeros_like(directions[0]), atol=1e-15
    )
