"""Quadrature rules for the line, the triangle and the sphere.

Two conventions are in use, each the standard one for its domain, and each
asserted by a test:

- a line rule integrates directly, ``∫_{-1}^{1} f dx = Σ w_i f(x_i)``, so its
  weights sum to 2;
- a triangle rule integrates a fraction of the area,
  ``∫_T f dA = area(T) · Σ w_i f(z_i)``, with the points in barycentric
  coordinates, so its weights sum to 1.

The sphere rule supplies directions only.
"""

from __future__ import annotations

import itertools

import torch

__all__ = [
    "dunavant",
    "gauss_legendre_1d",
    "gauss_triangle",
    "lebedev_26_directions",
]

# Dunavant, "High degree efficient symmetrical Gaussian quadrature rules for
# the triangle", Int. J. Numer. Methods Eng. 21 (1985) 1129-1148, Table 1 to
# Table 6. Each entry is one symmetry orbit: the barycentric generator and the
# weight every point of the orbit carries. An orbit of three is the cyclic
# permutations of (a, b, b); an orbit of six is all permutations of three
# distinct coordinates.
_DUNAVANT_ORBITS: dict[int, tuple[tuple[tuple[float, float, float], float], ...]] = {
    1: (((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 1.0),),
    2: (
        (
            (0.666666666666667, 0.166666666666667, 0.166666666666667),
            0.333333333333333,
        ),
    ),
    3: (
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), -0.562500000000000),
        ((0.600000000000000, 0.200000000000000, 0.200000000000000), 0.520833333333333),
    ),
    4: (
        ((0.108103018168070, 0.445948490915965, 0.445948490915965), 0.223381589678011),
        ((0.816847572980459, 0.091576213509771, 0.091576213509771), 0.109951743655322),
    ),
    5: (
        ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.225000000000000),
        ((0.059715871789770, 0.470142064105115, 0.470142064105115), 0.132394152788506),
        ((0.797426985353087, 0.101286507323456, 0.101286507323456), 0.125939180544827),
    ),
    6: (
        ((0.501426509658179, 0.249286745170910, 0.249286745170910), 0.116786275726379),
        ((0.873821971016996, 0.063089014491502, 0.063089014491502), 0.050844906370207),
        ((0.053145049844817, 0.310352451033784, 0.636502499121399), 0.082851075618374),
    ),
}


def gauss_legendre_1d(
    order: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Gauss-Legendre rule of the given order on ``[-1, 1]``.

    The nodes and weights come from the Golub-Welsch eigenvalue problem for the
    Legendre three-term recurrence, so no table is carried.

    Parameters
    ----------
    order
        Number of points. The rule integrates polynomials of degree
        ``2 * order - 1`` exactly.
    device
        Device the result lives on.
    dtype
        Real floating-point dtype of the result.

    Returns
    -------
    weights : torch.Tensor
        Shape ``(order,)``, summing to 2.
    nodes : torch.Tensor
        Shape ``(order,)``, ascending, inside ``(-1, 1)``.

    Raises
    ------
    ValueError
        ``order`` is not positive.
    """
    if order < 1:
        raise ValueError(f"a quadrature rule needs at least one point, got {order}")

    companion = torch.zeros((order, order), device=device, dtype=dtype)
    if order > 1:
        k = torch.arange(1, order, device=device, dtype=dtype)
        off_diagonal = k / torch.sqrt(4.0 * k * k - 1.0)
        companion += torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)

    nodes, vectors = torch.linalg.eigh(companion)
    weights = 2.0 * vectors[0, :] ** 2
    return weights, nodes


def gauss_triangle(
    order: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a tensor-product Gauss rule collapsed onto the reference triangle.

    The square rule of ``order`` points per axis is mapped onto the triangle by
    the Duffy transformation, which absorbs the collapsed edge into the weight.

    Parameters
    ----------
    order
        Number of Gauss-Legendre points per axis; the rule has ``order ** 2``
        points.
    device
        Device the result lives on.
    dtype
        Real floating-point dtype of the result.

    Returns
    -------
    weights : torch.Tensor
        Shape ``(order ** 2,)``, summing to 1.
    points : torch.Tensor
        Shape ``(order ** 2, 3)``, barycentric coordinates summing to 1 in each
        row.
    """
    line_weights, line_nodes = gauss_legendre_1d(order, device=device, dtype=dtype)

    x = line_nodes[:, None].expand(order, order).reshape(-1)
    y = line_nodes[None, :].expand(order, order).reshape(-1)
    weight_x = line_weights[:, None].expand(order, order).reshape(-1)
    weight_y = line_weights[None, :].expand(order, order).reshape(-1)

    z2 = (1.0 + y) / 2.0
    z3 = (1.0 - z2) * (1.0 + x) / 2.0
    z1 = 1.0 - z2 - z3
    weights = weight_x * weight_y * (1.0 - y) / 4.0

    return weights, torch.stack((z1, z2, z3), dim=1)


def dunavant(
    degree: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Dunavant triangle rule exact to the given degree.

    Parameters
    ----------
    degree
        Degree of polynomial the rule integrates exactly, 1 to 6.
    device
        Device the result lives on.
    dtype
        Real floating-point dtype of the result.

    Returns
    -------
    weights : torch.Tensor
        Shape ``(n_points,)``, summing to 1. The degree-3 rule carries a
        negative weight, as the published rule does.
    points : torch.Tensor
        Shape ``(n_points, 3)``, barycentric coordinates summing to 1 in each
        row.

    Raises
    ------
    ValueError
        No rule of that degree is tabulated.
    """
    try:
        orbits = _DUNAVANT_ORBITS[degree]
    except KeyError:
        available = ", ".join(str(key) for key in sorted(_DUNAVANT_ORBITS))
        raise ValueError(
            f"no Dunavant rule of degree {degree}; tabulated degrees are {available}"
        ) from None

    points: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for generator, weight in orbits:
        for point in _orbit(generator):
            points.append(point)
            weights.append(weight)

    return (
        torch.tensor(weights, device=device, dtype=dtype),
        torch.tensor(points, device=device, dtype=dtype),
    )


def _orbit(generator: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    """Return the distinct permutations of one barycentric generator."""
    return sorted(set(itertools.permutations(generator)))


def lebedev_26_directions(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the 26 directions of the degree-7 Lebedev rule on the sphere.

    The precorrected FFT projection places its collocation points along these
    directions and does not use the rule's weights, so only the directions are
    built here: the three orbits of the octahedral group, being the six axes,
    the twelve edge midpoints and the eight cube corners, each normalised.

    Parameters
    ----------
    device
        Device the result lives on.
    dtype
        Real floating-point dtype of the result.

    Returns
    -------
    torch.Tensor
        Shape ``(26, 3)``, each row a unit vector.
    """
    orbits = (
        _signed_placements((1.0, 0.0, 0.0)),
        _signed_placements((1.0, 1.0, 0.0)),
        _signed_placements((1.0, 1.0, 1.0)),
    )
    directions = torch.tensor(
        [point for orbit in orbits for point in orbit], device=device, dtype=dtype
    )
    return directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)


def _signed_placements(pattern: tuple[float, float, float]) -> list[tuple[float, ...]]:
    """Return every distinct sign and axis placement of one coordinate pattern."""
    placements = set()
    for permutation in set(itertools.permutations(pattern)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            placements.add(
                tuple(
                    sign * value for sign, value in zip(signs, permutation, strict=True)
                )
            )
    return sorted(placements)
