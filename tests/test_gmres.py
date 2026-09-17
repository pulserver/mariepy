"""GMRES reaches the tolerance it was given, and survives a breakdown."""

import pytest
import torch

from mariepy.gmres import gmres, refine


def _system(n, device, seed=0, conditioning=1.0):
    """Return a well-posed complex system and its exact solution."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((n, n), generator=generator, dtype=torch.float64)
    imaginary = torch.randn((n, n), generator=generator, dtype=torch.float64)
    matrix = torch.complex(real, imaginary) / n**0.5
    matrix = matrix + conditioning * torch.eye(n, dtype=torch.complex128)

    real = torch.randn(n, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(n, generator=generator, dtype=torch.float64)
    x = torch.complex(real, imaginary)
    return matrix.to(device), x.to(device)


def _relative_residual(matrix, x, b):
    return (
        torch.linalg.vector_norm(b - matrix @ x) / torch.linalg.vector_norm(b)
    ).item()


@pytest.mark.parametrize("tol", [1e-4, 1e-8, 1e-12])
def test_gmres_reaches_the_tolerance_it_was_given(tol, device):
    matrix, exact = _system(40, device, conditioning=3.0)
    b = matrix @ exact
    solution = gmres(lambda v: matrix @ v, b, tol=tol, restart=40)
    assert solution.converged
    assert _relative_residual(matrix, solution.x, b) <= tol


def test_gmres_agrees_with_a_direct_solve(device):
    matrix, exact = _system(30, device, conditioning=3.0)
    b = matrix @ exact
    solution = gmres(lambda v: matrix @ v, b, tol=1e-12, restart=30)
    assert torch.allclose(solution.x, exact, atol=1e-9)


def test_gmres_solves_in_at_most_n_iterations_without_restarting(device):
    """A Krylov space of full dimension contains the exact solution."""
    n = 25
    matrix, exact = _system(n, device, conditioning=3.0)
    b = matrix @ exact
    solution = gmres(lambda v: matrix @ v, b, tol=1e-13, restart=n, maxit=1)
    assert solution.inner <= n
    assert _relative_residual(matrix, solution.x, b) <= 1e-12


def test_gmres_restarts_when_the_cycle_is_shorter_than_the_problem(device):
    matrix, exact = _system(40, device, conditioning=3.0)
    b = matrix @ exact
    solution = gmres(lambda v: matrix @ v, b, tol=1e-10, restart=5, maxit=200)
    assert solution.restarts > 1
    assert solution.converged


def test_gmres_reports_a_residual_history_that_does_not_rise(device):
    matrix, exact = _system(40, device, conditioning=3.0)
    b = matrix @ exact
    solution = gmres(lambda v: matrix @ v, b, tol=1e-10, restart=40)
    history = solution.residuals
    assert float(history[0]) == pytest.approx(1.0)
    assert torch.all(history[1:] <= history[:-1] * (1.0 + 1e-9))


def test_gmres_starts_from_the_iterate_it_is_given(device):
    matrix, exact = _system(30, device, conditioning=3.0)
    b = matrix @ exact
    warm = gmres(lambda v: matrix @ v, b, tol=1e-12, restart=30, x0=exact.clone())
    assert warm.restarts == 0
    assert torch.allclose(warm.x, exact)


def test_gmres_exits_on_a_breakdown_rather_than_dividing_by_zero(device):
    """An invariant Krylov space makes the Arnoldi subdiagonal vanish.

    A diagonal operator with a right-hand side supported on two eigenvectors
    spans its whole Krylov space in two steps. MARIE divides by the subdiagonal
    without testing it, which gives NaN; the iterate at that step is exact.
    """
    n = 12
    diagonal = torch.arange(1, n + 1, dtype=torch.float64, device=device).to(
        torch.complex128
    )
    b = torch.zeros(n, dtype=torch.complex128, device=device)
    b[0] = 1.0
    b[1] = 1.0

    solution = gmres(lambda v: diagonal * v, b, tol=1e-14, restart=n)
    assert torch.all(torch.isfinite(solution.x))
    assert solution.inner <= 3
    assert torch.allclose(solution.x, b / diagonal, atol=1e-13)


def test_gmres_returns_immediately_for_a_zero_right_hand_side(device):
    n = 8
    diagonal = torch.arange(1, n + 1, dtype=torch.float64, device=device).to(
        torch.complex128
    )
    b = torch.zeros(n, dtype=torch.complex128, device=device)
    solution = gmres(lambda v: diagonal * v, b)
    assert solution.converged
    assert torch.all(solution.x == 0)


def test_a_left_preconditioner_cuts_the_iterations_it_is_meant_to(device):
    """Scaling away a spread of diagonal magnitudes should converge sooner."""
    n = 60
    generator = torch.Generator(device="cpu").manual_seed(3)
    spread = torch.logspace(0, 3, n, dtype=torch.float64).to(device)
    off = torch.complex(
        torch.randn((n, n), generator=generator, dtype=torch.float64),
        torch.randn((n, n), generator=generator, dtype=torch.float64),
    ).to(device) / (20.0 * n**0.5)
    matrix = torch.diag(spread.to(torch.complex128)) + off
    b = torch.ones(n, dtype=torch.complex128, device=device)

    plain = gmres(lambda v: matrix @ v, b, tol=1e-10, restart=n, maxit=50)
    scaled = gmres(
        lambda v: matrix @ v,
        b,
        preconditioner=lambda v: v / spread.to(torch.complex128),
        tol=1e-10,
        restart=n,
        maxit=50,
    )
    assert scaled.inner < plain.inner
    assert _relative_residual(matrix, scaled.x, b) <= 1e-8


def test_gmres_rejects_a_right_hand_side_that_is_not_a_vector():
    with pytest.raises(ValueError, match="must be a vector"):
        gmres(lambda v: v, torch.zeros(3, 3, dtype=torch.complex128))


def test_gmres_rejects_a_restart_cycle_of_no_iterations():
    with pytest.raises(ValueError, match="at least one iteration"):
        gmres(lambda v: v, torch.ones(3, dtype=torch.complex128), restart=0)


def _any_precision(matrix):
    """Apply a matrix in the precision of the vector it is given."""
    return lambda v: matrix.to(v.dtype) @ v


@pytest.mark.parametrize("tol", [1e-4, 1e-8, 1e-12])
def test_refinement_reaches_a_double_precision_tolerance_from_single_precision_products(
    tol, device
):
    matrix, exact = _system(40, device, conditioning=3.0)
    b = matrix @ exact
    solution = refine(_any_precision(matrix), b, tol=tol, restart=40)
    assert solution.converged
    assert solution.x.dtype == torch.complex128
    assert _relative_residual(matrix, solution.x, b) <= tol


def test_refinement_takes_its_products_in_single_precision(device):
    matrix, exact = _system(30, device, conditioning=3.0)
    seen = set()

    def operator(v):
        seen.add(v.dtype)
        return matrix.to(v.dtype) @ v

    refine(operator, matrix @ exact, tol=1e-10, restart=30)
    assert seen == {torch.complex64, torch.complex128}


def test_refinement_takes_a_left_preconditioner_in_double_precision(device):
    matrix, exact = _system(30, device, conditioning=3.0)
    diagonal = torch.linspace(1.0, 50.0, 30, dtype=torch.float64, device=device)
    scaled = diagonal[:, None] * matrix
    b = scaled @ exact
    solution = refine(
        _any_precision(scaled),
        b,
        preconditioner=lambda v: v / diagonal.to(v.dtype),
        tol=1e-10,
        restart=30,
    )
    assert solution.converged
    assert torch.allclose(solution.x, exact, atol=1e-8)


def test_refinement_returns_immediately_for_a_zero_right_hand_side(device):
    matrix, _ = _system(10, device)
    b = torch.zeros(10, dtype=torch.complex128, device=device)
    solution = refine(_any_precision(matrix), b)
    assert solution.converged
    assert not bool(solution.x.any())


def test_single_precision_products_leave_the_iteration_count_of_a_double_solve(device):
    matrix, exact = _system(60, device, conditioning=2.0)
    b = matrix @ exact
    double = gmres(lambda v: matrix @ v, b, tol=1e-6, restart=60)
    mixed = refine(_any_precision(matrix), b, tol=1e-6, restart=60)
    assert len(mixed.residuals) - 1 <= len(double.residuals) - 1 + 3
