"""Restarted GMRES with a left preconditioner.

Ported from MARIE 3.0's ``src_solver/src_iterative_solvers/is_gmres_svie.m``,
``is_iter_gmres_svie.m`` and ``src_mathematics/src_numerical_linear_algebra/
src_inverse/fast_pinv.m``. MathWorks' ``iterchk.m`` and ``iterapp.m``, which
MARIE calls only to apply the operator, are not ported: the operator is called
directly.

The least-squares problem on the Hessenberg matrix is solved by an economy SVD
rather than by Givens rotations, as MARIE solves it. The residual the iteration
reports is the preconditioned one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["Solution", "gmres"]

Operator = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Solution:
    """What one solve returned.

    Attributes
    ----------
    x
        The iterate.
    residuals
        Preconditioned relative residual after each inner iteration, starting
        with the residual of ``x0``.
    inner
        Iterations in the final restart cycle.
    restarts
        Restart cycles taken.
    converged
        Whether the final residual is at or below the tolerance.
    """

    x: torch.Tensor
    residuals: torch.Tensor
    inner: int
    restarts: int
    converged: bool


def gmres(
    operator: Operator,
    b: torch.Tensor,
    *,
    preconditioner: Operator | None = None,
    restart: int = 50,
    tol: float = 1e-5,
    maxit: int = 200,
    x0: torch.Tensor | None = None,
) -> Solution:
    """Solve ``operator(x) = b`` by restarted GMRES.

    Parameters
    ----------
    operator
        Applies the system matrix to a vector.
    b
        Right-hand side, shape ``(n,)``.
    preconditioner
        Applies a left preconditioner. ``None`` runs unpreconditioned, which is
        the body-only path; the coupled path passes the split preconditioner
        that treats the coil and body blocks differently.
    restart
        Iterations per restart cycle.
    tol
        Target for the preconditioned relative residual.
    maxit
        Maximum restart cycles.
    x0
        Starting iterate. ``None`` starts from zero.

    Returns
    -------
    Solution
        The iterate and the residual history.

    Raises
    ------
    ValueError
        ``b`` is not one-dimensional, or ``restart`` is not positive.
    """
    if b.ndim != 1:
        raise ValueError(f"the right-hand side must be a vector, got {b.ndim} axes")
    if restart < 1:
        raise ValueError(f"a restart cycle needs at least one iteration, got {restart}")

    apply_prec = preconditioner if preconditioner is not None else (lambda v: v)

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    residual = b - operator(x) if torch.any(x != 0) else b.clone()

    residual = apply_prec(residual)
    scale = torch.linalg.vector_norm(apply_prec(b))
    if scale == 0:
        return Solution(
            x=x,
            residuals=torch.zeros(1, dtype=b.real.dtype, device=b.device),
            inner=0,
            restarts=0,
            converged=True,
        )

    history = [torch.linalg.vector_norm(residual) / scale]
    restarts = 0
    inner = 0

    while history[-1] > tol and restarts < maxit:
        restarts += 1
        x, residual, inner, cycle = _cycle(
            operator, apply_prec, x, residual, restart, tol * scale
        )
        history.extend(value / scale for value in cycle)

    return Solution(
        x=x,
        residuals=torch.stack(history),
        inner=inner,
        restarts=restarts,
        converged=bool(history[-1] <= tol),
    )


def _cycle(
    operator: Operator,
    apply_prec: Operator,
    x: torch.Tensor,
    residual: torch.Tensor,
    restart: int,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, list[torch.Tensor]]:
    """Run one restart cycle of Arnoldi, and return the updated iterate."""
    beta = torch.linalg.vector_norm(residual)
    basis = torch.zeros(
        (residual.shape[0], restart + 1), dtype=residual.dtype, device=residual.device
    )
    basis[:, 0] = residual / beta

    hessenberg = torch.zeros(
        (restart + 1, restart), dtype=residual.dtype, device=residual.device
    )
    history: list[torch.Tensor] = []
    y = torch.zeros(0, dtype=residual.dtype, device=residual.device)
    defect = torch.zeros(1, dtype=residual.dtype, device=residual.device)
    k = 0

    for k in range(1, restart + 1):
        w = apply_prec(operator(basis[:, k - 1]))
        hessenberg[:k, k - 1] = basis[:, :k].conj().transpose(0, 1) @ w
        w = w - basis[:, :k] @ hessenberg[:k, k - 1]

        subdiagonal = torch.linalg.vector_norm(w)
        hessenberg[k, k - 1] = subdiagonal

        y, defect = _least_squares(hessenberg[: k + 1, :k], beta)
        history.append(torch.linalg.vector_norm(defect))

        # A vanishing subdiagonal means the Krylov space is invariant and the
        # iterate this step gives is exact; dividing by it would give NaN.
        if subdiagonal <= torch.finfo(residual.real.dtype).eps * beta:
            break
        basis[:, k] = w / subdiagonal

        if history[-1] < target:
            break

    x = x + basis[:, :k] @ y
    residual = basis[:, : defect.shape[0]] @ defect
    return x, residual, k, history


def _least_squares(
    hessenberg: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimise ``|beta e_1 - H y|`` through an economy SVD of ``H``.

    Parameters
    ----------
    hessenberg
        Shape ``(k + 1, k)``.
    beta
        Norm of the residual the cycle started from.

    Returns
    -------
    y : torch.Tensor
        The minimiser, shape ``(k,)``.
    defect : torch.Tensor
        ``beta e_1 - H y``, shape ``(k + 1,)``.
    """
    left, singular_values, right = torch.linalg.svd(hessenberg, full_matrices=False)
    keep = singular_values > torch.finfo(singular_values.dtype).eps * singular_values[0]
    inverse = torch.where(
        keep, 1.0 / singular_values, torch.zeros_like(singular_values)
    )
    y = (right.conj().transpose(0, 1) * inverse) @ (left[0, :].conj() * beta)

    rhs = torch.zeros(
        hessenberg.shape[0], dtype=hessenberg.dtype, device=hessenberg.device
    )
    rhs[0] = beta.to(hessenberg.dtype)
    return y, rhs - hessenberg @ y
