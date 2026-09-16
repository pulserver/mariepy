"""Virtual observation points: a small set of matrices that bounds local SAR.

A body model gives one SAR matrix per averaging volume, hundreds of thousands of
them, and a pulse designer needs the largest ``v^H Q v`` over all of them. The
compression of Eichfelder and Gebhardt (MRM 2011, doi 10.1002/mrm.22927)
replaces the set by a few matrices, each dominating a cluster of the originals,
so that the largest ``v^H A v`` over the virtual observation points is never
below the true peak and overestimates it by a bounded amount.

The bound is set by ``margin``, a fraction of the largest eigenvalue in the set:
that is how much spectral norm a cluster's matrix is allowed to add over the
cluster's own worst member.
"""

from __future__ import annotations

import torch

__all__ = ["compress", "dominates"]


def _deficiency(matrix: torch.Tensor) -> torch.Tensor:
    """Give the smallest positive semi-definite ``D`` with ``matrix + D`` still so."""
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * (-values).clamp(min=0)) @ vectors.conj().transpose(-2, -1)


def compress(
    matrices: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a stack of SAR matrices into virtual observation points.

    Parameters
    ----------
    matrices
        Shape ``(n, n_channels, n_channels)``, complex, Hermitian and positive
        semi-definite, as :func:`mariepy.sar.local_matrices` gives them.
    margin
        Overestimation allowed, as a fraction of the largest eigenvalue over the
        whole stack. A cluster grows while the spectral norm its matrix adds
        over the cluster's worst member stays within that.

    Returns
    -------
    vops : torch.Tensor
        Shape ``(n_vops, n_channels, n_channels)``, Hermitian, each dominating
        its cluster.
    cluster : torch.Tensor
        Shape ``(n,)``, which virtual observation point covers each input
        matrix.

    Raises
    ------
    ValueError
        ``matrices`` is not a stack of square matrices, or ``margin`` is
        negative.
    """
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            f"the matrices are (n, n_channels, n_channels), got {tuple(matrices.shape)}"
        )
    if margin < 0.0:
        raise ValueError(
            f"the margin is a fraction of the largest eigenvalue, got {margin}"
        )

    allowed = margin * float(torch.linalg.eigvalsh(matrices)[:, -1].max())
    remaining = matrices
    origin = torch.arange(matrices.shape[0], device=matrices.device)
    cluster = torch.empty(matrices.shape[0], dtype=torch.long, device=matrices.device)
    vops = []

    while remaining.shape[0]:
        worst = remaining[int(torch.linalg.eigvalsh(remaining)[:, -1].argmax())]
        gaps = torch.linalg.eigvalsh(worst[None] - remaining)[:, 0]
        order = torch.argsort(gaps, descending=True, stable=True)
        remaining, origin = remaining[order], origin[order]

        added = torch.zeros_like(worst)
        taken = 0
        while taken < remaining.shape[0]:
            trial = added + _deficiency(worst - remaining[taken])
            # The order puts a matrix that ``worst`` already dominates first, so
            # its deficiency is zero and a cluster is never empty -- except at a
            # zero margin, where rounding in that zero decides, which is what
            # ``taken`` guards.
            if taken and float(torch.linalg.eigvalsh(trial)[-1]) > allowed:
                break
            added, taken = trial, taken + 1

        cluster[origin[:taken]] = len(vops)
        vops.append(worst + added)
        remaining, origin = remaining[taken:], origin[taken:]

    return torch.stack(vops), cluster


def dominates(vops: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Give how far each matrix is from being dominated by a set of points.

    ``A`` dominates ``Q`` when ``A - Q`` is positive semi-definite, so that
    ``v^H Q v <= v^H A v`` for every drive. The value returned is the smallest
    eigenvalue of ``A - Q`` for the best ``A``: non-negative where the bound
    holds.

    Parameters
    ----------
    vops
        Shape ``(n_vops, n_channels, n_channels)``.
    matrices
        Shape ``(n, n_channels, n_channels)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``, real.
    """
    gaps = torch.linalg.eigvalsh(vops[:, None] - matrices[None, :])[..., 0]
    return gaps.max(dim=0).values
