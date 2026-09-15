"""Preconditioners for the integral-equation solves.

Ported from MARIE 3.0's ``src_solver/src_preconditioners/prec_vie.m``.

The body block of the operator is dominated by its own Galerkin mass term, so
inverting that term alone conditions the solve well. For the piecewise-constant
basis that term is diagonal, one entry per degree of freedom.
"""

from __future__ import annotations

import torch

from mariepy.body import VoxelBody
from mariepy.constants import Medium

__all__ = ["body_diagonal"]


def body_diagonal(body: VoxelBody, medium: Medium) -> torch.Tensor:
    """Return the diagonal left preconditioner for the body block.

    The Galerkin mass term of the body block is ``res**3 / (j omega eps_0 *
    Mcr)`` on each degree of freedom, so the preconditioner is its reciprocal,
    ``j omega eps_0 * Mcr / res**3``.

    Parameters
    ----------
    body
        Supplies the mask, the grid pitch and the contrast.
    medium
        Supplies the frequency.

    Returns
    -------
    torch.Tensor
        Shape ``(3 * n_voxels,)``, complex, one entry per degree of freedom.
    """
    reduced = body.contrast(medium).reduced[body.mask]
    scaling = torch.tensor(
        medium.electric_scaling, device=reduced.device, dtype=reduced.dtype
    )
    diagonal = scaling * reduced / body.resolution**3
    return diagonal.repeat(3)
