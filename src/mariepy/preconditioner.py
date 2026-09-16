"""Preconditioners for the integral-equation solves.

Ported from MARIE 3.0's ``src_solver/src_preconditioners/prec_vie.m``.

The body block of the operator is dominated by its own Galerkin mass term, so
inverting that term alone conditions the solve well. For both bases that term is
diagonal, one entry per degree of freedom.
"""

from __future__ import annotations

import torch

from mariepy import vie
from mariepy.body import VoxelBody
from mariepy.constants import Medium

__all__ = ["body_diagonal"]


def body_diagonal(
    body: VoxelBody, medium: Medium, *, linear: bool = False
) -> torch.Tensor:
    """Return the diagonal left preconditioner for the body block.

    The Galerkin mass term of the body block is ``G / (j omega eps_0 * Mcr)`` on
    each degree of freedom, with ``G`` the mass of its basis function, so the
    preconditioner is its reciprocal, ``j omega eps_0 * Mcr / G``.

    Parameters
    ----------
    body
        Supplies the mask, the grid pitch and the contrast.
    medium
        Supplies the frequency.
    linear
        Build it for the twelve linear basis functions of each voxel.

    Returns
    -------
    torch.Tensor
        Shape ``(3 * n_voxels,)``, or ``(12 * n_voxels,)`` when ``linear``,
        complex, one entry per degree of freedom.
    """
    reduced = body.contrast(medium).reduced[body.mask]
    scaling = torch.tensor(
        medium.electric_scaling, device=reduced.device, dtype=reduced.dtype
    )
    weights = vie.mass(12 if linear else 3, body.resolution).to(reduced.device)
    return (scaling * reduced[None, :] / weights[:, None]).reshape(-1)
