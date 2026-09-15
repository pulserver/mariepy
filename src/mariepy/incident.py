"""Incident fields that drive the body without a coil.

MARIE excites the body from a precomputed basis when no coil is present, in
``src_solver/src_ie_solver/ie_solver_vie/ie_solver_vie.m``. A plane wave is the
excitation the analytic Mie series is written for, so it is the one the body
solver is checked against.
"""

from __future__ import annotations

import torch

from mariepy.body import VoxelBody
from mariepy.constants import Medium

__all__ = ["plane_wave"]


def plane_wave(
    body: VoxelBody,
    medium: Medium,
    *,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    polarisation: tuple[float, float, float] = (1.0, 0.0, 0.0),
    amplitude: complex = 1.0,
) -> torch.Tensor:
    """Return a plane wave sampled at the voxel centres.

    Parameters
    ----------
    body
        Supplies the grid the wave is sampled on.
    medium
        Supplies the wavenumber.
    direction
        Propagation direction; normalised here.
    polarisation
        Electric field direction. Its component along ``direction`` is removed,
        since a plane wave is transverse.
    amplitude
        Field strength in V/m.

    Returns
    -------
    torch.Tensor
        Shape ``(3, n1, n2, n3)``, complex, the electric field at each voxel
        centre.

    Raises
    ------
    ValueError
        The direction is zero, or the polarisation is parallel to it.
    """
    unit = torch.tensor(direction, dtype=torch.float64, device=body.device)
    norm = torch.linalg.vector_norm(unit)
    if norm == 0:
        raise ValueError("a plane wave needs a direction of propagation")
    unit = unit / norm

    field = torch.tensor(polarisation, dtype=torch.float64, device=body.device)
    field = field - torch.dot(field, unit) * unit
    transverse = torch.linalg.vector_norm(field)
    if transverse == 0:
        raise ValueError(
            "the polarisation is parallel to the direction of propagation, so "
            "nothing transverse is left of it"
        )
    field = field / transverse

    phase = torch.einsum("a,axyz->xyz", unit, body.coordinates())
    wave = amplitude * torch.exp(-1j * medium.wavenumber * phase)
    return field.to(torch.complex128)[:, None, None, None] * wave
