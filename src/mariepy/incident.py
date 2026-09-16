"""Incident fields that drive the body without a coil.

MARIE excites the body from a precomputed basis when no coil is present, in
``src_solver/src_ie_solver/ie_solver_vie/ie_solver_vie.m``. A plane wave is the
excitation the analytic Mie series is written for, so it is the one the body
solver is checked against.
"""

from __future__ import annotations

import torch

from mariepy import vie
from mariepy.body import VoxelBody
from mariepy.constants import Medium
from mariepy.quadrature import gauss_legendre_1d

__all__ = ["plane_wave"]


def plane_wave(
    body: VoxelBody,
    medium: Medium,
    *,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    polarisation: tuple[float, float, float] = (1.0, 0.0, 0.0),
    amplitude: complex = 1.0,
    linear: bool = False,
    order: int = 4,
) -> torch.Tensor:
    """Return a plane wave in the body's basis.

    For the constant basis the wave is sampled at the voxel centres. For the
    linear basis it is projected onto each voxel's four scalar functions: each
    coefficient is the wave's moment against that function over the voxel,
    divided by the function's own mass, which is the expansion the linear
    operator's right-hand side takes.

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
    linear
        Project onto the linear basis rather than sample at the centres.
    order
        Gauss-Legendre points per axis of the projection.

    Returns
    -------
    torch.Tensor
        Shape ``(3, n1, n2, n3)``, complex, the electric field at each voxel
        centre; or ``(12, n1, n2, n3)`` when ``linear``, ordered
        ``4 * direction + function``.

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
    if not linear:
        return field.to(torch.complex128)[:, None, None, None] * wave

    weights, nodes = gauss_legendre_1d(order, device=body.device)
    local = torch.cartesian_prod(nodes, nodes, nodes) / 2.0
    weight = (
        weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    ).reshape(-1) / 8.0
    # Across a voxel the wave only picks up the phase of the local offset.
    within = torch.exp(-1j * medium.wavenumber * body.resolution * (local @ unit))
    functions = torch.cat([torch.ones_like(local[:, :1]), local], dim=1)
    moments = torch.einsum(
        "q,qf->f", weight.to(within.dtype) * within, functions.to(within.dtype)
    )
    # Each function's mass over a voxel, in units of the voxel volume.
    masses = (vie.mass(12, 1.0)[:4]).to(body.device)
    coefficients = moments / masses
    return (
        field.to(torch.complex128)[:, None, None, None, None]
        * coefficients[None, :, None, None, None]
        * wave[None, None]
    ).reshape(12, *body.shape)
