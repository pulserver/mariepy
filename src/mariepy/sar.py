"""Specific absorption rate as a matrix in the channel drives.

Local SAR at a voxel is a quadratic form in the channel drive phasors: for a
drive ``v`` at peak amplitude it is ``v^H Q v`` with

    Q_ij = sigma / (2 rho) * conj(e_i) . e_j,

``e_c`` being channel ``c``'s electric field per unit drive, ``sigma`` the
conductivity and ``rho`` the mass density. ``PLAN.md`` records that convention
and the unit drive it is written in; every matrix here follows it, so each has
units of W/kg per unit drive squared.

Writing SAR this way separates the electromagnetics from the pulse: the
matrices are solved once for a coil and a body, and any drive is evaluated
against them without returning to the fields.
"""

from __future__ import annotations

import torch

from mariepy.fields import at_centres

__all__ = [
    "average",
    "local_matrices",
    "peak",
    "sar",
    "voxel_mass",
]


def _components(field: torch.Tensor) -> torch.Tensor:
    """Read a field at the voxel centres, as ``(n_channels, 3, n1, n2, n3)``."""
    if field.ndim != 5 or field.shape[-4] not in (3, 12):
        raise ValueError(
            "an electric field is (n_channels, c, n1, n2, n3) with c 3 or 12, "
            f"got {tuple(field.shape)}"
        )
    return at_centres(field)


def local_matrices(
    electric: torch.Tensor,
    conductivity: torch.Tensor,
    density: torch.Tensor | float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Give the local SAR matrix of every voxel.

    Parameters
    ----------
    electric
        Each channel's electric field per unit drive, shape
        ``(n_channels, 3, n1, n2, n3)``, or ``(n_channels, 12, n1, n2, n3)`` in
        the piecewise-linear basis, which is read at the cell centres.
    conductivity
        Conductivity in S/m, shape ``(n1, n2, n3)``.
    density
        Mass density in kg/m^3, the same shape or one number for every voxel.
    mask
        Which voxels to take, shape ``(n1, n2, n3)``. All of them by default.

    Returns
    -------
    torch.Tensor
        Shape ``(n_voxels, n_channels, n_channels)``, complex, Hermitian, in
        W/kg per unit drive squared. The voxels are in the mask's own order,
        as :meth:`mariepy.body.VoxelBody.to_dof` orders them.
    """
    centres = _components(electric)
    selected = torch.ones_like(conductivity, dtype=torch.bool) if mask is None else mask
    flat = selected.reshape(-1)
    field = centres.reshape(*centres.shape[:2], -1)[..., flat]
    scale = conductivity.reshape(-1)[flat] / (
        2.0 * _spread(density, conductivity).reshape(-1)[flat]
    )
    return torch.einsum("iav,jav,v->vij", field.conj(), field, scale.to(field.dtype))


def _spread(density: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
    """Give the density as a grid, whether it came as one or as a number."""
    if isinstance(density, torch.Tensor):
        return density
    return torch.full_like(like, float(density))


def voxel_mass(
    density: torch.Tensor | float,
    resolution: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Give the mass of every voxel, in kilograms.

    Parameters
    ----------
    density
        Mass density in kg/m^3, shape ``(n1, n2, n3)`` or one number.
    resolution
        Voxel pitch in metres.
    mask
        Which voxels carry tissue.

    Returns
    -------
    torch.Tensor
        Shape ``(n_voxels,)``, the voxels in the mask's own order.
    """
    grid = _spread(density, mask.to(torch.float64))
    return (grid.reshape(-1)[mask.reshape(-1)] * resolution**3).to(torch.float64)


def sar(matrices: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """Evaluate ``v^H Q v`` for one drive against a stack of matrices.

    Parameters
    ----------
    matrices
        Shape ``(n, n_channels, n_channels)``, Hermitian.
    drive
        The channel phasors at peak amplitude, shape ``(n_channels,)``, or
        several drives, shape ``(n_drives, n_channels)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)`` for one drive, or ``(n_drives, n)``, real, in W/kg.
    """
    phasors = drive.reshape(-1, drive.shape[-1]).to(matrices.dtype)
    values = torch.einsum("di,nij,dj->dn", phasors.conj(), matrices, phasors).real
    return values[0] if drive.ndim == 1 else values


def peak(matrices: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
    """Give the largest ``v^H Q v`` over a stack of matrices.

    Parameters
    ----------
    matrices
        Shape ``(n, n_channels, n_channels)``, Hermitian.
    drive
        Shape ``(n_channels,)`` or ``(n_drives, n_channels)``.

    Returns
    -------
    torch.Tensor
        A scalar, or shape ``(n_drives,)``, in W/kg.
    """
    return sar(matrices, drive).max(dim=-1).values


def average(matrices: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Give the mass-weighted mean of a stack of matrices.

    With one matrix per voxel and each voxel's mass, this is the SAR matrix of
    the whole region: ``v^H Q v`` is the power the drive dissipates there
    divided by its mass.

    Parameters
    ----------
    matrices
        Shape ``(n, n_channels, n_channels)``.
    mass
        Shape ``(n,)``, in kilograms.

    Returns
    -------
    torch.Tensor
        Shape ``(n_channels, n_channels)``, Hermitian.

    Raises
    ------
    ValueError
        The masses sum to zero, so there is nothing to average over.
    """
    total = float(mass.sum())
    if total == 0.0:
        raise ValueError("a mass-weighted mean needs a region of non-zero mass")
    return torch.einsum("nij,n->ij", matrices, mass.to(matrices.dtype)) / total
