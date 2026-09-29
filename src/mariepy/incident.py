"""Incident fields that drive the body without a coil.

MARIE excites the body from a precomputed basis when no coil is present, in
``src_solver/src_ie_solver/ie_solver_vie/ie_solver_vie.m``. A plane wave is the
excitation the analytic Mie series is written for, so it is the one the body
solver is checked against. A mode of an infinitely long birdcage stands in for
a body coil, whose conductors are far enough from a head that their field there
is the coil's own.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from mariepy import vie
from mariepy.body import VoxelBody
from mariepy.constants import SPEED_OF_LIGHT, VACUUM_PERMEABILITY, Medium
from mariepy.quadrature import gauss_legendre_1d

__all__ = ["birdcage", "plane_wave"]

_SMALL_ARGUMENT = 1e-3
"""Below this ``k0 rho`` the Bessel ratios are taken from their series."""


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

    polarised = field.to(torch.complex128)[:, None, None, None]

    def wave(points: torch.Tensor) -> torch.Tensor:
        phase = torch.einsum("a,axyz->xyz", unit, points)
        return polarised * (amplitude * torch.exp(-1j * medium.wavenumber * phase))

    return _in_basis(wave, body, linear=linear, order=order)


def birdcage(
    body: VoxelBody,
    medium: Medium,
    *,
    angle: float = 0.0,
    field: float = 1e-6,
    axis: tuple[float, float] = (0.0, 0.0),
    linear: bool = False,
    order: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one linear mode of an infinitely long birdcage in the body's basis.

    The mode is the transverse-magnetic cylindrical wave about the line
    through ``axis`` along z,

        E_z = -2 j c B J_1(k_0 rho) sin(phi - angle),

    a free-space solution of Maxwell's equations whose magnetic flux density
    on the line is ``B = field`` along ``(cos angle, sin angle, 0)``, with the
    time dependence ``exp(+j omega t)``. Two modes a quarter turn apart, driven
    a quarter period apart, make the rotating field of a quadrature birdcage.
    The fields are sampled or projected as :func:`plane_wave` takes them.

    Parameters
    ----------
    body
        Supplies the grid the mode is taken on.
    medium
        Supplies the frequency.
    angle
        Direction of the magnetic field on the axis, in radians from x
        towards y.
    field
        Magnetic flux density on the axis, in tesla.
    axis
        The (x, y) the axis passes through, in metres.
    linear
        Project onto the linear basis rather than sample at the centres.
    order
        Gauss-Legendre points per axis of the projection.

    Returns
    -------
    electric : torch.Tensor
        The electric field in V/m, shape ``(3, n1, n2, n3)``, complex, or
        ``(12, n1, n2, n3)`` when ``linear``.
    magnetic : torch.Tensor
        The magnetic field in A/m, the same shape.
    """
    wavenumber = medium.wavenumber
    # H = j / (omega mu0) curl E, and E_z is -2 j c field times a real shape.
    scale = 2.0 * field / (wavenumber * VACUUM_PERMEABILITY)
    cos, sin = math.cos(angle), math.sin(angle)

    def shape(points: torch.Tensor):
        """``J_1(k0 rho) sin(phi - angle)`` and its x- and y-derivatives."""
        x = points[0] - axis[0]
        y = points[1] - axis[1]
        u = wavenumber * torch.sqrt(x * x + y * y)
        small = u < _SMALL_ARGUMENT
        safe = torch.where(small, torch.ones_like(u), u)
        ratio = torch.special.bessel_j1(safe) / safe
        ratio = torch.where(small, 0.5 - u * u / 16.0, ratio)
        slope = (torch.special.bessel_j0(safe) - 2.0 * ratio) / (safe * safe)
        slope = torch.where(small, -0.125 + u * u / 96.0, slope)
        across = y * cos - x * sin
        value = wavenumber * ratio * across
        d_x = wavenumber * (slope * wavenumber**2 * x * across - ratio * sin)
        d_y = wavenumber * (slope * wavenumber**2 * y * across + ratio * cos)
        return value, d_x, d_y

    def electric(points: torch.Tensor) -> torch.Tensor:
        value, _, _ = shape(points)
        e_z = (-2j * SPEED_OF_LIGHT * field) * value.to(torch.complex128)
        zero = torch.zeros_like(e_z)
        return torch.stack([zero, zero, e_z])

    def magnetic(points: torch.Tensor) -> torch.Tensor:
        _, d_x, d_y = shape(points)
        return torch.stack([scale * d_y, -scale * d_x, torch.zeros_like(d_x)]).to(
            torch.complex128
        )

    return (
        _in_basis(electric, body, linear=linear, order=order),
        _in_basis(magnetic, body, linear=linear, order=order),
    )


def _in_basis(
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    body: VoxelBody,
    *,
    linear: bool,
    order: int,
) -> torch.Tensor:
    """Sample a field at the voxel centres, or project it onto the linear basis.

    ``evaluate`` takes points ``(3, n1, n2, n3)`` and returns the field there,
    ``(3, n1, n2, n3)``. Projected, each coefficient is the field's moment
    against one of a voxel's four scalar functions, divided by that function's
    mass, ordered ``4 * direction + function``.
    """
    centres = body.coordinates()
    if not linear:
        return evaluate(centres)
    weights, nodes = gauss_legendre_1d(order, device=body.device)
    local = torch.cartesian_prod(nodes, nodes, nodes) / 2.0
    weight = (
        weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
    ).reshape(-1) / 8.0
    functions = torch.cat([torch.ones_like(local[:, :1]), local], dim=1)
    moments = torch.zeros(
        (3, 4, *body.shape), dtype=torch.complex128, device=body.device
    )
    for point, point_weight, point_functions in zip(
        local, weight, functions, strict=True
    ):
        values = evaluate(centres + body.resolution * point[:, None, None, None])
        moments += (
            point_weight
            * values[:, None]
            * point_functions.to(values.dtype)[None, :, None, None, None]
        )
    # Each function's mass over a voxel, in units of the voxel volume.
    masses = vie.mass(12, 1.0)[:4].to(body.device)
    return (moments / masses[None, :, None, None, None]).reshape(12, *body.shape)
