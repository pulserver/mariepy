"""The field of an infinitely long birdcage's linear mode, driving a body without a coil."""

import math

import pytest
import torch

from mariepy import incident
from mariepy.body import VoxelBody
from mariepy.constants import SPEED_OF_LIGHT, VACUUM_PERMEABILITY, Medium

DIRECTIONS = 32


def _body(device):
    return VoxelBody.sphere(0.04, 0.01, 52.0, 0.55, padding=1, device=device)


def _plane_waves(body, medium, angle, field, linear):
    """The mode as a sum of z-polarised plane waves crossing it in the xy-plane.

    ``J_1(u) sin(phi - angle)`` is ``j / (2 pi)`` times the integral over the
    direction ``alpha`` of ``sin(alpha - angle) exp(-j u cos(phi - alpha))``,
    the Jacobi-Anger expansion read backwards, and the trapezoidal rule over a
    period converges geometrically in the number of directions.
    """
    impedance = VACUUM_PERMEABILITY * SPEED_OF_LIGHT
    electric = magnetic = 0.0
    for alpha in (
        2 * math.pi * torch.arange(DIRECTIONS, dtype=torch.float64) / DIRECTIONS
    ):
        alpha = float(alpha)
        wave = incident.plane_wave(
            body,
            medium,
            direction=(math.cos(alpha), math.sin(alpha), 0.0),
            polarisation=(0.0, 0.0, 1.0),
            linear=linear,
        )
        weight = (2 * SPEED_OF_LIGHT * field / DIRECTIONS) * math.sin(alpha - angle)
        electric = electric + weight * wave
        # A plane wave's magnetic field is its direction crossed with its
        # electric field, over the impedance of free space.
        scalar = wave[8:12] if linear else wave[2:3]
        turned = torch.cat(
            [math.sin(alpha) * scalar, -math.cos(alpha) * scalar, 0 * scalar]
        )
        magnetic = magnetic + weight * turned / impedance
    return electric, magnetic


@pytest.mark.parametrize("linear", [False, True], ids=["constant", "linear"])
@pytest.mark.parametrize("angle", [0.0, 0.7])
def test_a_birdcage_mode_is_the_plane_waves_that_cross_it(angle, linear, device):
    medium = Medium(3.0)
    body = _body(device)

    electric, magnetic = incident.birdcage(
        body, medium, angle=angle, field=2e-6, linear=linear
    )
    want_electric, want_magnetic = _plane_waves(body, medium, angle, 2e-6, linear)

    torch.testing.assert_close(
        electric, want_electric, rtol=0, atol=1e-10 * float(want_electric.abs().max())
    )
    torch.testing.assert_close(
        magnetic, want_magnetic, rtol=0, atol=1e-10 * float(want_magnetic.abs().max())
    )


def test_on_its_axis_a_birdcage_mode_has_the_flux_density_asked_for(device):
    medium = Medium(3.0)
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1, device=device)
    middle = tuple(n // 2 for n in body.shape)

    _, magnetic = incident.birdcage(body, medium, angle=math.pi / 3, field=1e-6)

    flux = VACUUM_PERMEABILITY * magnetic[(slice(None), *middle)]
    want = torch.tensor(
        [1e-6 * math.cos(math.pi / 3), 1e-6 * math.sin(math.pi / 3), 0.0],
        dtype=torch.complex128,
        device=device,
    )
    torch.testing.assert_close(flux, want, rtol=0, atol=1e-18)
