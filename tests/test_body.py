"""The body grid, its contrast and its degree-of-freedom map."""

import math

import pytest
import torch

from mariepy.body import VoxelBody
from mariepy.constants import Medium
from mariepy.preconditioner import body_diagonal

PERMITTIVITY = 52.0
CONDUCTIVITY = 0.55
RADIUS = 0.05
RESOLUTION = 0.01


def _sphere(device, resolution=RESOLUTION, padding=1):
    return VoxelBody.sphere(
        RADIUS,
        resolution,
        PERMITTIVITY,
        CONDUCTIVITY,
        padding=padding,
        device=device,
    )


def _field(body, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shape = (3, *body.shape)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary).to(body.device)


def test_a_voxel_sphere_fills_the_volume_of_the_sphere_it_approximates(device):
    body = _sphere(device, resolution=0.002)
    volume = body.n_voxels * body.resolution**3
    assert volume == pytest.approx(4.0 / 3.0 * math.pi * RADIUS**3, rel=0.02)


def test_the_sphere_mask_is_symmetric_about_the_grid_centre(device):
    body = _sphere(device)
    for axis in range(3):
        assert torch.equal(body.mask, body.mask.flip(axis))


def test_voxels_outside_the_mask_carry_no_scattering_contrast(device):
    body = _sphere(device)
    contrast = body.contrast(Medium(3.0))
    assert torch.allclose(
        contrast.scattering[~body.mask],
        torch.zeros_like(contrast.scattering[~body.mask]),
    )
    assert torch.all(contrast.scattering[body.mask].abs() > 0)


def test_the_contrast_is_the_complex_permittivity_less_one(device):
    body = _sphere(device)
    medium = Medium(3.0)
    contrast = body.contrast(medium)
    expected = PERMITTIVITY + CONDUCTIVITY / medium.electric_scaling
    inside = contrast.relative[body.mask]
    assert torch.allclose(inside, torch.full_like(inside, expected))
    assert torch.allclose(contrast.scattering, contrast.relative - 1.0)
    assert torch.allclose(contrast.reduced, contrast.scattering / contrast.relative)


def test_a_field_restricted_to_the_mask_and_spread_back_keeps_the_masked_part(device):
    body = _sphere(device)
    field = _field(body)
    recovered = body.from_dof(body.to_dof(field))
    mask = body.mask.expand(3, *body.shape)
    assert torch.allclose(recovered[mask], field[mask])
    assert torch.all(recovered[~mask] == 0)


def test_the_degree_of_freedom_map_round_trips_a_solution_vector(device):
    body = _sphere(device)
    generator = torch.Generator(device="cpu").manual_seed(1)
    vector = torch.complex(
        torch.randn(body.n_dof, generator=generator, dtype=torch.float64),
        torch.randn(body.n_dof, generator=generator, dtype=torch.float64),
    ).to(device)
    assert torch.allclose(body.to_dof(body.from_dof(vector)), vector)


def test_the_degree_of_freedom_map_carries_a_leading_port_axis(device):
    body = _sphere(device)
    fields = torch.stack([_field(body, 0), _field(body, 1)])
    vectors = body.to_dof(fields)
    assert vectors.shape == (2, body.n_dof)
    assert torch.allclose(body.to_dof(fields[1]), vectors[1])


def test_there_are_three_degrees_of_freedom_for_every_masked_voxel(device):
    body = _sphere(device)
    assert body.n_dof == 3 * body.n_voxels
    assert body.n_voxels == int(body.mask.sum())


def test_the_grid_coordinates_step_by_the_resolution(device):
    body = _sphere(device)
    coordinates = body.coordinates()
    assert coordinates.shape == (3, *body.shape)
    for axis in range(3):
        step = coordinates[axis].diff(dim=axis)
        assert torch.allclose(step, torch.full_like(step, body.resolution))


def test_a_body_rejects_grids_that_disagree():
    with pytest.raises(ValueError, match="must share a shape"):
        VoxelBody(
            permittivity=torch.ones(2, 2, 2),
            conductivity=torch.ones(3, 3, 3),
            mask=torch.ones(2, 2, 2, dtype=torch.bool),
            resolution=0.01,
        )


def test_a_body_rejects_a_pitch_that_is_not_positive():
    with pytest.raises(ValueError, match="pitch must be positive"):
        VoxelBody(
            permittivity=torch.ones(2, 2, 2),
            conductivity=torch.ones(2, 2, 2),
            mask=torch.ones(2, 2, 2, dtype=torch.bool),
            resolution=0.0,
        )


def test_a_field_of_the_wrong_shape_is_refused(device):
    body = _sphere(device)
    with pytest.raises(ValueError, match="a field must end in"):
        body.to_dof(torch.zeros(3, 2, 2, 2, dtype=torch.complex128, device=device))


def test_a_solution_vector_of_the_wrong_length_is_refused(device):
    body = _sphere(device)
    with pytest.raises(ValueError, match="expected"):
        body.from_dof(torch.zeros(7, dtype=torch.complex128, device=device))


def test_the_body_preconditioner_inverts_the_galerkin_mass_term(device):
    body = _sphere(device)
    medium = Medium(3.0)
    diagonal = body_diagonal(body, medium)
    assert diagonal.shape == (body.n_dof,)

    reduced = body.contrast(medium).reduced[body.mask]
    mass = body.resolution**3 / (medium.electric_scaling * reduced)
    assert torch.allclose(diagonal[: body.n_voxels] * mass, torch.ones_like(mass))


def test_the_body_preconditioner_repeats_across_the_three_components(device):
    body = _sphere(device)
    diagonal = body_diagonal(body, Medium(3.0))
    one = diagonal[: body.n_voxels]
    assert torch.allclose(diagonal[body.n_voxels : 2 * body.n_voxels], one)
    assert torch.allclose(diagonal[2 * body.n_voxels :], one)
