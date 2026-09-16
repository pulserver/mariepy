"""The body solver against the analytic sphere.

The Mie series is the one absolute answer in milestone 1, so this module checks
the reference itself before checking the solver against it.

The piecewise-constant basis approximates the sphere by a staircase, and the
normal electric field jumps across a dielectric boundary by the contrast ratio.
The error is therefore concentrated in the boundary voxels and converges slowly
there, so convergence is measured over the interior.
"""

import itertools

import numpy as np
import pytest
import torch

from mariepy.body import VoxelBody
from mariepy.constants import Medium
from mariepy.incident import plane_wave
from mariepy.solver import BodyOperator, solve_body

from . import mie

RADIUS = 0.05
FIELD_STRENGTH = 3.0

# Measured on the grids named here, at the interior fraction named here. A
# refinement study records these; they are not guaranteed across releases.
COARSEST_RESOLUTION = 0.02
COARSEST_INTERIOR_ERROR = 0.025
COARSEST_LINEAR_INTERIOR_ERROR = 0.022
INTERIOR_FRACTION = 0.6


def _solve(
    radius, resolution, permittivity, conductivity=0.0, device=None, linear=False
):
    medium = Medium(FIELD_STRENGTH)
    body = VoxelBody.sphere(
        radius, resolution, permittivity, conductivity, padding=0, device=device
    )
    operator = BodyOperator.build(
        body,
        medium,
        tol=1e-10,
        far_order=4,
        medium_order=6,
        near_order=6,
        linear=linear,
    )
    incident = plane_wave(body, medium, linear=linear)
    solution = solve_body(operator, incident, tol=1e-10, maxit=400)
    total = operator.total_field(solution.x, incident)
    if linear:
        # The linear functions vanish at the centre, so the constant term is
        # the field there.
        incident = incident.reshape(3, 4, *body.shape)[:, 0]
        total = total.reshape(3, 4, *body.shape)[:, 0]
    return body, medium, incident, solution, total


def _interior_error(
    radius, resolution, permittivity, fraction=INTERIOR_FRACTION, linear=False
):
    body, medium, _, _, total = _solve(radius, resolution, permittivity, linear=linear)
    coordinates = body.coordinates()
    points = torch.stack(
        [coordinates[axis][body.mask] for axis in range(3)], dim=1
    ).numpy()
    got = torch.stack([total[axis][body.mask] for axis in range(3)], dim=1).numpy()
    exact = mie.internal_field(
        points, radius, np.sqrt(complex(permittivity, 0.0)), medium.wavenumber
    )
    inside = np.linalg.norm(points, axis=1) < fraction * radius
    return float(
        np.linalg.norm(got[inside] - exact[inside]) / np.linalg.norm(exact[inside])
    )


def test_the_mie_series_reproduces_the_quasi_static_internal_field():
    """A sphere far smaller than the wavelength holds a uniform 3/(m^2+2) E_0.

    This checks the reference against a closed form, before anything is checked
    against the reference.
    """
    index = np.sqrt(complex(4.0, 0.0))
    points = 1e-5 * np.array(
        [[0.0, 0, 0], [5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0], [-3.0, 2.0, 1.0]]
    )
    field = mie.internal_field(points, 1e-4, index, 1.0)
    expected = 3.0 / (index**2 + 2.0)
    assert np.allclose(field[:, 0], expected, atol=1e-4)
    # The transverse components are the magnetic dipole's correction, of order
    # m k r: here 1e-4 of the field they sit beside.
    assert np.abs(field[:, 1:]).max() <= 2 * abs(index) * 1.0 * 1e-4


def test_a_plane_wave_is_transverse(device):
    body = VoxelBody.sphere(RADIUS, 0.02, 1.0, 0.0, device=device)
    wave = plane_wave(
        body,
        Medium(FIELD_STRENGTH),
        direction=(0.0, 0.0, 1.0),
        polarisation=(1.0, 0.0, 1.0),
    )
    assert torch.allclose(wave[2], torch.zeros_like(wave[2]), atol=1e-14)
    assert torch.linalg.vector_norm(wave[0]) > 0


def test_a_plane_wave_has_unit_magnitude_everywhere(device):
    body = VoxelBody.sphere(RADIUS, 0.02, 1.0, 0.0, device=device)
    wave = plane_wave(body, Medium(FIELD_STRENGTH))
    magnitude = torch.linalg.vector_norm(wave, dim=0)
    assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-12)


def test_a_plane_wave_advances_its_phase_along_the_direction_it_travels(device):
    body = VoxelBody.sphere(RADIUS, 0.02, 1.0, 0.0, device=device)
    medium = Medium(FIELD_STRENGTH)
    wave = plane_wave(body, medium)
    z = body.coordinates()[2]
    expected = torch.exp(-1j * medium.wavenumber * z)
    assert torch.allclose(wave[0], expected.to(wave.dtype), atol=1e-12)


def test_a_plane_wave_refuses_a_polarisation_along_its_direction(device):
    body = VoxelBody.sphere(RADIUS, 0.02, 1.0, 0.0, device=device)
    with pytest.raises(ValueError, match="nothing transverse"):
        plane_wave(
            body, Medium(FIELD_STRENGTH), direction=(0, 0, 1), polarisation=(0, 0, 1)
        )


def test_a_body_with_no_contrast_leaves_the_incident_field_untouched():
    """Free space scatters nothing, so the solve must return exactly zero."""
    _, _, incident, solution, total = _solve(RADIUS, 0.02, 1.0)
    assert torch.linalg.vector_norm(solution.x) == 0.0
    mask = total != 0
    assert torch.allclose(total[mask], incident[mask])


def test_a_sphere_far_smaller_than_the_wavelength_takes_the_quasi_static_field():
    """The internal field tends to 3/(m^2+2) times the incident one."""
    body, _, incident, _, total = _solve(0.004, 0.001, 4.0)
    ratio = (total[0][body.mask] / incident[0][body.mask]).abs()
    assert ratio.mean().item() == pytest.approx(0.5, rel=0.03)


def test_the_solve_reaches_the_tolerance_it_was_given():
    _, _, _, solution, _ = _solve(RADIUS, 0.02, 52.0)
    assert solution.converged
    assert solution.residuals[-1] <= 1e-10


def test_the_interior_field_matches_the_mie_series_on_the_coarsest_grid():
    """Measured on the grid named in this module, not guaranteed elsewhere."""
    error = _interior_error(RADIUS, COARSEST_RESOLUTION, 2.0)
    assert error <= COARSEST_INTERIOR_ERROR, (
        f"interior error {error:.4f} at resolution {COARSEST_RESOLUTION}"
    )


@pytest.mark.slow
def test_the_interior_field_converges_to_the_mie_series_as_the_grid_is_refined():
    errors = [_interior_error(RADIUS, res, 2.0) for res in (0.02, 0.0125, 0.01)]
    assert all(later < earlier for earlier, later in itertools.pairwise(errors)), (
        f"interior errors did not fall: {[f'{e:.4f}' for e in errors]}"
    )


def _quasi_static_error(radius, resolution, permittivity, linear=False):
    """Return how far the internal field is from ``3 / (eps_r + 2)``."""
    body, _, incident, _, total = _solve(
        radius, resolution, permittivity, linear=linear
    )
    deep = body.mask & (
        torch.linalg.vector_norm(body.coordinates(), dim=0) < 0.5 * radius
    )
    got = (total[0][deep] / incident[0][deep]).abs().mean().item()
    expected = 3.0 / (permittivity + 2.0)
    return abs(got - expected) / expected


@pytest.mark.slow
def test_a_high_contrast_sphere_converges_only_as_its_staircase_is_resolved():
    """The boundary, not the wave, sets the error at brain-like permittivity.

    The normal electric field jumps by the contrast ratio across a dielectric
    boundary, and the piecewise-constant basis puts that jump on a staircase.
    The error falls about as fast as the voxel size, so a sphere needs far more
    than ten voxels across its radius before it is worth quoting. MARIE's own
    example uses the piecewise-linear basis of milestone 2 for this reason.
    """
    errors = [_quasi_static_error(0.02, res, 52.0) for res in (0.004, 0.002, 0.00125)]
    assert all(later < earlier for earlier, later in itertools.pairwise(errors)), (
        f"the staircase error did not fall: {[f'{e:.4f}' for e in errors]}"
    )
    assert errors[-1] < 0.2


def test_the_linear_basis_matches_the_mie_series_closer_than_the_constant_one():
    """Measured on the grid named in this module, not guaranteed elsewhere."""
    linear = _interior_error(RADIUS, COARSEST_RESOLUTION, 2.0, linear=True)
    constant = _interior_error(RADIUS, COARSEST_RESOLUTION, 2.0)
    assert linear <= COARSEST_LINEAR_INTERIOR_ERROR, (
        f"linear interior error {linear:.4f} at resolution {COARSEST_RESOLUTION}"
    )
    assert linear < constant, f"linear {linear:.4f}, constant {constant:.4f}"


@pytest.mark.slow
def test_the_linear_basis_converges_to_the_mie_series_as_the_grid_is_refined():
    errors = [
        _interior_error(RADIUS, res, 2.0, linear=True) for res in (0.02, 0.0125, 0.01)
    ]
    assert all(later < earlier for earlier, later in itertools.pairwise(errors)), (
        f"linear interior errors did not fall: {[f'{e:.4f}' for e in errors]}"
    )


@pytest.mark.slow
def test_at_brain_like_contrast_the_linear_basis_cuts_the_staircase_error():
    """The linear basis carries the jump across the boundary that the constant cannot.

    On the grid named here its error is half the constant basis's; the bound
    leaves room for the quasi-static reference's own error at this size.
    """
    linear = _quasi_static_error(0.02, 0.004, 52.0, linear=True)
    constant = _quasi_static_error(0.02, 0.004, 52.0)
    assert linear < 0.6 * constant, f"linear {linear:.4f}, constant {constant:.4f}"


def _random_points(radius, count, seed):
    generator = np.random.default_rng(seed)
    points = generator.uniform(-1, 1, (4 * count, 3))
    points = points[np.linalg.norm(points, axis=1) < 1][:count]
    return points * radius


def test_one_layer_is_the_homogeneous_sphere():
    index = np.conj(np.sqrt(52 - 0.55j / (2 * np.pi * 128e6 * 8.854e-12)))
    points = _random_points(0.1, 200, seed=1)
    wavenumber = 2.68
    homogeneous = mie.internal_field(points, 0.1, index, wavenumber)
    for layers, indices in (([0.1], [index]), ([0.04, 0.1], [index, index])):
        layered = mie.layered_internal_field(points, layers, indices, wavenumber)
        np.testing.assert_allclose(layered, homogeneous, rtol=1e-8, atol=1e-12)


def test_the_layered_field_meets_the_interface_conditions():
    """Tangential E and normal D are continuous across each interface."""
    radii = [0.04, 0.07, 0.1]
    permittivities = np.array([60 - 8j, 40 - 5j, 20 - 3j])
    indices = np.conj(np.sqrt(permittivities))
    directions = _random_points(1.0, 50, seed=2)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for layer, radius in enumerate(radii[:-1]):
        inner = mie.layered_internal_field(
            directions * radius * (1 - 1e-9), radii, indices, 2.68
        )
        outer = mie.layered_internal_field(
            directions * radius * (1 + 1e-9), radii, indices, 2.68
        )
        normal_inner = np.sum(inner * directions, axis=1)
        normal_outer = np.sum(outer * directions, axis=1)
        tangential = (inner - normal_inner[:, None] * directions) - (
            outer - normal_outer[:, None] * directions
        )
        assert np.abs(tangential).max() <= 1e-6 * np.abs(inner).max()
        np.testing.assert_allclose(
            permittivities[layer] * normal_inner,
            permittivities[layer + 1] * normal_outer,
            rtol=1e-6,
        )


@pytest.mark.slow
def test_the_solver_converges_to_the_mie_series_at_brain_permittivity():
    """Where the magnetic multipoles matter, the interior error halves with the voxel."""
    errors = [_interior_error(RADIUS, res, 52.0) for res in (0.01, 0.005)]
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 0.2
