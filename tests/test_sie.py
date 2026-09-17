"""The coil impedance matrix: its four blocks, its loads, its drive and its physics."""

import math

import pytest
import torch

from mariepy import network, sie
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.quadrature import gauss_triangle
from tests.pec import scattering_cross_section

RADIUS = 0.05
WIDTH = 0.01
FIELD_STRENGTH = 3.0

# MARIE's GMRES tolerance, which PLAN.md makes the default for every other.
TOLERANCE = 1e-5

# The size parameter the sphere check runs at. Below it the scattering is
# Rayleigh and the discretisation error is the same; above it the mesh has to
# resolve the wave as well as the geometry.
SIZE_PARAMETER = 1.0

# What the icosphere of 320 triangles achieves against the series, measured on
# the mesh SurfaceMesh.sphere(subdivisions=2) builds.
SPHERE_ERROR = 0.039
FINE_SPHERE_ERROR = 0.011


def _medium():
    return Medium(FIELD_STRENGTH)


def _loop(device, n_around=16, n_across=2, ports=1):
    mesh = SurfaceMesh.loop(
        radius=RADIUS,
        width=WIDTH,
        n_around=n_around,
        n_across=n_across,
        ports=ports,
        device=device,
    )
    elements = tuple(
        Port(
            tag=tag,
            kind="port",
            load="capacitorSeries",
            value=0.0,
            quality=1.0,
            voltage=1.0,
        )
        for tag in range(1, ports + 1)
    )
    return SurfaceCoil.build(mesh, elements)


def _sphere(device, subdivisions):
    medium = _medium()
    mesh = SurfaceMesh.sphere(
        radius=SIZE_PARAMETER / medium.wavenumber,
        subdivisions=subdivisions,
        device=device,
    )
    return SurfaceCoil.build(mesh)


def _far_pairs(coil, count=3, reach=3.0):
    """Triangle pairs about ``reach`` element widths apart, so they do not touch."""
    centroids = coil.mesh.centroids()
    separation = torch.cdist(centroids, centroids)
    here, there = torch.nonzero(torch.tril(separation, diagonal=-1) > 0, as_tuple=True)
    target = reach * float(coil.mesh.edge_lengths().mean())
    chosen = torch.argsort((separation[here, there] - target).abs())[:count]
    return torch.stack([here[chosen], there[chosen]], dim=1)


def _cross_section(coil, medium):
    """Take the cross section from the power the incident field does on the current."""
    matrix, _ = sie.impedance(coil, medium)
    drive = sie.plane_wave_excitation(coil, medium)
    current = torch.linalg.solve(matrix, -drive)
    power = 0.5 * torch.real(torch.vdot(-drive, current))
    return 2.0 * medium.impedance * float(power)


def test_the_non_singular_block_reproduces_a_direct_quadrature_of_the_galerkin_entry(
    device,
):
    coil = _loop(device)
    wavenumber = _medium().wavenumber
    pairs = _far_pairs(coil)
    got = sie.near_block(coil, wavenumber, pairs, order=8)

    weights, points = gauss_triangle(16, device=coil.mesh.device)
    weights = (0.5 * weights).to(torch.complex128)
    vertices = coil.mesh.vertices()
    areas = coil.mesh.areas()
    lengths = coil.mesh.edge_lengths()
    signs = coil.signs

    expected = torch.zeros_like(got)
    for row, (here, there) in enumerate(pairs.tolist()):
        observer = torch.einsum("pu,uc->pc", points, vertices[here])
        source = torch.einsum("qv,vc->qc", points, vertices[there])
        distance = torch.linalg.vector_norm(
            observer[:, None, :] - source[None, :, :], dim=-1
        )
        green = torch.exp(-1j * wavenumber * distance) / distance
        for first in range(3):
            scale = signs[here, first] * lengths[here, first]
            basis = scale / (2 * areas[here]) * (observer - vertices[here, first])
            divergence = scale / areas[here]
            for second in range(3):
                other = signs[there, second] * lengths[there, second]
                partner = (
                    other / (2 * areas[there]) * (source - vertices[there, second])
                )
                overlap = torch.einsum(
                    "p,q,pq,pqc->",
                    weights,
                    weights,
                    green,
                    (basis[:, None, :] * partner[None, :, :]).to(torch.complex128),
                )
                scalar = torch.einsum("p,q,pq->", weights, weights, green)
                measure = 4 * areas[here] * areas[there]
                expected[row, first, second] = 1j * wavenumber * measure * overlap + (
                    measure * scalar * divergence * other / areas[there]
                ) / (1j * wavenumber)
    torch.testing.assert_close(
        got, expected, rtol=1e-9, atol=1e-9 * float(expected.abs().max())
    )


@pytest.mark.parametrize("adjacency", ["near", "edge", "vertex"])
def test_each_interaction_block_equals_its_own_transposed_pair(device, adjacency):
    coil = _sphere(device, subdivisions=1)
    wavenumber = _medium().wavenumber
    if adjacency == "near":
        pairs, block = _far_pairs(coil, count=16), sie.near_block
    elif adjacency == "edge":
        pairs, block = coil.adjacency.edge[:16], sie.edge_block
    else:
        pairs, block = coil.adjacency.vertex[:16], sie.vertex_block
    forward = block(coil, wavenumber, pairs)
    backward = block(coil, wavenumber, pairs.flip(1))
    error = (forward - backward.transpose(1, 2)).abs().max() / forward.abs().max()
    assert float(error) <= TOLERANCE


def test_the_self_block_is_symmetric_in_its_two_edges(device):
    coil = _loop(device)
    block = sie.self_block(coil, _medium().wavenumber)
    error = (block - block.transpose(1, 2)).abs().max() / block.abs().max()
    assert float(error) <= 1e-12


def test_the_edge_adjacent_block_becomes_reciprocal_as_its_quadrature_is_refined(
    device,
):
    coil = _loop(device)
    wavenumber = _medium().wavenumber
    pairs = coil.adjacency.edge[:16]
    errors = []
    for order in (6, 20):
        forward = sie.edge_block(coil, wavenumber, pairs, orders=(order, order))
        backward = sie.edge_block(
            coil, wavenumber, pairs.flip(1), orders=(order, order)
        )
        errors.append(
            float(
                (forward - backward.transpose(1, 2)).abs().max() / forward.abs().max()
            )
        )
    assert errors[-1] <= errors[0] / 100.0, errors


def test_the_surface_impedance_block_is_the_gram_matrix_of_the_basis_functions(device):
    coil = _loop(device)
    resistance = _medium().surface_resistance
    got = sie.surface_block(coil, resistance)

    weights, points = gauss_triangle(12, device=coil.mesh.device)
    weights = 0.5 * weights
    vertices = coil.mesh.vertices()
    areas = coil.mesh.areas()
    scale = coil.signs * coil.mesh.edge_lengths()
    quadrature = torch.einsum("pu,nuc->npc", points, vertices)
    basis = (
        scale[:, None, :, None]
        / (2 * areas)[:, None, None, None]
        * (quadrature[:, :, None, :] - vertices[:, None, :, :])
    )
    expected = (
        resistance
        * 2
        * areas[:, None, None]
        * torch.einsum("p,npac,npbc->nab", weights, basis, basis)
    )
    torch.testing.assert_close(got.real, expected, rtol=1e-10, atol=0.0)


def test_the_port_drive_is_minus_the_edge_length_at_its_own_edges_and_zero_elsewhere(
    device,
):
    coil = _loop(device, ports=2)
    drive = sie.port_excitation(coil)
    lengths = coil.dof_lengths().to(torch.complex128)
    assert drive.shape == (2, coil.n_dof)
    for row, port in enumerate(coil.ports):
        torch.testing.assert_close(drive[row, port.dofs], -lengths[port.dofs])
        elsewhere = torch.ones(coil.n_dof, dtype=torch.bool, device=drive.device)
        elsewhere[port.dofs] = False
        assert not bool(drive[row, elsewhere].any())


def test_a_lumped_element_adds_its_impedance_over_the_edges_it_spans(device):
    mesh = SurfaceMesh.loop(
        radius=RADIUS, width=WIDTH, n_around=16, n_across=2, ports=2, device=device
    )
    elements = (
        Port(
            tag=1,
            kind="port",
            load="capacitorSeries",
            value=0.0,
            quality=1.0,
            voltage=1.0,
        ),
        Port(
            tag=2,
            kind="element",
            load="capacitor",
            value=3.3e-12,
            quality=250.0,
            voltage=0.0,
        ),
    )
    coil = SurfaceCoil.build(mesh, elements)
    matrix = torch.zeros(
        (coil.n_dof, coil.n_dof), dtype=torch.complex128, device=coil.mesh.device
    )
    omega = _medium().angular_frequency
    loaded, loss = sie.lumped_loads(matrix, coil, omega)

    load = coil.ports[1]
    value, resistance = load.impedance(omega)
    lengths = coil.dof_lengths().to(torch.complex128)[load.dofs]
    torch.testing.assert_close(
        loaded[load.dofs[:, None], load.dofs[None, :]],
        value * lengths[:, None] * lengths[None, :],
    )
    dense = loss.to_dense()
    torch.testing.assert_close(
        dense[load.dofs[:, None], load.dofs[None, :]],
        resistance * lengths[:, None] * lengths[None, :],
    )
    driven = coil.ports[0]
    assert not bool(loaded[driven.dofs[:, None], driven.dofs[None, :]].any())


def test_a_mutual_inductor_couples_the_edges_of_the_element_it_names(device):
    mesh = SurfaceMesh.loop(
        radius=RADIUS, width=WIDTH, n_around=16, n_across=2, ports=2, device=device
    )
    elements = (
        Port(
            tag=1,
            kind="element",
            load="mutual_inductor",
            value=5e-9,
            quality=100.0,
            voltage=0.0,
            coupled_tag=2,
            coupled_value=1e-9,
        ),
        Port(
            tag=2,
            kind="element",
            load="inductor",
            value=5e-9,
            quality=100.0,
            voltage=0.0,
        ),
    )
    coil = SurfaceCoil.build(mesh, elements)
    matrix = torch.zeros(
        (coil.n_dof, coil.n_dof), dtype=torch.complex128, device=coil.mesh.device
    )
    omega = _medium().angular_frequency
    loaded, _ = sie.lumped_loads(matrix, coil, omega)

    here, there = coil.ports[0].dofs, coil.ports[1].dofs
    lengths = coil.dof_lengths().to(torch.complex128)
    torch.testing.assert_close(
        loaded[here[:, None], there[None, :]],
        1j * omega * 1e-9 * lengths[here][:, None] * lengths[there][None, :],
    )


def test_a_conducting_sphere_scatters_the_cross_section_the_mie_series_gives(device):
    medium = _medium()
    coil = _sphere(device, subdivisions=2)
    expected = scattering_cross_section(SIZE_PARAMETER) / medium.wavenumber**2
    got = _cross_section(coil, medium)
    assert abs(got - expected) / expected <= SPHERE_ERROR


@pytest.mark.slow
def test_the_scattering_error_falls_as_the_sphere_mesh_is_refined(device):
    medium = _medium()
    expected = scattering_cross_section(SIZE_PARAMETER) / medium.wavenumber**2
    errors = [
        abs(_cross_section(_sphere(device, subdivisions), medium) - expected) / expected
        for subdivisions in (1, 2, 3)
    ]
    assert errors == sorted(errors, reverse=True), errors
    assert errors[-1] <= FINE_SPHERE_ERROR


def test_the_loop_coil_carries_the_inductance_its_own_geometry_implies(device):
    medium = _medium()
    coil = _loop(device, n_around=24, n_across=2)
    system = sie.assemble(coil, medium)
    current = torch.linalg.solve(system.impedance, system.excitation.transpose(0, 1))
    admittance = network.port_admittance(system.excitation, current.transpose(0, 1))
    impedance = network.y_to_z(admittance)[0, 0]

    wire = WIDTH / 4  # a flat strip radiates like a round wire a quarter as wide
    inductance = medium.permeability * RADIUS * (math.log(8 * RADIUS / wire) - 2.0)
    assert float(impedance.real) > 0.0
    assert float(impedance.imag) == pytest.approx(
        medium.angular_frequency * inductance, rel=0.2
    )


def test_a_plane_wave_needs_a_polarisation_across_its_own_direction(device):
    coil = _loop(device)
    with pytest.raises(ValueError, match="transverse"):
        sie.plane_wave_excitation(
            coil, _medium(), direction=(0.0, 0.0, 1.0), polarisation=(0.0, 0.0, 2.0)
        )


def test_a_plane_wave_needs_a_direction_of_propagation(device):
    coil = _loop(device)
    with pytest.raises(ValueError, match="direction of propagation"):
        sie.plane_wave_excitation(coil, _medium(), direction=(0.0, 0.0, 0.0))


def test_the_losses_reach_only_the_conductor_and_the_elements_they_sit_on(device):
    mesh = SurfaceMesh.loop(
        radius=RADIUS, width=WIDTH, n_around=16, n_across=2, ports=2, device=device
    )
    elements = (
        Port(
            tag=1,
            kind="port",
            load="capacitorSeries",
            value=0.0,
            quality=1.0,
            voltage=1.0,
        ),
        Port(
            tag=2,
            kind="element",
            load="capacitor",
            value=3.3e-12,
            quality=250.0,
            voltage=0.0,
        ),
    )
    coil = SurfaceCoil.build(mesh, elements)
    system = sie.assemble(coil, _medium())

    assert system.copper_loss.is_sparse
    assert system.lumped_loss.is_sparse
    assert system.loss.is_sparse
    torch.testing.assert_close(
        system.loss.to_dense(),
        system.copper_loss.to_dense() + system.lumped_loss.to_dense(),
    )

    shares = torch.zeros(
        (coil.n_dof, coil.n_dof), dtype=torch.bool, device=coil.mesh.device
    )
    dof = coil.dof_of_triangle()
    for triangle in range(coil.mesh.n_triangles):
        here = dof[triangle][dof[triangle] >= 0]
        shares[here[:, None], here[None, :]] = True
    reached = system.copper_loss.to_dense() != 0
    assert bool((reached <= shares).all())

    element = next(port for port in coil.ports if port.kind == "element")
    spanned = torch.zeros_like(shares)
    spanned[element.dofs[:, None], element.dofs[None, :]] = True
    assert bool(((system.lumped_loss.to_dense() != 0) <= spanned).all())


def test_a_perfect_conductor_loses_nothing_to_copper(device):
    coil = _loop(device)
    system = sie.assemble(coil, _medium(), surface_resistance=0.0)
    assert system.copper_loss.is_sparse
    assert not bool(system.copper_loss.to_dense().any())


def test_two_coils_joined_on_the_diagonal_keep_each_block_where_it_was(device):
    first = sie.assemble(_loop(device, n_around=8, n_across=1), _medium()).copper_loss
    second = sie.assemble(_loop(device, n_around=6, n_across=1), _medium()).copper_loss
    joined = sie.block_diagonal(first, second).to_dense()

    n = first.shape[0]
    torch.testing.assert_close(joined[:n, :n], first.to_dense())
    torch.testing.assert_close(joined[n:, n:], second.to_dense())
    assert not bool(joined[:n, n:].any())
    assert not bool(joined[n:, :n].any())
