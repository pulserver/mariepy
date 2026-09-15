"""Port admittance, impedance and scattering parameters."""

import pytest
import torch

from mariepy import network

REFERENCE = 50.0


def _random(n_ports, device, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((n_ports, n_ports), generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        (n_ports, n_ports), generator=generator, dtype=torch.float64
    )
    return torch.complex(real, imaginary).to(device)


def test_the_admittance_contracts_each_drive_with_each_solution(device):
    excitation = _random(3, device, seed=1)
    current = _random(3, device, seed=2)
    got = network.port_admittance(excitation, current)
    for here in range(3):
        for there in range(3):
            expected = -(excitation[here] * current[there]).sum()
            torch.testing.assert_close(got[here, there], expected)


def test_the_admittance_of_a_reciprocal_solution_is_already_symmetric(device):
    matrix = _random(4, device, seed=3)
    symmetric = matrix + matrix.transpose(0, 1)
    excitation = _random(4, device, seed=4)
    current = torch.linalg.solve(symmetric, excitation.transpose(0, 1)).transpose(0, 1)
    admittance = network.port_admittance(excitation, current)
    torch.testing.assert_close(admittance, admittance.transpose(0, 1))


def test_symmetrising_keeps_a_symmetric_matrix_and_drops_the_rest(device):
    matrix = _random(3, device, seed=5)
    symmetric = network.symmetrise(matrix)
    torch.testing.assert_close(symmetric, symmetric.transpose(0, 1))
    torch.testing.assert_close(network.symmetrise(symmetric), symmetric)


def test_admittance_and_impedance_invert_each_other(device):
    admittance = _random(3, device, seed=6)
    torch.testing.assert_close(network.z_to_y(network.y_to_z(admittance)), admittance)


def test_a_port_matched_to_the_line_reflects_nothing(device):
    impedance = REFERENCE * torch.eye(2, dtype=torch.complex128, device=device)
    scattering = network.z_to_s(impedance, REFERENCE)
    torch.testing.assert_close(scattering, torch.zeros_like(scattering))


def test_an_open_port_reflects_the_whole_wave_and_a_short_reflects_its_negative(device):
    large = 1e12 * torch.eye(1, dtype=torch.complex128, device=device)
    small = 1e-12 * torch.eye(1, dtype=torch.complex128, device=device)
    assert float(network.z_to_s(large, REFERENCE).real) == pytest.approx(1.0, abs=1e-6)
    assert float(network.z_to_s(small, REFERENCE).real) == pytest.approx(-1.0, abs=1e-6)


def test_a_one_port_reflection_is_the_reflection_coefficient_of_its_own_impedance(
    device,
):
    impedance = torch.tensor([[30.0 + 40.0j]], dtype=torch.complex128, device=device)
    expected = (impedance - REFERENCE) / (impedance + REFERENCE)
    torch.testing.assert_close(network.z_to_s(impedance, REFERENCE), expected)
