"""Lumped circuits on a multiport, against closed forms."""

import math

import pytest
import torch

from mariepy import circuit, network

OMEGA = 2 * math.pi * 128e6
Z0 = 50.0


def _random_scattering(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(n, n, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(n, n, generator=generator, dtype=torch.float64)
    matrix = torch.complex(real, imaginary)
    return 0.3 * matrix / torch.linalg.matrix_norm(matrix, ord=2)


def _stage(loads, values, quality=None):
    return circuit.Stage(
        loads=tuple(loads),
        values=torch.tensor(values, dtype=torch.complex128),
        quality=None if quality is None else torch.tensor(quality, dtype=torch.float64),
    )


@pytest.mark.parametrize("m", [1, 3])
def test_scattering_and_abcd_undo_each_other(m):
    scattering = _random_scattering(2 * m)
    back = circuit.abcd_to_s(circuit.s_to_abcd(scattering, Z0), Z0)
    torch.testing.assert_close(back, scattering)


def test_the_scattering_matrix_is_the_one_the_network_module_gives():
    generator = torch.Generator().manual_seed(1)
    impedance = torch.complex(
        torch.randn(3, 3, generator=generator, dtype=torch.float64),
        torch.randn(3, 3, generator=generator, dtype=torch.float64),
    )
    torch.testing.assert_close(
        circuit.z_to_s(impedance, Z0), network.z_to_s(impedance, Z0)
    )


def test_a_cascade_of_two_thru_lines_is_a_thru_line():
    zero = torch.zeros(2, 2, dtype=torch.complex128)
    eye = torch.eye(2, dtype=torch.complex128)
    thru = torch.cat([torch.cat([zero, eye], 1), torch.cat([eye, zero], 1)], 0)
    abcd = circuit.s_to_abcd(thru, Z0)
    torch.testing.assert_close(circuit.abcd_to_s(abcd @ abcd, Z0), thru)


def test_an_l_section_matches_its_load_to_the_reference():
    """A series inductor then a parallel capacitor, sized from the closed form."""
    load = torch.tensor([[10.0 - 5.0j]], dtype=torch.complex128)
    r_load, x_load = 10.0, -5.0
    # Series reactance to reach the conductance circle, then a shunt susceptance.
    q = math.sqrt(Z0 / r_load - 1.0)
    x_series = q * r_load - x_load
    b_shunt = q / Z0
    inductance = x_series / OMEGA
    capacitance = b_shunt / OMEGA
    impedance, _ = circuit.place_matching(
        torch.linalg.inv(load),
        (
            _stage(["inductorSeries"], [inductance]),
            _stage(["capacitorParallel"], [capacitance]),
        ),
        OMEGA,
    )
    torch.testing.assert_close(
        impedance, torch.tensor([[Z0 + 0j]], dtype=torch.complex128)
    )


def test_a_series_rlc_is_resistive_at_resonance_and_its_loss_is_its_quality():
    """A resistor terminating a series inductor and capacitor, at resonance."""
    resistance, inductance = 2.0, 50e-9
    capacitance = 1.0 / (OMEGA**2 * inductance)
    quality = 200.0
    load = torch.tensor([[1.0 / resistance]], dtype=torch.complex128)
    lossy, lossless = circuit.place_matching(
        load,
        (
            _stage(["inductorSeries"], [inductance], [quality]),
            _stage(["capacitorSeries"], [capacitance], [quality]),
        ),
        OMEGA,
        lossless=load,
    )
    torch.testing.assert_close(
        lossless, torch.tensor([[resistance + 0j]], dtype=torch.complex128)
    )
    reactance = OMEGA * inductance
    expected = resistance + 2 * reactance / quality
    torch.testing.assert_close(
        lossy, torch.tensor([[expected + 0j]], dtype=torch.complex128)
    )


def test_a_stage_leaves_the_ports_that_do_not_carry_it_alone():
    """The copy of MARIE ported adds each stage only where it sits."""
    generator = torch.Generator().manual_seed(2)
    admittance = torch.complex(
        torch.randn(2, 2, generator=generator, dtype=torch.float64),
        torch.randn(2, 2, generator=generator, dtype=torch.float64),
    )
    impedance, _ = circuit.place_matching(
        admittance, (_stage(["capacitorParallel", ""], [1e-12, 0.0]),), OMEGA
    )
    expected = admittance.clone()
    expected[0, 0] += 1j * OMEGA * 1e-12
    torch.testing.assert_close(impedance, torch.linalg.inv(expected))


def test_eliminating_open_ports_is_inverting_the_kept_block_of_the_impedance():
    generator = torch.Generator().manual_seed(4)
    base = torch.complex(
        torch.randn(5, 5, generator=generator, dtype=torch.float64),
        torch.randn(5, 5, generator=generator, dtype=torch.float64),
    )
    admittance = base + 5 * torch.eye(5, dtype=torch.complex128)
    keep, drop = [0, 3], [1, 2, 4]
    impedance = torch.linalg.inv(admittance)
    torch.testing.assert_close(
        circuit.reduce(admittance, keep, drop),
        torch.linalg.inv(impedance[keep][:, keep]),
    )


def test_the_voltage_map_reproduces_the_open_element_ports():
    """With no current into the element ports, their voltage follows the coil's."""
    generator = torch.Generator().manual_seed(3)
    base = torch.complex(
        torch.randn(4, 4, generator=generator, dtype=torch.float64),
        torch.randn(4, 4, generator=generator, dtype=torch.float64),
    )
    admittance = base + base.transpose(0, 1) + 4 * torch.eye(4, dtype=torch.complex128)
    mapping = circuit.tuning_calibration(admittance, [0, 2], [1, 3])
    voltage = torch.tensor([1.0 + 0.5j, -0.3 + 0j], dtype=torch.complex128)
    everywhere = mapping @ voltage
    current = admittance @ everywhere
    torch.testing.assert_close(everywhere[[0, 2]], voltage)
    torch.testing.assert_close(
        current[[1, 3]], torch.zeros(2, dtype=torch.complex128), atol=1e-12, rtol=0
    )


def test_a_lossless_matched_network_passes_the_incident_power_to_the_coil():
    """Energy is conserved through a lossless network: what arrives is what enters."""
    unmatched = torch.tensor([[0.6 - 0.3j]], dtype=torch.complex128)
    matched = torch.tensor([[0.1 + 0.05j]], dtype=torch.complex128)
    transmission = circuit.matching_calibration(unmatched, matched, Z0)
    arriving = transmission[0, 0]
    delivered = abs(arriving) ** 2 * (1 - abs(unmatched[0, 0]) ** 2)
    accepted = 1 - abs(matched[0, 0]) ** 2
    assert delivered == pytest.approx(float(accepted), rel=1e-10)


def test_the_toeplitz_matrix_takes_its_row_above_and_its_column_below():
    matrix = circuit.toeplitz([0, 1, 2], [0, 5, 6])
    expected = torch.tensor([[0, 1, 2], [5, 0, 1], [6, 5, 0]], dtype=torch.float64)
    torch.testing.assert_close(matrix, expected)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_marie_s_decoupling_weights_count_every_pair_once(n):
    weights = circuit.decoupling_weights(n)
    expected = torch.tril(torch.ones(n, n, dtype=torch.float64), diagonal=-1)
    torch.testing.assert_close(weights, expected)
