"""Co-simulation on small synthetic multiports, against closed forms."""

import json
import math

import pytest
import torch

from mariepy import circuit, cosim

OMEGA = 2 * math.pi * 128e6
Z0 = 50.0
SMALL = cosim.Search(population=60, iterations=150, restarts=3)
LOSSLESS = 1e15


def _write(tmp_path, elements):
    path = tmp_path / "coil.json"
    path.write_text(json.dumps({"coil_configuration": {"elements": elements}}))
    return path


CAPACITIVE = ("capacitorParallel_capacitorSeries", [1e-12, 1e-12], [1e-10, 1e-10])
L_SECTION = ("inductorSeries_capacitorParallel", [1e-9, 1e-12], [5e-8, 5e-10])


def _port(
    number, entity, role="TxRx", optimise=1, symmetry=1, values=None, network=CAPACITIVE
):
    load, minim, maxim = network
    return {
        "number": number,
        "type": "port",
        "load": load,
        "value": values or [2e-11, 5e-12],
        "Q": [LOSSLESS, LOSSLESS],
        "optim": {
            "boolean": optimise,
            "minim": minim,
            "maxim": maxim,
            "symmetry": symmetry,
        },
        "cross_talk": {},
        "excitation": {"entity": entity, "TxRx": role},
    }


def _element(number, entity, symmetry, load="capacitor", value=1e-11, **bounds):
    return {
        "number": number,
        "type": "element",
        "load": load,
        "value": value,
        "Q": LOSSLESS,
        "optim": {
            "boolean": 1,
            "minim": bounds.get("minim", 2e-12),
            "maxim": bounds.get("maxim", 5e-11),
            "symmetry": symmetry,
        },
        "cross_talk": bounds.get("cross_talk", {}),
        "excitation": {"entity": entity, "TxRx": "TxRx"},
    }


PARASITIC = 1e-13


def _loop_impedance(n_loops, coupling=0.05, seed=0):
    """The loop-current impedance of ``n_loops`` neighbour-coupled loops."""
    generator = torch.Generator().manual_seed(seed)
    inductance = 60e-9 * (1 + 0.1 * torch.rand(n_loops, generator=generator))
    resistance = 2.0 * (1 + 0.1 * torch.rand(n_loops, generator=generator))
    loop = torch.diag(resistance + 1j * OMEGA * inductance).to(torch.complex128)
    for i in range(n_loops - 1):
        mutual = coupling * torch.sqrt(inductance[i] * inductance[i + 1])
        loop[i, i + 1] = loop[i + 1, i] = 1j * OMEGA * mutual
    return loop


def _loops(n_loops, coupling=0.05, seed=0):
    """``n_loops`` loops, each broken by a port gap and a tuning-capacitor gap.

    Each gap is bridged by a small parasitic capacitance of impedance ``p``.
    A current ``I`` injected across the gaps and the loop currents ``i`` give
    gap voltages ``V = p (I + P i)``, where ``P`` sends each loop's current
    through its two gaps, and Kirchhoff's law around each loop,
    ``Z_loop i + P^T V = 0``, closes the system:
    ``Z = p I - p^2 P (Z_loop + 2 p I)^-1 P^T``.
    Rows: every loop's port, then every loop's element, as a file listing the
    ports first opens them.
    """
    loop = _loop_impedance(n_loops, coupling, seed)
    p = 1.0 / (1j * OMEGA * PARASITIC)
    eye = torch.eye(n_loops, dtype=torch.complex128)
    through = torch.cat([eye, eye], 0)
    impedance = p * torch.eye(2 * n_loops, dtype=torch.complex128) - p**2 * (
        through @ torch.linalg.solve(loop + 2 * p * eye, through.T)
    )
    return torch.linalg.inv(impedance)


def _loop_network(tmp_path, n_loops, role="TxRx"):
    elements = [
        _port(i + 1, i + 1, role=role, symmetry=i + 1, network=L_SECTION)
        for i in range(n_loops)
    ] + [
        _element(n_loops + i + 1, i + 1, symmetry=n_loops + 1 + i)
        for i in range(n_loops)
    ]
    return cosim.read_network(_write(tmp_path, elements), tmd=True)


# -- reading ------------------------------------------------------------------


def test_symmetric_elements_share_a_variable_and_ports_offset_theirs(tmp_path):
    """MARIE's symmetry arithmetic on the layout of its birdcage file."""
    elements = [_port(1, 1, symmetry=1), _port(2, 2, symmetry=2)] + [
        _element(3 + i, [1, 2], symmetry=3) for i in range(4)
    ]
    network = cosim.read_network(_write(tmp_path, elements), tmd=True)
    assert network.ports == [0, 1]
    assert network.elements == [2, 3, 4, 5]
    groups = [[c.group for c in t.components] for t in network.terminals]
    assert groups == [[1, 2], [4, 5], [6], [6], [6], [6]]
    assert network.entities == [1, 2]
    assert network.rows_of(1) == [0, 2, 3, 4, 5]


def test_without_tmd_only_ports_are_rows_and_nothing_is_searched(tmp_path):
    elements = [_port(1, 1), _element(2, 1, symmetry=3)]
    network = cosim.read_network(_write(tmp_path, elements), tmd=False)
    assert [t.number for t in network.terminals] == [1]
    assert all(c.group is None for c in network.terminals[0].components)


def test_a_fixed_port_takes_no_variable_but_still_offsets_the_count(tmp_path):
    elements = [
        _port(1, 1, optimise=0, symmetry=1),
        _port(2, 1, symmetry=1),
        _element(3, 1, symmetry=1),
    ]
    network = cosim.read_network(_write(tmp_path, elements), tmd=True)
    groups = [[c.group for c in t.components] for t in network.terminals]
    assert groups == [[None, None], [3, 4], [4]]


def test_a_mutual_pair_takes_one_coefficient_bounded_by_its_inductors(tmp_path):
    bounds = {"minim": 10e-9, "maxim": 40e-9}
    elements = [
        _port(1, 1),
        _element(
            2,
            1,
            symmetry=5,
            load="mutual_inductor",
            value=20e-9,
            cross_talk={"coupled_port": 3, "coupled_value": 4e-9},
            **bounds,
        ),
        _element(
            3,
            1,
            symmetry=6,
            load="mutual_inductor",
            value=20e-9,
            cross_talk={"coupled_port": 2, "coupled_value": 4e-9},
            **bounds,
        ),
    ]
    network = cosim.read_network(_write(tmp_path, elements), tmd=True)
    (coupling,) = network.couplings
    assert (coupling.first, coupling.second) == (1, 2)
    assert coupling.minimum == pytest.approx(0.1)
    assert coupling.maximum == pytest.approx(0.4)
    assert network.start()[coupling.group] == pytest.approx(0.2)


def test_mixed_roles_are_refused(tmp_path):
    elements = [_port(1, 1, role="Tx"), _port(2, 2, role="Rx", symmetry=2)]
    network = cosim.read_network(_write(tmp_path, elements), tmd=False)
    with pytest.raises(NotImplementedError, match="mixed roles"):
        cosim.co_simulate(network, torch.eye(2, dtype=torch.complex128), OMEGA)


def test_an_admittance_of_the_wrong_size_is_refused(tmp_path):
    network = _loop_network(tmp_path, 1)
    with pytest.raises(ValueError, match="TMD"):
        cosim.co_simulate(network, torch.eye(1, dtype=torch.complex128), OMEGA)


# -- searching ----------------------------------------------------------------


@pytest.fixture(scope="module")
def tuned_loop(tmp_path_factory):
    network = _loop_network(tmp_path_factory.mktemp("one"), 1, role="Tx")
    admittance = _loops(1)
    result = cosim.co_simulate(
        network, admittance, OMEGA, tuning=SMALL, matching=SMALL, decoupling=SMALL
    )
    return network, admittance, result


def test_one_loop_is_tuned_and_matched_to_the_line(tuned_loop):
    """PLAN.md's milestone 3 criterion, on a loop whose answer is known."""
    _, _, result = tuned_loop
    assert result.costs["tuning"][1] < 1
    assert result.costs["matching"][1] < 1
    assert abs(result.scattering[0, 0]) < 0.02
    assert result.costs["final"] is None


def test_the_tuning_search_resonates_the_loop(tmp_path):
    """Tuning leaves no reactance at the port: the capacitor cancels the loop's."""
    network = _loop_network(tmp_path, 1)
    tuned, costs = cosim._tune(network, _loops(1), OMEGA, 0, SMALL, network.start())
    assert costs[1] < 1
    inductance = float(_loop_impedance(1)[0, 0].imag) / OMEGA
    (capacitance,) = tuned.values()
    assert capacitance == pytest.approx(1.0 / (OMEGA**2 * inductance), rel=0.02)


def test_the_transmit_calibration_puts_the_accepted_power_into_the_structure(
    tuned_loop,
):
    """With lossless elements, the solved structure takes what the port accepts."""
    _, admittance, result = tuned_loop
    voltage = result.transmit[:, 0]
    taken = 0.5 * float((voltage.conj() @ admittance @ voltage).real)
    accepted = 0.5 * (1 - abs(result.scattering[0, 0]) ** 2)
    assert taken == pytest.approx(float(accepted), rel=1e-9)


def test_the_calibration_leaves_no_current_in_the_element_ports_but_their_own(
    tuned_loop,
):
    """The tuning capacitor carries exactly the current the structure sends it."""
    _, admittance, result = tuned_loop
    voltage = result.transmit[:, 0]
    current = admittance @ voltage
    capacitor = 1j * OMEGA * result.values[1][0]
    torch.testing.assert_close(current[1], -capacitor * voltage[1])


def test_the_sweep_passes_through_the_calibrated_point(tuned_loop):
    network, admittance, result = tuned_loop
    swept = cosim.sweep(network, admittance, OMEGA, result, points=101)
    assert float(swept.frequency[swept.index]) == pytest.approx(OMEGA / (2 * math.pi))
    torch.testing.assert_close(
        swept.transmit_scattering[swept.index], result.scattering
    )
    reflection = swept.transmit_scattering[:, 0, 0].abs()
    assert int(reflection.argmin()) == pytest.approx(swept.index, abs=2)
    assert swept.receive_impedance is None


@pytest.mark.parametrize("role", ["Tx", "TxRx"])
def test_two_weakly_coupled_loops_are_matched_together(tmp_path, role):
    """The joint search keeps both ports matched and the calibration lossless."""
    network = _loop_network(tmp_path, 2, role=role)
    admittance = _loops(2, coupling=0.01)
    result = cosim.co_simulate(
        network, admittance, OMEGA, tuning=SMALL, matching=SMALL, decoupling=SMALL
    )
    assert result.costs["final"] is not None
    assert bool((torch.diagonal(result.scattering).abs() < 0.1).all())
    wave = torch.tensor([0.6 + 0.2j, -0.3 + 0.5j], dtype=torch.complex128)
    applied = result.transmit @ wave
    taken = 0.5 * (applied.conj() @ admittance @ applied).real
    reflected = (result.scattering @ wave).abs().square().sum()
    torch.testing.assert_close(taken, 0.5 * (wave.abs().square().sum() - reflected))
    assert (result.receive is None) == (role == "Tx")


# -- file values --------------------------------------------------------------


def test_file_values_close_the_ports_without_searching(tmp_path):
    elements = [
        _port(1, 1, values=[1.2e-11, 7e-12]),
        _port(2, 2, values=[2e-11, 5e-12]),
    ]
    network = cosim.read_network(_write(tmp_path, elements), tmd=False)
    admittance = circuit.place_tuning(
        _loops(2), [2, 3], ["capacitor"] * 2, torch.tensor([2.5e-11, 2.5e-11]), OMEGA
    )
    coil = circuit.reduce(admittance, [0, 1], [2, 3])
    result = cosim.co_simulate(network, coil, OMEGA)
    assert result.costs == {}
    assert result.values == ((1.2e-11, 7e-12), (2e-11, 5e-12))
    stages = (
        circuit.Stage(("capacitorParallel",) * 2, torch.tensor([1.2e-11, 2e-11]), None),
        circuit.Stage(("capacitorSeries",) * 2, torch.tensor([7e-12, 5e-12]), None),
    )
    expected, _ = circuit.place_matching(coil, stages, OMEGA)
    torch.testing.assert_close(result.impedance, expected, rtol=1e-7, atol=0)
    torch.testing.assert_close(
        result.transmit, circuit.coil_voltage(coil, stages, OMEGA, Z0)
    )


def test_the_receive_calibration_terminates_the_other_ports_in_the_preamplifier(
    tmp_path,
):
    elements = [
        _port(1, 1, role="Rx", values=[1.2e-11, 7e-12]),
        _port(2, 2, role="Rx", values=[2e-11, 5e-12]),
    ]
    network = cosim.read_network(_write(tmp_path, elements), tmd=False)
    coil = circuit.reduce(
        circuit.place_tuning(
            _loops(2, coupling=0.2),
            [2, 3],
            ["capacitor"] * 2,
            torch.tensor([2.5e-11, 2.5e-11]),
            OMEGA,
        ),
        [0, 1],
        [2, 3],
    )
    result = cosim.co_simulate(network, coil, OMEGA)
    assert result.transmit is None
    for port, other in ((0, 1), (1, 0)):
        voltage = result.receive[:, port]
        current = coil @ voltage
        torch.testing.assert_close(
            current[other], -voltage[other] / cosim.PREAMPLIFIER_RESISTANCE
        )
    assert result.receive_scattering.shape == (2,)
    swept = cosim.sweep(network, coil, OMEGA, result, points=11)
    torch.testing.assert_close(
        swept.receive_scattering[swept.index], result.receive_scattering
    )


def test_calibrating_per_port_fields_combines_them_by_the_map():
    per_port = torch.randn(3, 2, 4, dtype=torch.complex128)
    mapping = torch.randn(3, 2, dtype=torch.complex128)
    combined = cosim.calibrate(per_port, mapping)
    torch.testing.assert_close(
        combined[1], (mapping[:, 1, None, None] * per_port).sum(0)
    )
