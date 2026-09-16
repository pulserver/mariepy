"""Wire coils: their loops, their own matrix and their coupling to the body."""

import itertools
import math

import numpy as np
import pytest
import torch

from mariepy import network, pfft, vie, wire
from mariepy.body import VoxelBody
from mariepy.coil import Port
from mariepy.constants import VACUUM_PERMEABILITY, Medium
from mariepy.quadrature import gauss_legendre_1d
from mariepy.solver import solve
from mariepy.wire import WireCoil
from tests import parity

from .marie_files import write_wire_gmsh22

FIELD_STRENGTH = 3.0
LOOP_RADIUS = 0.05
WIRE_RADIUS_TEST = 0.0005


def _port(tag, kind="port", load="none", value=0.0, **extra):
    return Port(
        tag=tag,
        kind=kind,
        load=load,
        value=value,
        quality=extra.pop("quality", 1e12),
        voltage=1.0 if kind == "port" else 0.0,
        **extra,
    )


# -- geometry ------------------------------------------------------------------


def test_each_basis_function_peaks_at_its_own_node_between_its_neighbours():
    coil = WireCoil.loop(LOOP_RADIUS, 8)
    torch.testing.assert_close(coil.first, torch.roll(coil.centre, 1, dims=0))
    torch.testing.assert_close(coil.last, torch.roll(coil.centre, -1, dims=0))
    assert coil.loops == ((0, 8),)
    assert coil.following(7) == 0


def test_a_port_spans_the_two_basis_functions_of_the_segment_it_sits_on():
    coil = WireCoil.loop(LOOP_RADIUS, 8, (_port(1), _port(2, "element")), at=(7, 3))
    assert [port.dofs.tolist() for port in coil.ports] == [[7, 0], [3, 4]]
    assert coil.n_driven == 1


def test_two_loops_in_one_file_are_two_loops(tmp_path):
    first = WireCoil.loop(LOOP_RADIUS, 6)
    nodes = torch.cat([first.centre, first.centre + 0.2])
    segments = [(k, (k + 1) % 6) for k in range(6)]
    segments += [(6 + k, 6 + (k + 1) % 6) for k in range(6)]
    coil = WireCoil.build(nodes, segments, [11], (_port(1),))
    assert coil.loops == ((0, 6), (6, 12))
    assert coil.ports[0].dofs.tolist() == [11, 6]


def test_an_open_wire_carries_a_basis_function_at_each_interior_node():
    nodes = torch.rand(5, 3, dtype=torch.float64)
    segments = [(0, 1), (1, 2), (2, 3), (3, 4)]
    elements = (_port(1),)
    coil = WireCoil.build(nodes, segments, [1], elements)
    torch.testing.assert_close(coil.centre, nodes[1:4])
    torch.testing.assert_close(coil.first, nodes[0:3])
    torch.testing.assert_close(coil.last, nodes[2:5])
    assert coil.closed == (False,)
    assert coil.following(2) is None
    assert coil.ports[0].dofs.tolist() == [0, 1]
    assert coil.points().shape == (5, 3)


def test_a_port_on_an_open_wire_s_end_segment_is_refused():
    nodes = torch.rand(5, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="end segment"):
        WireCoil.build(nodes, [(0, 1), (1, 2), (2, 3), (3, 4)], [0], (_port(1),))


def test_a_loop_and_an_open_wire_in_one_file_are_told_apart():
    nodes = torch.rand(8, 3, dtype=torch.float64)
    segments = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 7)]
    coil = WireCoil.build(nodes, segments, [])
    assert coil.loops == ((0, 3), (3, 6))
    assert coil.closed == (True, False)


def _dipole(length, n_segments, medium_radius=WIRE_RADIUS_TEST):
    z = torch.linspace(-length / 2, length / 2, n_segments + 1, dtype=torch.float64)
    nodes = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    segments = [(k, k + 1) for k in range(n_segments)]
    return WireCoil.build(
        nodes, segments, [n_segments // 2], (_port(1),), radius=medium_radius
    )


def test_a_half_wave_dipole_has_its_textbook_input_impedance():
    """A centre-fed half-wave dipole of finite radius.

    The infinitely thin dipole's 73 + j42 ohms rises for a wire a few
    ten-thousandths of a wavelength thick; this one gives about 82 + j46.
    """
    medium = Medium(3.0)
    wavelength = 2 * math.pi / medium.wavenumber
    coil = _dipole(wavelength / 2, 80)
    system = wire.assemble(coil, medium)
    current = torch.linalg.solve(system.impedance, system.excitation.T).T
    admittance = network.port_admittance(system.excitation, current)
    impedance = complex(1 / admittance[0, 0])
    assert 75 < impedance.real < 90
    assert 35 < impedance.imag < 55


def test_a_wire_mesh_reads_back_the_loop_it_was_written_from(tmp_path):
    path = tmp_path / "loop.msh"
    write_wire_gmsh22(path, n_segments=10, port_nodes=[0, 5], radius=LOOP_RADIUS)
    coil = WireCoil.read_gmsh22(path, (_port(1), _port(2, "element")))
    built = WireCoil.loop(LOOP_RADIUS, 10)
    torch.testing.assert_close(coil.centre, built.centre)
    assert [port.dofs.tolist() for port in coil.ports] == [[0, 1], [5, 6]]


# -- the wire's own matrix -----------------------------------------------------


@pytest.fixture(scope="module")
def slow_loop():
    """A loop at a frequency low enough to be quasi-static, with one port."""
    medium = Medium(0.05)
    coil = WireCoil.loop(LOOP_RADIUS, 48, (_port(1),))
    return medium, coil, wire.assemble(coil, medium)


def test_the_wire_matrix_is_symmetric(slow_loop):
    _, _, system = slow_loop
    torch.testing.assert_close(system.impedance, system.impedance.T)


def test_a_slow_loop_has_the_inductance_of_a_thin_ring(slow_loop):
    """``L = mu0 R (ln(8R/a) - 2)``, the thin circular loop's inductance."""
    medium, coil, system = slow_loop
    current = torch.linalg.solve(system.impedance, system.excitation.T).T
    admittance = network.port_admittance(system.excitation, current)
    impedance = 1 / admittance[0, 0]
    inductance = float(impedance.imag) / medium.angular_frequency
    ring = (
        VACUUM_PERMEABILITY
        * LOOP_RADIUS
        * (math.log(8 * LOOP_RADIUS / coil.radius) - 2)
    )
    assert inductance == pytest.approx(ring, rel=0.03)
    assert float(impedance.real) > 0


def test_the_copper_loss_is_the_skin_resistance_over_each_hat(slow_loop):
    medium, coil, system = slow_loop
    depth = medium.skin_depth
    per_length = 1 / (5.96e7 * math.pi * (2 * coil.radius - depth) * depth)
    hat = (coil.left_lengths() + coil.right_lengths()) / 3
    torch.testing.assert_close(
        torch.diagonal(system.copper_loss).real, per_length * hat
    )


def test_an_element_puts_half_its_impedance_on_each_of_its_basis_functions():
    medium = Medium(FIELD_STRENGTH)
    omega = medium.angular_frequency
    elements = (
        _port(1),
        _port(2, "element", "capacitor", 5e-12, quality=100.0),
        _port(3, "element", "mutual_inductor", 1e-8, coupled_tag=4, coupled_value=2e-9),
        _port(4, "element", "mutual_inductor", 1e-8, coupled_tag=3, coupled_value=2e-9),
    )
    coil = WireCoil.loop(LOOP_RADIUS, 12, elements, at=(0, 3, 6, 9))
    bare, _ = wire.impedance(coil, medium)
    loaded, loss = wire.lumped_loads(bare, coil, omega)
    added = loaded - bare
    capacitor = 1 / (1j * omega * 5e-12) + 1 / (omega * 5e-12 * 100.0)
    assert complex(added[3, 3]) == pytest.approx(0.5 * capacitor)
    assert complex(added[4, 4]) == pytest.approx(0.5 * capacitor)
    assert float(loss[3, 3].real) == pytest.approx(0.5 / (omega * 5e-12 * 100.0))
    mutual = 0.5j * omega * 2e-9
    assert complex(added[6, 9]) == pytest.approx(mutual)
    assert complex(added[7, 10]) == pytest.approx(mutual)
    assert complex(added[9, 6]) == pytest.approx(mutual)


def test_a_port_drives_half_its_voltage_into_each_of_its_basis_functions():
    coil = WireCoil.loop(LOOP_RADIUS, 12, (_port(1), _port(2)), at=(2, 8))
    excitation = wire.port_excitation(coil)
    expected = torch.zeros((2, 12), dtype=torch.complex128)
    expected[0, [2, 3]] = -0.5
    expected[1, [8, 9]] = -0.5
    torch.testing.assert_close(excitation, expected)


# -- coupling to the body ------------------------------------------------------


def _random_wire(n_dof, seed):
    generator = np.random.default_rng(seed)
    centre = generator.normal(scale=0.02, size=(n_dof, 3))
    first = centre + generator.normal(scale=0.01, size=(n_dof, 3))
    last = centre + generator.normal(scale=0.01, size=(n_dof, 3))
    return WireCoil(
        first=torch.tensor(first),
        centre=torch.tensor(centre),
        last=torch.tensor(last),
        loops=((0, n_dof),),
    )


WIRE_ORDER = 4
CELL_ORDER = 2
RESOLUTION = 0.004


@pytest.mark.skipif(parity.reason() is not None, reason=str(parity.reason()))
@pytest.mark.parametrize(("operator", "component", "term"), list(parity.variants()))
def test_the_wire_coupling_is_marie_s_with_the_falling_segment_falling(
    operator, component, term
):
    medium = Medium(FIELD_STRENGTH)
    coil = _random_wire(4, seed=5)
    offset = np.array([0.2, 0.0, 0.0])
    points = np.random.default_rng(6).normal(scale=0.05, size=(5, 3)) + offset
    points = torch.tensor(points)
    index = list(parity.variants()).index((operator, component, term))

    wire_weights, wire_nodes = gauss_legendre_1d(WIRE_ORDER, dtype=torch.float64)
    cell_weights, cell_nodes = gauss_legendre_1d(CELL_ORDER, dtype=torch.float64)
    expected = parity.build_wire()(
        index,
        coil.first.numpy(),
        coil.centre.numpy(),
        coil.last.numpy(),
        points.numpy(),
        parity.packed_line_rule(wire_weights.numpy(), wire_nodes.numpy()),
        parity.packed_line_rule(cell_weights.numpy(), cell_nodes.numpy()),
        RESOLUTION,
        medium.wavenumber,
    )

    kernel = wire.coupling_k if operator == "K" else wire.coupling_n
    dofs, observers = zip(
        *itertools.product(range(coil.n_dof), range(points.shape[0])), strict=True
    )
    got = kernel(
        coil,
        torch.tensor(dofs),
        points[list(observers)],
        medium,
        order=WIRE_ORDER,
        cell_size=RESOLUTION,
        cell_order=CELL_ORDER,
        basis_term=term,
    )[:, component].reshape(coil.n_dof, points.shape[0])
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-8, atol=0)


@pytest.mark.skipif(parity.reason() is not None, reason=str(parity.reason()))
def test_marie_s_wire_sources_as_shipped_give_the_falling_segment_a_rising_ramp():
    """The departure the port makes is a real one: the shipped sources differ."""
    medium = Medium(FIELD_STRENGTH)
    coil = _random_wire(2, seed=7)
    points = np.array([[0.2, 0.0, 0.0]])
    weights, nodes = gauss_legendre_1d(WIRE_ORDER, dtype=torch.float64)
    rule = parity.packed_line_rule(weights.numpy(), nodes.numpy())
    arguments = (
        coil.first.numpy(),
        coil.centre.numpy(),
        coil.last.numpy(),
        points,
        rule,
        parity.packed_line_rule([2.0], [0.0]),
        RESOLUTION,
        medium.wavenumber,
    )
    fixed = parity.build_wire(True)(0, *arguments)
    shipped = parity.build_wire(False)(0, *arguments)
    assert np.abs(fixed - shipped).max() > 1e-3 * np.abs(fixed).max()


def test_a_straight_hat_s_field_mirrors_about_its_peak():
    """Symmetric current, antisymmetric charge: E_z and H_y even, E_x odd."""
    medium = Medium(FIELD_STRENGTH)
    half = 0.01
    coil = WireCoil(
        first=torch.tensor([[0.0, 0.0, -half]], dtype=torch.float64),
        centre=torch.zeros((1, 3), dtype=torch.float64),
        last=torch.tensor([[0.0, 0.0, half]], dtype=torch.float64),
        loops=((0, 1),),
    )
    points = torch.tensor([[0.03, 0.0, 0.02], [0.03, 0.0, -0.02]], dtype=torch.float64)
    dofs = torch.zeros(2, dtype=torch.long)
    electric = wire.coupling_n(coil, dofs, points, medium, order=8)
    magnetic = wire.coupling_k(coil, dofs, points, medium, order=8)
    torch.testing.assert_close(electric[0, 2], electric[1, 2])
    torch.testing.assert_close(electric[0, 0], -electric[1, 0])
    torch.testing.assert_close(magnetic[0, 1], magnetic[1, 1])


def test_the_wire_field_is_the_body_kernel_applied_to_the_hat():
    """Against the dyadic kernel sampled along the wire on a fine rule."""
    medium = Medium(FIELD_STRENGTH)
    coil = _random_wire(1, seed=9)
    point = torch.tensor([[0.15, 0.02, -0.01]], dtype=torch.float64)
    got = wire.coupling_n(
        coil, torch.zeros(1, dtype=torch.long), point, medium, order=12
    )

    weights, nodes = gauss_legendre_1d(40, dtype=torch.float64)
    u = (nodes + 1) / 2
    total = torch.zeros(3, dtype=torch.complex128)
    for start, stop, shape in (
        (coil.first[0], coil.centre[0], u),
        (coil.centre[0], coil.last[0], 1 - u),
    ):
        length = torch.linalg.vector_norm(stop - start)
        along = start + u[:, None] * (stop - start)
        current = (shape[:, None] * (stop - start) / length).to(torch.complex128)
        dyadic = vie.green_n(point - along, medium.wavenumber)
        rows = torch.stack(
            [
                sum(
                    dyadic[:, vie._DYADIC_INDEX[row][column]] * current[:, column]
                    for column in range(3)
                )
                for row in range(3)
            ],
            dim=-1,
        )
        total += (length / 2 * weights[:, None] * rows).sum(0)
    torch.testing.assert_close(got[0], total / medium.electric_scaling)


# -- the wire against a body ---------------------------------------------------

ORDERS = {"far_order": 2, "medium_order": 2, "near_order": 4}


def _small_case():
    medium = Medium(FIELD_STRENGTH)
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    elements = (_port(1), _port(2, "element", "capacitor", 30e-12, quality=500.0))
    coil = WireCoil.loop(0.05, 16, elements, at=(0, 8))
    return medium, body, coil


def test_the_wire_expansion_blocks_sit_on_its_nodes():
    medium, body, coil = _small_case()
    grid = pfft.extended_domain(body, coil)
    near = pfft.near_lists(grid, coil)
    torch.testing.assert_close(
        grid.centres(near.centre), coil.centre, atol=grid.resolution / 2 * 1.001, rtol=0
    )
    del medium


@pytest.fixture(scope="module")
def wire_solution():
    medium, body, coil = _small_case()
    return solve(body, coil, medium, tol=1e-8, **ORDERS)


def test_a_wire_coil_solves_to_its_tolerance(wire_solution):
    assert all(residual <= 1e-8 for residual in wire_solution.ports.residual)


def test_a_wire_coil_s_body_takes_what_it_absorbs_plus_what_it_scatters(
    wire_solution,
):
    from mariepy import fields

    operator = wire_solution.operator
    taken, absorbed, scattered = fields.power_balance(
        operator, wire_solution.fields, wire_solution.ports.body
    )
    torch.testing.assert_close(taken, absorbed + scattered)
    assert bool((absorbed > 0).all())


@pytest.mark.slow
def test_the_corrected_wire_coupling_is_the_direct_integral_where_it_corrects():
    """The precorrection subtracts exactly what the grid carries, as for a surface."""
    from tests.test_pfft import _through_the_grid

    medium, body, coil = _small_case()
    grid = pfft.extended_domain(body, coil)
    near = pfft.near_lists(grid, coil)
    _, _, cube_n, cube_k = pfft.kernels(grid, near, medium, medium_order=2)
    weights = pfft.projection(grid, coil, medium, near)
    response = pfft.expansion_response(grid, near, medium, cube_n, cube_k)
    direct = pfft.direct_coupling(grid, coil, medium, near)
    projected = pfft.projected_coupling(grid, coil, medium, near, weights, response)
    corrected = (direct[0] - projected[0]).to_dense() + _through_the_grid(
        grid, coil, medium, near, weights, cube_n
    )

    n_voxels = int(grid.mask.sum())
    expected = torch.zeros((3 * n_voxels, coil.n_dof), dtype=torch.complex128)
    numbering = pfft.body_numbering(grid)
    for dof, cells in pfft.near_body_pairs(grid, near, chunk=1 << 30):
        points = grid.centres(pfft.unflatten(grid, cells))
        value = body.resolution**3 * wire.coupling_n(
            coil, dof, points, medium, cell_size=grid.resolution
        )
        for component in range(3):
            expected[component * n_voxels + numbering[cells], dof] = value[:, component]
    covered = expected != 0
    assert int(covered.sum()) > 0
    error = (corrected[covered] - expected[covered]).abs().max()
    assert float(error / expected[covered].abs().max()) <= 1e-9


def test_a_two_port_wire_coil_is_reciprocal_before_it_is_symmetrised():
    medium, body, _ = _small_case()
    coil = WireCoil.loop(0.05, 16, (_port(1), _port(2)), at=(0, 8))
    result = solve(body, coil, medium, tol=1e-10, **ORDERS)
    admittance = network.port_admittance(
        result.operator.system.excitation, result.ports.coil
    )
    asymmetry = (admittance - admittance.T).abs().max()
    assert float(asymmetry) <= 1e-6 * float(admittance.abs().max())


# -- a wire coil and a surface coil together -------------------------------------


def _pair(separation, *, n_around=32):
    from mariepy.coil import SurfaceCoil
    from mariepy.mesh import SurfaceMesh

    surface = SurfaceCoil.build(
        SurfaceMesh.loop(
            radius=LOOP_RADIUS, width=0.004, n_around=n_around, n_across=1
        ),
        (_port(1),),
    )
    loop = WireCoil.loop(
        LOOP_RADIUS, n_around, (_port(1),), centre=(0.0, 0.0, separation)
    )
    return wire.CombinedCoil(wire=loop, surface=surface)


def test_a_wire_loop_and_a_surface_loop_share_neumann_s_mutual_inductance():
    """Two coaxial loops well below resonance: ``M`` from the elliptic integrals.

    The wire port drives its loop anticlockwise and the surface port drives
    its loop clockwise, so the mutual impedance is ``-j omega M``. At 21 MHz
    rather than lower: the surface EFIE loses its inductance below a few MHz.
    """
    from scipy.special import ellipe, ellipk

    medium = Medium(0.5)
    separation = 0.03
    coil = _pair(separation)
    system = wire.assemble_combined(coil, medium)
    current = torch.linalg.solve(system.impedance, system.excitation.T).T
    impedance = torch.linalg.inv(network.port_admittance(system.excitation, current))
    mutual = -float(impedance[0, 1].imag) / medium.angular_frequency

    m = 4 * LOOP_RADIUS**2 / (4 * LOOP_RADIUS**2 + separation**2)
    k = math.sqrt(m)
    neumann = (
        VACUUM_PERMEABILITY
        * LOOP_RADIUS
        * ((2 / k - k) * ellipk(m) - (2 / k) * ellipe(m))
    )
    assert mutual == pytest.approx(neumann, rel=0.02)
    torch.testing.assert_close(impedance[0, 1], impedance[1, 0])


def test_a_wire_and_a_surface_coil_solve_together_against_a_body():
    from mariepy import fields

    medium, body, _ = _small_case()
    coil = _pair(0.02, n_around=12)
    result = solve(body, coil, medium, tol=1e-9, **ORDERS)
    assert all(residual <= 1e-9 for residual in result.ports.residual)
    admittance = network.port_admittance(
        result.operator.system.excitation, result.ports.coil
    )
    asymmetry = (admittance - admittance.T).abs().max()
    assert float(asymmetry) <= 1e-5 * float(admittance.abs().max())
    taken, absorbed, scattered = fields.power_balance(
        result.operator, result.fields, result.ports.body
    )
    torch.testing.assert_close(taken, absorbed + scattered)


# -- a solved coil, tuned and matched --------------------------------------------


def test_a_solved_wire_coil_is_tuned_matched_and_its_fields_calibrated(tmp_path):
    """PLAN.md's milestone 3 criterion on a coil the solver produced.

    The coil is solved with its tuning capacitor opened into a port, as MARIE
    does with ``TMD`` set; co-simulation closes it and matches the port. With
    lossless elements the structure takes all the port accepts, and the body
    no more than that.
    """
    import dataclasses
    import json

    from mariepy import cosim, fields
    from mariepy.coil import read_lumped_elements

    elements = {
        "coil_configuration": {
            "elements": [
                {
                    "number": 1,
                    "type": "port",
                    "load": "inductorSeries_capacitorParallel",
                    "value": [1e-8, 1e-10],
                    "Q": [1e15, 1e15],
                    "optim": {
                        "boolean": 1,
                        "minim": [1e-10, 1e-12],
                        "maxim": [5e-8, 5e-10],
                        "symmetry": 1,
                    },
                    "cross_talk": {},
                    "excitation": {"entity": 1, "TxRx": "Tx"},
                },
                {
                    "number": 2,
                    "type": "element",
                    "load": "capacitor",
                    "value": 5e-12,
                    "Q": 1e15,
                    "optim": {
                        "boolean": 1,
                        "minim": 1e-12,
                        "maxim": 3e-11,
                        "symmetry": 2,
                    },
                    "cross_talk": {},
                    "excitation": {"entity": 1, "TxRx": "Tx"},
                },
            ]
        }
    }
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(elements))

    medium, body, _ = _small_case()
    coil = WireCoil.loop(0.05, 16, read_lumped_elements(path, tmd=True), at=(0, 8))
    result = solve(body, coil, medium, tol=1e-9, **ORDERS)
    small = cosim.Search(population=60, iterations=200, restarts=3)
    closed = cosim.co_simulate(
        cosim.read_network(path, tmd=True),
        result.admittance,
        medium.angular_frequency,
        tuning=small,
        matching=small,
        decoupling=small,
    )
    assert closed.costs["tuning"][1] < 1
    assert closed.costs["matching"][1] < 1
    assert float(closed.scattering.abs().max()) < 0.05

    voltage = closed.transmit[:, 0]
    taken = 0.5 * float((voltage.conj() @ result.admittance.cpu() @ voltage).real)
    accepted = 0.5 * (1 - float(closed.scattering[0, 0].abs() ** 2))
    assert taken == pytest.approx(accepted, rel=1e-9)

    calibrated = dataclasses.replace(
        result.fields,
        **{
            name: cosim.calibrate(getattr(result.fields, name), closed.transmit)
            for name in ("electric", "magnetic", "incident", "scattered")
        },
    )
    absorbed = float(fields.absorbed_power(result.operator, calibrated)[0])
    assert 0 < absorbed <= taken
