"""The coupled coil-and-body operator, its preconditioner and the port solve."""

import pytest
import torch

from mariepy import network, pfft, sie, vie
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import BodyOperator, solve_ports
from mariepy.system import CoupledOperator

FIELD_STRENGTH = 3.0
RESOLUTION = 0.01
BODY_RADIUS = 0.02
COIL_RADIUS = 0.05
PERMITTIVITY = 52.0
CONDUCTIVITY = 0.55
TOLERANCE = 1e-8

# The quadrature order of the body kernel's second pass. MARIE's 8 is what a
# solve for an answer wants; 4 is what a check of the assembly's structure
# needs, and it is twenty times cheaper.
MEDIUM_ORDER = 4

# What the precorrected FFT reproduces of the coil's own port impedance on the
# grid these tests build, with the body made all but transparent.
COUPLING_ERROR = 0.02

_BUILT: dict = {}


def _medium():
    return Medium(FIELD_STRENGTH)


def _coil(device, ports=1):
    mesh = SurfaceMesh.loop(
        radius=COIL_RADIUS,
        width=0.01,
        n_around=12,
        n_across=1,
        ports=ports,
        device=device,
    )
    elements = tuple(
        Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
        for tag in range(1, ports + 1)
    )
    return SurfaceCoil.build(mesh, elements)


def _operator(device, *, permittivity=PERMITTIVITY, conductivity=CONDUCTIVITY, ports=1):
    key = (device, permittivity, conductivity, ports)
    if key not in _BUILT:
        medium = _medium()
        body = VoxelBody.sphere(
            BODY_RADIUS,
            RESOLUTION,
            permittivity,
            conductivity,
            padding=1,
            device=device,
        )
        coil = _coil(device, ports=ports)
        system = sie.assemble(coil, medium)
        coupling = pfft.assemble(
            body, coil, system.impedance, medium, medium_order=MEDIUM_ORDER
        )
        _BUILT[key] = CoupledOperator(
            body=body, coil=coil, medium=medium, system=system, coupling=coupling
        )
    return _BUILT[key]


def _random(size, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(size, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(size, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary)


def _port_impedance(operator, solution):
    admittance = network.port_admittance(operator.system.excitation, solution.coil)
    return network.y_to_z(network.symmetrise(admittance))


def test_the_coil_rows_of_the_coupled_operator_reproduce_the_coil_matrix(device):
    operator = _operator(device)
    vector = torch.zeros(
        operator.n_coil + operator.n_body, dtype=torch.complex128, device=device
    )
    vector[: operator.n_coil] = _random(operator.n_coil, seed=1).to(device)
    got = operator(vector)[: operator.n_coil]
    want = operator.system.impedance @ vector[: operator.n_coil]
    assert float((got - want).abs().max() / want.abs().max()) <= 1e-4


def test_the_coupled_operator_is_the_block_matrix_its_coupling_assembles(device):
    """The product equals the operator written out from the coupling's matrices."""
    operator = _operator(device)
    coupling = operator.coupling
    n_components = coupling.n_components
    grid = coupling.grid
    vector = _random(operator.n_coil + operator.n_body, seed=2).to(device)
    coil, body = vector[: operator.n_coil], vector[operator.n_coil :]

    def field_of(on_grid):
        field = on_grid.reshape(n_components, *grid.shape)
        return (
            vie.apply_n(coupling.symbols_n, field) - vie.apply_g(field, grid.resolution)
        ).reshape(-1)

    scaling = operator.medium.electric_scaling
    on_body = coupling.scatter @ body
    applied = field_of(coupling.project @ coil + on_body) / scaling
    contrast = operator.body.contrast(operator.medium).scattering
    inverse = torch.zeros(grid.shape, dtype=torch.complex128, device=device)
    start = grid.body_origin
    inverse[
        start[0] : start[0] + operator.body.shape[0],
        start[1] : start[1] + operator.body.shape[1],
        start[2] : start[2] + operator.body.shape[2],
    ] = torch.where(operator.body.mask, 1.0 / contrast, torch.zeros_like(contrast))
    induced = (
        vie.apply_g(
            inverse * on_body.reshape(n_components, *grid.shape), grid.resolution
        ).reshape(-1)
        / scaling
    )
    scatter_t = coupling.scatter.transpose(0, 1)
    want = torch.cat(
        [
            coupling.project.transpose(0, 1) @ applied
            + coupling.coil @ coil
            + coupling.electric.transpose(0, 1) @ body,
            scatter_t @ (induced - applied) - coupling.electric @ coil,
        ]
    )
    torch.testing.assert_close(operator(vector), want, rtol=1e-12, atol=0.0)


def test_the_preconditioned_body_rows_are_the_body_operator_with_no_coil_current(
    device,
):
    operator = _operator(device)
    vector = torch.zeros(
        operator.n_coil + operator.n_body, dtype=torch.complex128, device=device
    )
    current = _random(operator.n_body, seed=2).to(device)
    vector[operator.n_coil :] = current

    preconditioned = operator.preconditioner()(operator(vector))[operator.n_coil :]
    body = BodyOperator.build(operator.body, operator.medium, medium_order=MEDIUM_ORDER)
    torch.testing.assert_close(preconditioned, body(current), rtol=1e-9, atol=0.0)


def test_the_preconditioner_inverts_the_coil_block_exactly(device):
    operator = _operator(device)
    vector = torch.zeros(
        operator.n_coil + operator.n_body, dtype=torch.complex128, device=device
    )
    vector[: operator.n_coil] = _random(operator.n_coil, seed=3).to(device)
    applied = operator.preconditioner()(
        torch.cat(
            [
                operator.system.impedance @ vector[: operator.n_coil],
                torch.zeros(operator.n_body, dtype=torch.complex128, device=device),
            ]
        )
    )
    torch.testing.assert_close(applied[: operator.n_coil], vector[: operator.n_coil])


def test_the_drive_of_a_port_reaches_the_coil_rows_and_nothing_else(device):
    operator = _operator(device)
    drives = operator.right_hand_side()
    assert drives.shape == (operator.coil.n_driven, operator.n_coil + operator.n_body)
    torch.testing.assert_close(drives[:, : operator.n_coil], operator.system.excitation)
    assert not bool(drives[:, operator.n_coil :].any())


def test_every_port_solve_reaches_the_tolerance_it_was_given(device):
    operator = _operator(device)
    solution = solve_ports(operator, tol=TOLERANCE)
    assert all(residual <= TOLERANCE for residual in solution.residual)
    assert solution.coil.shape == (operator.coil.n_driven, operator.n_coil)
    assert solution.body.shape == (operator.coil.n_driven, operator.n_body)


def test_a_body_of_free_space_leaves_the_port_impedance_where_the_empty_coil_had_it(
    device,
):
    operator = _operator(device, permittivity=1.000001, conductivity=0.0)
    solution = solve_ports(operator, tol=TOLERANCE)
    coupled = _port_impedance(operator, solution)[0, 0]

    system = operator.system
    alone = torch.linalg.solve(system.impedance, system.excitation.transpose(0, 1))
    empty = network.y_to_z(
        network.port_admittance(system.excitation, alone.transpose(0, 1))
    )[0, 0]
    assert abs(complex(coupled - empty)) / abs(complex(empty)) <= COUPLING_ERROR


def test_a_lossy_body_raises_the_port_resistance_and_barely_moves_its_reactance(device):
    loaded = _operator(device)
    empty = _operator(device, permittivity=1.000001, conductivity=0.0)
    with_body = _port_impedance(loaded, solve_ports(loaded, tol=TOLERANCE))[0, 0]
    without = _port_impedance(empty, solve_ports(empty, tol=TOLERANCE))[0, 0]

    assert float(with_body.real) > float(without.real)
    assert float(with_body.imag) == pytest.approx(float(without.imag), rel=0.01)


def test_the_port_matrix_is_reciprocal_before_it_is_symmetrised(device):
    operator = _operator(device, ports=2)
    solution = solve_ports(operator, tol=TOLERANCE)
    admittance = network.port_admittance(operator.system.excitation, solution.coil)
    defect = (admittance - admittance.transpose(0, 1)).abs().max()
    assert float(defect / admittance.abs().max()) <= 1e-5
