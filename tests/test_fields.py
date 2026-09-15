"""The field the ports drive in the body, and the power it carries."""

import pytest
import torch

from mariepy import coupling, fields, pfft, sie, vie
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import solve_ports
from mariepy.system import CoupledOperator
from mariepy.tucker import circulant_tucker

FIELD_STRENGTH = 3.0
RESOLUTION = 0.01
BODY_RADIUS = 0.02
COIL_RADIUS = 0.05
PERMITTIVITY = 52.0
CONDUCTIVITY = 0.55
TOLERANCE = 1e-10
MEDIUM_ORDER = 4

# What the precorrected FFT and the two Tucker compressions leave between the
# field taken through the grid and the same field integrated pair by pair, on
# the grid these tests build.
FIELD_ERROR = 1e-4

_SOLVED: dict = {}


def _solved(device, ports=1):
    if (device, ports) not in _SOLVED:
        medium = Medium(FIELD_STRENGTH)
        body = VoxelBody.sphere(
            BODY_RADIUS,
            RESOLUTION,
            PERMITTIVITY,
            CONDUCTIVITY,
            padding=1,
            device=device,
        )
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
        coil = SurfaceCoil.build(mesh, elements)
        system = sie.assemble(coil, medium)
        operator = CoupledOperator(
            body=body,
            coil=coil,
            medium=medium,
            system=system,
            coupling=pfft.assemble(
                body, coil, system.impedance, medium, medium_order=MEDIUM_ORDER
            ),
        )
        solution = solve_ports(operator, tol=TOLERANCE)
        _SOLVED[(device, ports)] = (
            operator,
            solution,
            fields.compute(operator, solution.coil, solution.body),
        )
    return _SOLVED[(device, ports)]


def _by_direct_integration(operator, solution):
    """Evaluate the same field without the projection: one integral per pair."""
    body, medium = operator.body, operator.medium
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    corners = operator.coil.rwg_vertices()
    n_cells, n_dof = centres.shape[0], operator.coil.n_dof

    block = coupling.coupling_n(
        corners.repeat_interleave(n_cells, dim=0),
        centres.repeat(n_dof, 1),
        medium,
        cell_size=body.resolution,
    ).reshape(n_dof, n_cells, 3)
    from_coil = torch.einsum("pn,ncq->pqc", solution.coil, block)

    symbols = circulant_tucker(
        vie.kernel_n(
            body.shape, body.resolution, medium.wavenumber, medium_order=MEDIUM_ORDER
        ).to(body.device),
        1e-7,
    )
    current = body.from_dof(solution.body)
    applied = vie.apply_n(symbols, current) - vie.apply_g(current, body.resolution)
    from_body = applied / medium.electric_scaling / body.resolution**3

    spread = torch.zeros_like(from_body)
    spread.reshape(spread.shape[0], 3, -1)[..., body.mask.reshape(-1)] = (
        from_coil.reshape(from_coil.shape[0], 3, -1)
    )
    return spread + body.mask * from_body


def test_the_field_in_the_body_is_the_one_a_direct_integration_gives(device):
    operator, solution, computed = _solved(device)
    expected = _by_direct_integration(operator, solution)
    error = (computed.electric - expected).abs().max()
    assert float(error / expected.abs().max()) <= FIELD_ERROR


def test_the_field_is_zero_outside_the_body(device):
    _, _, computed = _solved(device)
    operator, _, _ = _solved(device)
    outside = ~operator.body.mask
    assert not bool(computed.electric[..., outside].any())
    assert not bool(computed.magnetic[..., outside].any())


def test_the_total_field_is_the_coil_s_part_plus_the_body_s_own(device):
    _, _, computed = _solved(device)
    torch.testing.assert_close(
        computed.electric, computed.incident + computed.scattered
    )


def test_the_power_the_body_takes_from_the_coil_is_what_it_absorbs_plus_what_it_scatters(
    device,
):
    operator, solution, computed = _solved(device)
    taken, absorbed, scattered = fields.power_balance(operator, computed, solution.body)
    torch.testing.assert_close(taken, absorbed + scattered)
    assert bool((absorbed > 0).all())
    assert bool((scattered > 0).all())


def test_the_ohmic_loss_integrated_from_the_field_is_the_power_the_current_takes(
    device,
):
    operator, solution, computed = _solved(device)
    _, absorbed, _ = fields.power_balance(operator, computed, solution.body)
    torch.testing.assert_close(fields.absorbed_power(operator, computed), absorbed)


def test_the_port_delivers_more_power_than_the_body_absorbs(device):
    operator, solution, computed = _solved(device)
    delivered = fields.delivered_power(operator, solution.coil)
    absorbed = fields.absorbed_power(operator, computed)
    assert bool((delivered > absorbed).all())
    assert bool((absorbed > 0).all())


def test_the_circular_components_add_back_to_the_transverse_magnetic_field(device):
    operator, _, computed = _solved(device)
    plus, minus = fields.circular_components(operator, computed)
    permeability = operator.medium.permeability
    torch.testing.assert_close(
        0.5 * (plus + minus), permeability * computed.magnetic[:, 0]
    )
    torch.testing.assert_close(
        -0.5j * (plus - minus), permeability * computed.magnetic[:, 1].to(plus.dtype)
    )


def test_two_ports_driven_together_carry_the_power_their_admittance_says(device):
    operator, solution, _ = _solved(device, ports=2)
    delivered = fields.delivered_power(operator, solution.coil)
    from mariepy import network

    admittance = network.port_admittance(operator.system.excitation, solution.coil)
    expected = 0.5 * torch.real(torch.diagonal(admittance))
    torch.testing.assert_close(delivered, expected)
    assert delivered.shape == (2,)


@pytest.mark.parametrize("ports", [1, 2])
def test_every_port_drives_a_field_of_its_own(device, ports):
    _, _, computed = _solved(device, ports=ports)
    assert computed.electric.shape[0] == ports
    assert bool((computed.electric.abs().amax(dim=(1, 2, 3, 4)) > 0).all())


def test_one_call_drives_the_coil_against_the_body_and_returns_its_ports_and_fields(
    device,
):
    from mariepy.solver import solve

    medium = Medium(FIELD_STRENGTH)
    body = VoxelBody.sphere(
        BODY_RADIUS, RESOLUTION, PERMITTIVITY, CONDUCTIVITY, padding=1, device=device
    )
    coil = SurfaceCoil.build(
        SurfaceMesh.loop(
            radius=COIL_RADIUS, width=0.01, n_around=12, n_across=1, device=device
        ),
        (Port(tag=1, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0),),
    )
    result = solve(body, coil, medium, medium_order=MEDIUM_ORDER)

    assert result.impedance.shape == (1, 1)
    assert float(result.impedance[0, 0].real) > 0.0
    assert float(result.impedance[0, 0].imag) > 0.0
    assert float(result.scattering.abs()[0, 0]) <= 1.0
    torch.testing.assert_close(result.admittance, result.admittance.transpose(0, 1))
    assert bool((fields.absorbed_power(result.operator, result.fields) > 0).all())
    assert all(residual <= 1e-5 for residual in result.ports.residual)
