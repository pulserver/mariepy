"""The coil and a piecewise-linear body, solved together."""

import torch

from mariepy import coupling, fields, network, pfft, sie, vie
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import solve_ports
from mariepy.system import CoupledOperator
from mariepy.tucker import circulant_tucker

TOLERANCE = 1e-8
ORDERS = {"far_order": 2, "medium_order": 2, "near_order": 4}

# What the precorrected FFT and the two Tucker compressions leave between the
# field taken through the grid and the same field integrated pair by pair, on
# the grid these tests build.
FIELD_ERROR = 1e-4

_SOLVED: dict = {}


def _solved(device):
    if device not in _SOLVED:
        medium = Medium(3.0)
        # Small enough that the ten-pair kernels assemble in seconds.
        body = VoxelBody.sphere(0.01, 0.01, 52.0, 0.55, padding=1, device=device)
        mesh = SurfaceMesh.loop(
            radius=0.03, width=0.01, n_around=8, n_across=1, ports=2, device=device
        )
        elements = tuple(
            Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
            for tag in (1, 2)
        )
        coil = SurfaceCoil.build(mesh, elements)
        system = sie.assemble(coil, medium)
        operator = CoupledOperator(
            body=body,
            coil=coil,
            medium=medium,
            system=system,
            coupling=pfft.assemble(
                body, coil, system.impedance, medium, linear=True, **ORDERS
            ),
        )
        solution = solve_ports(operator, tol=TOLERANCE)
        _SOLVED[device] = (
            operator,
            solution,
            fields.compute(operator, solution.coil, solution.body),
        )
    return _SOLVED[device]


def test_the_body_carries_twelve_unknowns_per_voxel(device):
    operator, solution, computed = _solved(device)
    assert operator.n_body == 12 * operator.body.n_voxels
    assert solution.body.shape == (2, operator.n_body)
    assert computed.electric.shape == (2, 12, *operator.body.shape)


def test_every_port_solve_reaches_the_tolerance_it_was_given(device):
    _, solution, _ = _solved(device)
    assert all(residual <= TOLERANCE for residual in solution.residual)


def test_the_port_matrix_is_reciprocal_before_it_is_symmetrised(device):
    operator, solution, _ = _solved(device)
    admittance = network.port_admittance(operator.system.excitation, solution.coil)
    asymmetry = (admittance - admittance.T).abs().max()
    assert asymmetry <= 1e-6 * admittance.abs().max()


def test_the_power_the_body_takes_is_what_it_absorbs_plus_what_it_scatters(device):
    operator, solution, computed = _solved(device)
    taken, absorbed, scattered = fields.power_balance(operator, computed, solution.body)
    torch.testing.assert_close(taken, absorbed + scattered)
    assert bool((absorbed > 0).all())


def test_the_ohmic_loss_weighs_each_linear_coefficient_by_its_mass(device):
    """Integrating sigma |E|^2 over a voxel is the mass-weighted sum of squares."""
    operator, solution, computed = _solved(device)
    _, absorbed, _ = fields.power_balance(operator, computed, solution.body)
    torch.testing.assert_close(fields.absorbed_power(operator, computed), absorbed)


def test_the_port_delivers_more_power_than_the_body_absorbs(device):
    operator, solution, computed = _solved(device)
    delivered = fields.delivered_power(operator, solution.coil)
    assert bool((delivered > fields.absorbed_power(operator, computed)).all())


def _by_direct_integration(operator, solution):
    """Evaluate the same field without the projection: one integral per pair."""
    body, medium = operator.body, operator.medium
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    corners = operator.coil.rwg_vertices()
    n_cells, n_dof = centres.shape[0], operator.coil.n_dof
    mass = vie.mass(12, 1.0)[:4].to(body.device)

    per_term = []
    for term in range(4):
        average = coupling.coupling_n(
            corners.repeat_interleave(n_cells, dim=0),
            centres.repeat(n_dof, 1),
            medium,
            cell_size=body.resolution,
            basis_term=term,
        ).reshape(n_dof, n_cells, 3)
        per_term.append(average / mass[term])
    block = torch.stack(per_term, dim=-1).reshape(n_dof, n_cells, 12)
    from_coil = torch.einsum("pn,ncq->pqc", solution.coil, block)

    symbols = circulant_tucker(
        vie.kernel_n(
            body.shape, body.resolution, medium.wavenumber, linear=True, **ORDERS
        ).to(body.device),
        1e-7,
    )
    current = body.from_dof(solution.body)
    applied = vie.apply_n(symbols, current) - vie.apply_g(current, body.resolution)
    from_body = vie.apply_inverse_g(applied / medium.electric_scaling, body.resolution)

    spread = torch.zeros_like(from_body)
    spread.reshape(spread.shape[0], 12, -1)[..., body.mask.reshape(-1)] = (
        from_coil.reshape(from_coil.shape[0], 12, -1)
    )
    return spread + body.mask * from_body


def test_the_field_in_the_body_is_the_one_a_direct_integration_gives(device):
    operator, solution, computed = _solved(device)
    expected = _by_direct_integration(operator, solution)
    error = (computed.electric - expected).abs().max()
    assert float(error / expected.abs().max()) <= FIELD_ERROR


def test_the_field_at_the_voxel_centres_is_the_constant_coefficient(device):
    _, _, computed = _solved(device)
    centres = fields.at_centres(computed.electric)
    assert centres.shape[-4] == 3
    torch.testing.assert_close(centres, computed.electric[:, [0, 4, 8]])


def test_the_circular_components_are_taken_at_the_voxel_centres(device):
    operator, _, computed = _solved(device)
    plus, minus = fields.circular_components(operator, computed)
    centres = fields.at_centres(computed.magnetic)
    permeability = operator.medium.permeability
    torch.testing.assert_close(
        plus + minus, 2 * permeability * centres[:, 0], rtol=1e-12, atol=0.0
    )
    assert plus.shape == (2, *operator.body.shape)
