"""A closed shield around the coil and the body, solved with them."""

import pytest
import torch

from mariepy import coupling, fields, network, pfft, sie
from mariepy import shield as shield_module
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import solve_ports
from mariepy.system import CoupledOperator, ShieldedOperator

TOLERANCE = 1e-8
TT_TOLERANCE = 1e-6
ORDERS = {"far_order": 2, "medium_order": 2, "near_order": 4}

_BUILT: dict = {}


def _medium():
    return Medium(3.0)


def _body(device=None):
    return VoxelBody.sphere(0.01, 0.01, 52.0, 0.55, padding=1, device=device)


def _coil(device=None):
    mesh = SurfaceMesh.loop(
        radius=0.03, width=0.01, n_around=8, n_across=1, ports=2, device=device
    )
    elements = tuple(
        Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
        for tag in (1, 2)
    )
    return SurfaceCoil.build(mesh, elements)


def _surface(device=None):
    """A closed sphere around the coil and the body, with no port of its own."""
    return SurfaceCoil.build(
        SurfaceMesh.sphere(radius=0.06, subdivisions=1, device=device)
    )


def _operator(device, linear=False):
    key = (device, linear)
    if key not in _BUILT:
        medium, body, coil, surface = (
            _medium(),
            _body(device),
            _coil(device),
            _surface(device),
        )
        system = sie.assemble(coil, medium)
        coupled = CoupledOperator(
            body=body,
            coil=coil,
            medium=medium,
            system=system,
            coupling=pfft.assemble(
                body, coil, system.impedance, medium, linear=linear, **ORDERS
            ),
        )
        operator = ShieldedOperator(
            coupled=coupled,
            shield=shield_module.assemble(
                surface, coil, body, medium, tol=TT_TOLERANCE, linear=linear
            ),
        )
        solution = solve_ports(operator, tol=TOLERANCE)
        _BUILT[key] = (
            operator,
            solution,
            fields.compute(operator, solution.coil, solution.body, solution.shield),
        )
    return _BUILT[key]


@pytest.mark.parametrize("magnetic", [False, True])
@pytest.mark.parametrize("linear", [False, True])
def test_the_tensor_train_coupling_is_the_fully_assembled_one(magnetic, linear):
    """PLAN.md's milestone 2 criterion, on a grid small enough to assemble."""
    medium, body, surface = _medium(), _body(), _surface()
    trains = shield_module.body_coupling(
        body, surface, medium, magnetic=magnetic, tol=TT_TOLERANCE, linear=linear
    )
    kernel = coupling.coupling_k if magnetic else coupling.coupling_n
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)
    corners = surface.rwg_vertices()
    n_cells, n_dof = centres.shape[0], surface.n_dof
    terms = range(4) if linear else range(1)
    full = (
        torch.stack(
            [
                kernel(
                    corners.repeat(n_cells, 1, 1),
                    centres.repeat_interleave(n_dof, dim=0),
                    medium,
                    cell_size=body.resolution,
                    basis_term=term,
                )
                for term in terms
            ],
            dim=-1,
        ).reshape(*body.shape, n_dof, 3 * len(terms))
        * body.resolution**3
    )
    for index, train in enumerate(trains):
        expected = full[..., index]
        error = (train.full() - expected).abs().max() / expected.abs().max()
        assert float(error) <= 10 * TT_TOLERANCE, f"unknown {index}: {float(error):.1e}"


def test_the_coil_to_shield_block_is_the_transpose_of_the_shield_to_coil_one():
    """Two independent integrations of the same interaction, before any symmetry."""
    medium, coil, surface = _medium(), _coil(), _surface()
    forward = sie.coupling_matrix(surface, coil, medium)
    backward = sie.coupling_matrix(coil, surface, medium)
    torch.testing.assert_close(forward, backward.transpose(0, 1))


def test_without_shield_current_the_coil_and_body_rows_are_the_unshielded_ones(device):
    operator, _, _ = _operator(device)
    generator = torch.Generator().manual_seed(4)
    rest = torch.randn(
        operator.n_coil + operator.n_body, dtype=torch.complex128, generator=generator
    ).to(device)
    vector = torch.cat(
        [torch.zeros(operator.n_shield, dtype=torch.complex128, device=device), rest]
    )
    torch.testing.assert_close(
        operator(vector)[operator.n_shield :], operator.coupled(rest)
    )


def test_the_preconditioner_inverts_the_shield_and_coil_block_exactly(device):
    operator, _, _ = _operator(device)
    surfaces = operator.n_shield + operator.n_coil
    generator = torch.Generator().manual_seed(5)
    head = torch.randn(surfaces, dtype=torch.complex128, generator=generator).to(device)
    shield_current, coil_current = head[: operator.n_shield], head[operator.n_shield :]
    joint = torch.cat(
        [
            operator.shield.system.impedance @ shield_current
            + operator.shield.coil_coupling @ coil_current,
            operator.shield.coil_coupling.transpose(0, 1) @ shield_current
            + operator.system.impedance @ coil_current,
        ]
    )
    applied = operator.preconditioner()(
        torch.cat(
            [joint, torch.zeros(operator.n_body, dtype=torch.complex128, device=device)]
        )
    )
    torch.testing.assert_close(applied[:surfaces], head)


def test_a_shield_without_ports_takes_no_drive(device):
    operator, _, _ = _operator(device)
    drive = operator.right_hand_side()
    assert drive.shape == (2, operator.n_shield + operator.n_coil + operator.n_body)
    assert not bool(drive[:, : operator.n_shield].any())


def test_every_shielded_port_solve_reaches_the_tolerance_it_was_given(device):
    _, solution, _ = _operator(device)
    assert all(residual <= TOLERANCE for residual in solution.residual)
    assert solution.shield is not None


def test_the_shielded_port_matrix_is_reciprocal_before_it_is_symmetrised(device):
    operator, solution, _ = _operator(device)
    admittance = network.port_admittance(operator.system.excitation, solution.coil)
    asymmetry = (admittance - admittance.T).abs().max()
    assert asymmetry <= 10 * TT_TOLERANCE * admittance.abs().max()


def test_the_shielded_body_takes_what_it_absorbs_plus_what_it_scatters(device):
    operator, solution, computed = _operator(device)
    taken, absorbed, scattered = fields.power_balance(operator, computed, solution.body)
    torch.testing.assert_close(taken, absorbed + scattered)
    assert bool((absorbed > 0).all())
    torch.testing.assert_close(fields.absorbed_power(operator, computed), absorbed)


def test_the_shield_changes_the_field_the_coil_drives(device):
    """A closed conductor around the coil cannot leave the body's field alone."""
    operator, _, computed = _operator(device)
    unshielded = solve_ports(operator.coupled, tol=TOLERANCE)
    bare = fields.compute(operator.coupled, unshielded.coil, unshielded.body)
    change = (computed.electric - bare.electric).abs().max()
    assert float(change) > 1e-3 * float(bare.electric.abs().max())


def test_a_shielded_linear_body_is_reciprocal_and_balanced():
    operator, solution, computed = _operator("cpu", linear=True)
    admittance = network.port_admittance(operator.system.excitation, solution.coil)
    assert (admittance - admittance.T).abs().max() <= 10 * TT_TOLERANCE * (
        admittance.abs().max()
    )
    taken, absorbed, scattered = fields.power_balance(operator, computed, solution.body)
    torch.testing.assert_close(taken, absorbed + scattered)


# -- a shield with a port of its own -------------------------------------------------


def _loop_mesh(radius, n_around, ports, first_tag):
    mesh = SurfaceMesh.loop(
        radius=radius, width=0.01, n_around=n_around, n_across=1, ports=ports
    )
    return mesh, mesh.line_tags + (first_tag - 1) * (mesh.line_tags > 0)


def _driven_pair():
    """An outer loop with one port as the shield, an inner two-port loop as the coil."""
    outer, outer_tags = _loop_mesh(0.07, 24, 1, 1)
    inner, inner_tags = _loop_mesh(0.03, 12, 2, 2)
    port = [
        Port(tag=t, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
        for t in (1, 2, 3)
    ]
    from dataclasses import replace

    shield = SurfaceCoil.build(outer, (port[0],))
    coil = SurfaceCoil.build(
        replace(inner, line_tags=inner_tags - 1),
        (replace(port[1], tag=1), replace(port[2], tag=2)),
    )
    merged = SurfaceMesh(
        nodes=torch.cat([outer.nodes, inner.nodes]),
        triangles=torch.cat([outer.triangles, inner.triangles + outer.n_nodes]),
        triangle_tags=torch.cat([outer.triangle_tags, inner.triangle_tags]),
        lines=torch.cat([outer.lines, inner.lines + outer.n_nodes]),
        line_tags=torch.cat([outer_tags, inner_tags]),
    )
    both = SurfaceCoil.build(merged, tuple(port))
    return shield, coil, both


def test_a_driven_shield_s_port_comes_first_and_the_ports_are_reciprocal():
    from mariepy.solver import solve

    medium, body = _medium(), _body()
    shield, coil, both = _driven_pair()
    shielded = solve(body, coil, medium, tol=TOLERANCE, shield=shield, **ORDERS)
    admittance = network.port_admittance(
        shielded.operator.excitation,
        shielded.operator.conductors(shielded.ports.coil, shielded.ports.shield),
    )
    assert admittance.shape == (3, 3)
    asymmetry = (admittance - admittance.T).abs().max()
    assert float(asymmetry) <= 1e-4 * float(admittance.abs().max())

    # The same two loops as one coil, coupled through the precorrected FFT.
    whole = solve(body, both, medium, tol=TOLERANCE, **ORDERS)
    error = (shielded.admittance - whole.admittance).abs().max()
    assert float(error / whole.admittance.abs().max()) <= 1e-2

    taken, absorbed, scattered = fields.power_balance(
        shielded.operator, shielded.fields, shielded.ports.body
    )
    torch.testing.assert_close(taken, absorbed + scattered)
