"""The coil-implicit solve reproduces the coupled one, from one build per grid."""

import dataclasses

import pytest
import torch

from mariepy import implicit, network, pfft
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import assemble_coil, solve_ports
from mariepy.system import CoupledOperator

RESOLUTION = 0.01
ORDERS = {"far_order": 2, "medium_order": 2, "near_order": 4}

_BUILT: dict = {}


def _medium():
    return Medium(3.0)


def _coil(device, ports=2, n_around=12, n_across=1):
    mesh = SurfaceMesh.loop(
        radius=0.05,
        width=0.01,
        n_around=n_around,
        n_across=n_across,
        ports=ports,
        device=device,
    )
    elements = tuple(
        Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
        for tag in range(1, ports + 1)
    )
    return SurfaceCoil.build(mesh, elements)


def _body(device, radius=0.02):
    return VoxelBody.sphere(radius, RESOLUTION, 52.0, 0.55, padding=2, device=device)


def _coil_block(operator):
    """The coupled operator's coil block, column by column."""
    n = operator.n_coil
    columns = []
    for index in range(n):
        vector = torch.zeros(
            n + operator.n_body, dtype=torch.complex128, device=operator.body.device
        )
        vector[index] = 1.0
        columns.append(operator(vector)[:n])
    return torch.stack(columns, dim=1)


def _built(device, *, exact):
    key = (device, exact)
    if key not in _BUILT:
        body = _body(device)
        # The compressed build needs a coil fine enough for its coupling to have
        # a rank below its own size; the exact one is checked on a coarse coil.
        coil = _coil(device) if exact else _coil(device, n_around=32, n_across=2)
        medium = _medium()
        system = assemble_coil(coil, medium)
        impedance = None
        region = torch.ones_like(body.mask) if exact else None
        if exact:
            whole_grid = dataclasses.replace(body, mask=region)
            box = pfft.assemble(whole_grid, coil, system.impedance, medium, **ORDERS)
            whole = CoupledOperator(
                body=whole_grid,
                coil=coil,
                medium=medium,
                system=system,
                coupling=box,
            )
            impedance = _coil_block(whole)
        perturbation = implicit.CoilPerturbation.build(
            body,
            coil,
            medium,
            tol=1e-12 if exact else 1e-3,
            region=region,
            impedance=impedance,
            system=system,
            **ORDERS,
        )
        _BUILT[key] = (body, perturbation)
    return _BUILT[key]


def _coupled(body, perturbation):
    box = perturbation.operator
    return CoupledOperator(
        body=body,
        coil=box.coil,
        medium=box.medium,
        system=box.system,
        coupling=pfft.restrict(box.coupling, body.mask),
    )


def _random(size, device, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(size, generator=generator, dtype=torch.float64),
        torch.randn(size, generator=generator, dtype=torch.float64),
    ).to(device)


def test_the_coupling_and_its_transpose_are_transposes(device):
    body, perturbation = _built(device, exact=False)
    operator = _coupled(body, perturbation)
    coil = _random(operator.n_coil, device, 1)
    body_current = _random(operator.n_body, device, 2)
    forward = body_current @ operator.couple(coil)
    backward = operator.couple_transpose(body_current) @ coil
    assert abs(complex(forward - backward)) <= 1e-10 * abs(complex(forward))


def test_the_block_products_rebuild_the_coupled_operator(device):
    body, perturbation = _built(device, exact=False)
    operator = _coupled(body, perturbation)
    coil = _random(operator.n_coil, device, 3)
    body_current = _random(operator.n_body, device, 4)
    whole = operator(torch.cat([coil, body_current]))
    rows = operator.body_block(body_current) - operator.couple(coil)
    torch.testing.assert_close(whole[operator.n_coil :], rows, rtol=1e-10, atol=0.0)


def test_eliminating_the_coil_block_gives_the_coupled_solve(device):
    body, perturbation = _built(device, exact=True)
    implicit_result = perturbation.solve(body, tol=1e-11, precision="double")
    coupled = solve_ports(_coupled(body, perturbation), tol=1e-11)

    scale = coupled.body.abs().max()
    assert (
        float((implicit_result.ports.body - coupled.body).abs().max() / scale) <= 1e-7
    )
    want = network.symmetrise(
        network.port_admittance(perturbation.operator.system.excitation, coupled.coil)
    )
    torch.testing.assert_close(implicit_result.admittance, want, rtol=1e-7, atol=0.0)


def test_the_compressed_perturbation_is_the_exact_one_within_its_tolerance(device):
    body, perturbation = _built(device, exact=False)
    prepared = perturbation.prepare(body)
    operator = prepared.operator
    vector = _random(operator.n_body, device, 5)

    back = operator.couple_transpose(vector)
    exact = operator.couple(
        torch.linalg.lu_solve(*perturbation.factors, back[:, None])[:, 0]
    )
    compressed = prepared.left @ (prepared.right.transpose(0, 1) @ vector)
    error = torch.linalg.vector_norm(compressed - exact)
    assert float(error / torch.linalg.vector_norm(exact)) <= 1e-2
    assert perturbation.rank < operator.n_coil


def test_a_body_smaller_than_the_grid_reuses_the_build(device):
    body, perturbation = _built(device, exact=True)
    coordinates = body.coordinates()
    inside = (coordinates**2).sum(dim=0) <= 0.012**2
    smaller = dataclasses.replace(body, mask=body.mask & inside)
    assert 0 < smaller.n_voxels < body.n_voxels
    result = perturbation.solve(smaller, tol=1e-11, precision="double")
    coupled = solve_ports(_coupled(smaller, perturbation), tol=1e-11)
    scale = coupled.body.abs().max()
    assert float((result.ports.body - coupled.body).abs().max() / scale) <= 1e-7


def test_a_mixed_precision_solve_gives_the_double_precision_currents(device):
    body, perturbation = _built(device, exact=False)
    double = perturbation.solve(body, tol=1e-10, precision="double")
    mixed = perturbation.solve(body, tol=1e-10, precision="mixed")
    assert mixed.ports.body.dtype == torch.complex128
    assert all(residual <= 1e-10 for residual in mixed.ports.residual)
    # Two solutions within 1e-10 of the right-hand side differ by that times the
    # condition number of the coil-implicit system, which is large here.
    scale = double.ports.body.abs().max()
    assert float((mixed.ports.body - double.ports.body).abs().max() / scale) <= 1e-5


def test_a_body_off_the_grid_is_refused(device):
    body, perturbation = _built(device, exact=False)
    moved = dataclasses.replace(
        body, origin=tuple(o + RESOLUTION / 2 for o in body.origin)
    )
    with pytest.raises(ValueError, match="grid the perturbation was built on"):
        perturbation.prepare(moved)


def test_the_default_region_keeps_its_clearance_from_the_conductors(device):
    # A grid wide enough for the loop to cross it.
    body = _body(device, radius=0.045)
    coil = _coil(device, n_around=32, n_across=2)
    region = implicit.tissue_region(body, coil, clearance=0.015)
    assert not bool(region.all())
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[region.reshape(-1)]
    points = implicit._conductor_points(coil, 0.001)
    gap = (centres[:, None, :] - points[None, :, :]).abs().amax(dim=-1).min()
    assert float(gap) > 0.015 - RESOLUTION / 2
    assert bool(region.any())
