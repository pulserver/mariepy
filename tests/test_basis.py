"""A body's field basis, and a coil solved through it against the direct solve."""

import numpy as np
import pytest
import torch

from mariepy import basis as basis_module
from mariepy import fields, metrics, network, sie
from mariepy import shield as shield_module
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import BodyOperator, solve_body

ORDERS = {"far_order": 2, "medium_order": 2, "near_order": 4}
TOLERANCE = 1e-9

# A driven shield's own port is the one configuration here whose reduced model
# does not reach the solver tolerance. The other two reduced-against-dense
# checks agree to 4e-12; this one agrees to between 2e-9 and 2e-7, and which end
# of that it lands on is decided by the interpolation points DEIM picks. Those
# come from singular vectors that a near-degenerate spectrum leaves free to
# rotate, so a different BLAS picks differently: perturbing the coupling by
# 1e-14, which is nothing physical, moves the picked count between 30 and 33 and
# the agreement over two orders of magnitude, with 8 of 24 draws above 1e-7.
# The basis is rank-saturated -- tightening its tolerance from 1e-12 to 1e-14
# keeps no further vectors -- so this is the accuracy the rank allows, not a
# tolerance that can be tightened. The bound is on the spread, not on one draw
# of it.
DRIVEN_SHIELD_AGREEMENT = 1e-6


def _case():
    medium = Medium(3.0)
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    port = Port(tag=1, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
    coil = SurfaceCoil.build(
        SurfaceMesh.loop(radius=0.05, width=0.01, n_around=8, n_across=1, ports=2),
        (
            port,
            Port(tag=2, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0),
        ),
    )
    return medium, body, coil


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(False, id="constant"),
        pytest.param(True, id="linear", marks=pytest.mark.slow),
    ],
)
def solved(request):
    linear = request.param
    medium, body, coil = _case()
    incident = basis_module.surface_basis(
        body, coil, medium, tol=1e-12, interpolation_tol=1e-12, linear=linear
    )
    field_basis = basis_module.solve(incident, body, medium, tol=TOLERANCE, **ORDERS)
    return medium, body, coil, field_basis, linear


def test_the_interpolation_recovers_each_basis_field_from_its_samples(solved):
    _, body, _, field_basis, _ = solved
    sampled = basis_module._sampled(
        field_basis.incident_electric,
        field_basis.samples,
        body.n_voxels,
        field_basis.n_components,
    )
    torch.testing.assert_close(
        field_basis.interpolation @ sampled,
        torch.eye(field_basis.rank, dtype=torch.complex128),
        atol=1e-8,
        rtol=0,
    )


def _dense(coil, system, body, medium, linear):
    """Couple the coil to the body whole: one body solve per coil basis function."""
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    tested = basis_module.coupling_at(
        coil, centres, medium, body.resolution, linear=linear
    )
    return _dense_from(
        tested, system.impedance, system.excitation, body, medium, linear
    )


def _dense_from(tested, matrix, drive, body, medium, linear):
    """Add the body's whole response to a matrix, and give the port admittance."""
    inverse = basis_module._inverse_mass(
        12 if linear else 3, body.n_voxels, body.resolution, tested.device
    )
    operator = BodyOperator.build(body, medium, linear=linear, **ORDERS)
    currents = torch.stack(
        [
            solve_body(operator, body.from_dof(inverse * column), tol=TOLERANCE).x
            for column in tested.T
        ]
    )
    impedance = matrix + tested.T @ currents.T
    current = torch.linalg.solve(impedance, drive.T).T
    return network.symmetrise(network.port_admittance(drive, current))


def test_a_coil_solved_through_the_basis_is_the_coil_coupled_whole(solved):
    """PLAN.md's milestone 4 criterion, against the coil and body coupled densely.

    The support is the coil, so the basis spans its fields and the two agree to
    the solver tolerance. The precorrected FFT is not the reference here: on a
    coil whose triangles are several voxels long, its three-cell expansion
    block alone is off by a few per cent.
    """
    medium, body, coil, field_basis, linear = solved
    system = sie.assemble(coil, medium)
    reduced = basis_module.solve_coil(coil, system, field_basis, body, medium)
    dense = _dense(coil, system, body, medium, linear)
    error = (reduced.admittance - dense).abs().max()
    assert float(error / dense.abs().max()) <= 1e-7

    free = torch.linalg.solve(system.impedance, system.excitation.T).T
    bare = network.symmetrise(network.port_admittance(system.excitation, free))
    assert float((dense - bare).abs().max()) > 1e3 * float(error)


def test_a_shielded_coil_through_the_basis_is_the_shielded_coil_coupled_whole():
    """The shield is coupled to the body as the coil is, and takes no drive.

    The basis is built on one support holding both surfaces, so it spans both
    their fields.
    """
    medium, body, coil = _case()
    shield = SurfaceCoil.build(SurfaceMesh.sphere(radius=0.09, subdivisions=0))
    shield_mesh, coil_mesh = shield.mesh, coil.mesh
    support = SurfaceCoil.build(
        SurfaceMesh(
            nodes=torch.cat([shield_mesh.nodes, coil_mesh.nodes]),
            triangles=torch.cat(
                [shield_mesh.triangles, coil_mesh.triangles + shield_mesh.n_nodes]
            ),
            triangle_tags=torch.cat(
                [shield_mesh.triangle_tags, coil_mesh.triangle_tags]
            ),
            lines=torch.zeros((0, 2), dtype=torch.long),
            line_tags=torch.zeros(0, dtype=torch.long),
        )
    )
    incident = basis_module.surface_basis(
        body, support, medium, tol=1e-12, interpolation_tol=1e-12
    )
    joint = basis_module.solve(incident, body, medium, tol=TOLERANCE, **ORDERS)
    system = sie.assemble(coil, medium)
    reduced = basis_module.solve_coil(coil, system, joint, body, medium, shield=shield)

    own = sie.assemble(shield, medium).impedance
    cross = shield_module._coil_coupling(shield, coil, medium, 4)
    whole = torch.cat(
        [torch.cat([own, cross], dim=1), torch.cat([cross.T, system.impedance], dim=1)]
    )
    drive = torch.nn.functional.pad(system.excitation, (shield.n_dof, 0))
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    tested = torch.cat(
        [
            basis_module.coupling_at(part, centres, medium, body.resolution)
            for part in (shield, coil)
        ],
        dim=1,
    )
    dense = _dense_from(tested, whole, drive, body, medium, False)
    error = (reduced.admittance - dense).abs().max()
    assert float(error / dense.abs().max()) <= 1e-7
    assert reduced.shield.shape == (2, shield.n_dof)


def test_a_driven_shield_s_port_leads_the_reduced_ports():
    """A shield with a port: its drive is its own, and its port comes first."""
    from tests.test_shield import _driven_pair

    medium, body, _ = _case()
    shield, coil, _ = _driven_pair()
    support = SurfaceCoil.build(
        SurfaceMesh(
            nodes=torch.cat([shield.mesh.nodes, coil.mesh.nodes]),
            triangles=torch.cat(
                [shield.mesh.triangles, coil.mesh.triangles + shield.mesh.n_nodes]
            ),
            triangle_tags=torch.cat(
                [shield.mesh.triangle_tags, coil.mesh.triangle_tags]
            ),
            lines=torch.zeros((0, 2), dtype=torch.long),
            line_tags=torch.zeros(0, dtype=torch.long),
        )
    )
    incident = basis_module.surface_basis(
        body, support, medium, tol=1e-12, interpolation_tol=1e-12
    )
    joint = basis_module.solve(incident, body, medium, tol=TOLERANCE, **ORDERS)
    system = sie.assemble(coil, medium)
    reduced = basis_module.solve_coil(coil, system, joint, body, medium, shield=shield)

    own = sie.assemble(shield, medium)
    cross = shield_module._coil_coupling(shield, coil, medium, 4)
    whole = torch.cat(
        [
            torch.cat([own.impedance, cross], dim=1),
            torch.cat([cross.T, system.impedance], dim=1),
        ]
    )
    drive = torch.cat(
        [
            torch.nn.functional.pad(own.excitation, (0, coil.n_dof)),
            torch.nn.functional.pad(system.excitation, (shield.n_dof, 0)),
        ]
    )
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    tested = torch.cat(
        [
            basis_module.coupling_at(part, centres, medium, body.resolution)
            for part in (shield, coil)
        ],
        dim=1,
    )
    dense = _dense_from(tested, whole, drive, body, medium, False)
    assert reduced.admittance.shape == (3, 3)
    error = (reduced.admittance - dense).abs().max()
    assert float(error / dense.abs().max()) <= DRIVEN_SHIELD_AGREEMENT


def test_the_basis_fields_are_the_body_s_own_total_fields(solved):
    """Each basis field against the body solved directly for its incident field."""
    medium, body, _, field_basis, linear = solved
    operator = BodyOperator.build(body, medium, linear=linear, **ORDERS)
    incident = body.from_dof(field_basis.incident_electric[0])
    current = solve_body(operator, incident, tol=TOLERANCE).x
    total = body.to_dof(operator.total_field(current, incident))
    torch.testing.assert_close(field_basis.electric[0], total, rtol=1e-7, atol=1e-12)


def test_no_coil_in_the_basis_beats_the_ultimate_snr(solved):
    medium, body, coil, field_basis, _ = solved
    reduced = basis_module.solve_coil(
        coil, sie.assemble(coil, medium), field_basis, body, medium
    )
    covariance = metrics.noise_covariance(
        reduced.electric, body.conductivity, body.mask, body.resolution
    )
    centres = fields.at_centres(reduced.magnetic)
    b1_minus = medium.permeability * (centres[:, 0] - 1j * centres[:, 1])
    coil_snr = metrics.snr(b1_minus, covariance, medium, body.resolution, body.mask)
    ultimate, efficiency = basis_module.ultimate_maps(field_basis, body, medium)
    assert bool((coil_snr <= ultimate * (1 + 1e-6) + 1e-12).all())
    assert bool((efficiency[body.mask] > 0).all())


def test_a_saved_basis_reads_back_whole(solved, tmp_path):
    _, _, _, field_basis, _ = solved
    field_basis.save(tmp_path / "basis.pt")
    loaded = basis_module.FieldBasis.load(tmp_path / "basis.pt")
    for name in ("electric", "response", "covariance", "samples", "interpolation"):
        torch.testing.assert_close(getattr(loaded, name), getattr(field_basis, name))
    assert loaded.linear == field_basis.linear


# -- the dipole shell ---------------------------------------------------------------


def test_the_shell_wraps_the_body_at_its_distance_and_thickness():
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    around, padding = basis_module.shell(
        body.mask, body.resolution, distance=0.02, thickness=2
    )
    assert padding == 4
    inside = torch.nn.functional.pad(body.mask, (padding,) * 6)
    hull = basis_module._convex_hull_slices(inside)
    assert bool((hull >= inside).all())
    assert not bool((around & hull).any())
    body_points = inside.nonzero().to(torch.float64)
    hull_points = hull.nonzero().to(torch.float64)
    shell_points = around.nonzero().to(torch.float64)
    gaps = torch.cdist(shell_points, hull_points).min(dim=1).values
    assert float(gaps.min()) > 2.0
    assert float(gaps.max()) <= 2 + 2 + 1e-9
    # The shell closes around the body: every straight line out crosses it.
    centre = body_points.mean(dim=0).round().long()
    for axis in range(3):
        for direction in (1, -1):
            index = centre.clone()
            crossed = False
            while 0 <= int(index[axis]) < around.shape[axis]:
                crossed |= bool(around[tuple(index.tolist())])
                index[axis] += direction
            assert crossed


@pytest.fixture(scope="module")
def dipoles():
    """A body with enough unknowns that the kept rank stays well below them.

    On a body of a few dozen voxels the basis nearly exhausts its unknowns,
    and the electric fields of the kept currents turn linearly dependent.
    """
    medium = Medium(3.0)
    body = VoxelBody.sphere(0.03, 0.01, 52.0, 0.55, padding=1)
    incident = basis_module.dipole_basis(
        body,
        medium,
        distance=0.01,
        thickness=2,
        block=400,
        **ORDERS,
    )
    return (
        medium,
        body,
        basis_module.solve(incident, body, medium, tol=TOLERANCE, **ORDERS),
    )


def test_the_dipole_basis_interpolates_its_own_fields(dipoles):
    _, body, field_basis = dipoles
    sampled = basis_module._sampled(
        field_basis.incident_electric, field_basis.samples, body.n_voxels, 3
    )
    torch.testing.assert_close(
        field_basis.interpolation @ sampled,
        torch.eye(field_basis.rank, dtype=torch.complex128),
        atol=1e-6,
        rtol=0,
    )
    assert field_basis.rank < 3 * body.n_voxels


def test_a_loop_outside_the_shell_does_not_beat_the_ultimate_snr(dipoles):
    """Any source outside the shell has the field of some current in it."""
    medium, body, field_basis = dipoles
    port = Port(tag=1, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
    coil = SurfaceCoil.build(
        SurfaceMesh.loop(radius=0.08, width=0.01, n_around=16, n_across=1), (port,)
    )
    direct = _dense_fields(coil, body, medium)
    covariance = metrics.noise_covariance(
        direct[0], body.conductivity, body.mask, body.resolution
    )
    centres = fields.at_centres(direct[1])
    b1_minus = medium.permeability * (centres[:, 0] - 1j * centres[:, 1])
    coil_snr = metrics.snr(b1_minus, covariance, medium, body.resolution, body.mask)
    ultimate, _ = basis_module.ultimate_maps(field_basis, body, medium)
    inside = body.mask
    assert bool((coil_snr[inside] <= ultimate[inside] * 1.05).all())


def test_more_basis_fields_never_lower_the_ultimate_snr(dipoles):
    """The optimum over a larger set of currents is at least the smaller one's.

    On a grid this coarse the ultimate SNR does not settle as fields are
    added, which is why MARIE maps it over a run of mode counts.
    """
    medium, body, field_basis = dipoles
    previous = torch.zeros(body.shape, dtype=torch.float64)
    for modes in (5, 20, 80, field_basis.rank):
        snr, _ = basis_module.ultimate_maps(field_basis, body, medium, modes=modes)
        assert bool((snr[body.mask] >= previous[body.mask] * (1 - 1e-6)).all())
        previous = snr


def _dense_fields(coil, body, medium):
    """A coil's total fields in a body, coupled whole, for a unit port drive."""
    system = sie.assemble(coil, medium)
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    tested = basis_module.coupling_at(coil, centres, medium, body.resolution)
    tested_k = basis_module.coupling_at(
        coil, centres, medium, body.resolution, magnetic=True
    )
    inverse = basis_module._inverse_mass(3, body.n_voxels, body.resolution, "cpu")
    operator = BodyOperator.build(body, medium, **ORDERS)
    currents = torch.stack(
        [
            solve_body(operator, body.from_dof(inverse * column), tol=TOLERANCE).x
            for column in tested.T
        ]
    )
    impedance = system.impedance + tested.T @ currents.T
    coil_current = torch.linalg.solve(impedance, system.excitation.T).T
    incident = coil_current @ (inverse[:, None] * tested).T
    body_current = coil_current @ currents
    electric = torch.stack(
        [
            operator.total_field(j, body.from_dof(e))
            for j, e in zip(body_current, incident, strict=True)
        ]
    )
    kernel_k = basis_module.circulant_tucker(
        basis_module.vie.kernel_k(
            body.shape, body.resolution, medium.wavenumber, **ORDERS
        ),
        1e-7,
    )
    scattered = torch.stack(
        [
            body.mask
            * basis_module.vie.apply_inverse_g(
                basis_module.vie.apply_k(kernel_k, body.from_dof(j)), body.resolution
            )
            for j in body_current
        ]
    )
    magnetic = body.from_dof(coil_current @ (inverse[:, None] * tested_k).T) + scattered
    return electric, magnetic


def test_a_basis_saved_as_marie_saves_it_solves_a_coil_as_ours_does(solved, tmp_path):
    """MARIE's file layout, its voxel order and its tested form, read back."""
    pytest.importorskip("h5py")
    from tests.marie_files import write_marie_basis

    medium, body, coil, field_basis, linear = solved
    if linear:
        pytest.skip("MARIE's tested form matches the field form for the constant basis")
    gram = body.resolution**3
    n = body.n_voxels
    index = body.mask.nonzero()
    n1, n2, _ = body.shape
    fortran = index[:, 0] + n1 * (index[:, 1] + n2 * index[:, 2])
    to_marie = torch.argsort(fortran)  # MARIE position -> our voxel
    order = (torch.arange(3)[:, None] * n + to_marie[None, :]).reshape(-1)

    left, values, right_h = torch.linalg.svd(field_basis.response / gram)
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    sampled = centres[field_basis.samples]
    write_marie_basis(
        tmp_path / "basis.mat",
        {
            "Ue": (gram * field_basis.electric[:, order]).T.numpy(),
            "Ub": (gram * field_basis.magnetic[:, order]).T.numpy(),
            "X": (field_basis.interpolation / gram).numpy(),
            "U_hat_inv": left.numpy(),
            "S_hat_inv": torch.diag(values).numpy(),
            "V_hat_inv": right_h.conj().T.resolve_conj().numpy(),
            "xds": sampled[:, 0:1].numpy(),
            "yds": sampled[:, 1:2].numpy(),
            "zds": sampled[:, 2:3].numpy(),
        },
    )
    read = basis_module.read_marie(tmp_path / "basis.mat", body)
    assert read.tested and read.rank == field_basis.rank
    system = sie.assemble(coil, medium)
    ours = basis_module.solve_coil(coil, system, field_basis, body, medium)
    theirs = basis_module.solve_coil(coil, system, read, body, medium)
    torch.testing.assert_close(theirs.admittance, ours.admittance, rtol=1e-9, atol=0)
    torch.testing.assert_close(theirs.electric, ours.electric, rtol=1e-8, atol=1e-12)
    assert theirs.body is None


def test_a_basis_for_another_body_is_refused(solved, tmp_path):
    pytest.importorskip("h5py")
    from tests.marie_files import write_marie_basis

    _, body, _, _, _ = solved
    write_marie_basis(
        tmp_path / "basis.mat",
        {
            "Ue": np.ones((7, 2), dtype=complex),
            "Ub": np.ones((7, 2), dtype=complex),
            "X": np.ones((2, 3), dtype=complex),
            "U_hat_inv": np.eye(3, dtype=complex),
            "S_hat_inv": np.eye(3),
            "V_hat_inv": np.eye(3, dtype=complex),
            "xds": np.zeros((1, 1)),
            "yds": np.zeros((1, 1)),
            "zds": np.zeros((1, 1)),
        },
    )
    with pytest.raises(ValueError, match="3 or 12"):
        basis_module.read_marie(tmp_path / "basis.mat", body)


def test_the_spherical_shell_lies_between_its_two_spheres():
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    around, padding = basis_module.shell(
        body.mask, body.resolution, distance=0.01, thickness=2, shape="sphere"
    )
    shape = torch.tensor(body.shape, dtype=torch.float64)
    centre = torch.ceil(shape / 2) - 1 + padding
    half = float(torch.linalg.vector_norm((shape - 1) * body.resolution / 2))
    radius = (around.nonzero().to(torch.float64) - centre).norm(dim=1) * body.resolution
    assert float(radius.min()) >= half + 0.01 - 1e-12
    assert float(radius.max()) <= half + 0.01 + 2 * body.resolution + 1e-12
    inside = torch.nn.functional.pad(body.mask, (padding,) * 6)
    assert not bool((around & inside).any())
    # Every direction out of the centre crosses it.
    for axis in range(3):
        line = around.movedim(axis, 0)[
            :, int(centre[(axis + 1) % 3]), int(centre[(axis + 2) % 3])
        ]
        assert bool(line[: int(centre[axis])].any()) and bool(
            line[int(centre[axis]) :].any()
        )


def test_an_unknown_shell_shape_is_refused():
    with pytest.raises(ValueError, match="cube"):
        basis_module.shell(
            torch.ones(2, 2, 2, dtype=torch.bool),
            0.01,
            distance=0.01,
            thickness=1,
            shape="cube",
        )


def test_a_loop_outside_the_spherical_shell_does_not_beat_its_ultimate_snr():
    medium = Medium(3.0)
    body = VoxelBody.sphere(0.03, 0.01, 52.0, 0.55, padding=1)
    incident = basis_module.dipole_basis(
        body, medium, distance=0.005, thickness=1, support="sphere", block=400, **ORDERS
    )
    field_basis = basis_module.solve(incident, body, medium, tol=TOLERANCE, **ORDERS)
    assert field_basis.rank < 3 * body.n_voxels
    port = Port(tag=1, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
    coil = SurfaceCoil.build(
        SurfaceMesh.loop(radius=0.09, width=0.01, n_around=16, n_across=1), (port,)
    )
    electric, magnetic = _dense_fields(coil, body, medium)
    covariance = metrics.noise_covariance(
        electric, body.conductivity, body.mask, body.resolution
    )
    centres = fields.at_centres(magnetic)
    b1_minus = medium.permeability * (centres[:, 0] - 1j * centres[:, 1])
    coil_snr = metrics.snr(b1_minus, covariance, medium, body.resolution, body.mask)
    ultimate, _ = basis_module.ultimate_maps(field_basis, body, medium)
    assert bool((coil_snr[body.mask] <= ultimate[body.mask] * 1.05).all())
