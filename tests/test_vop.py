"""Virtual observation points bound the local SAR of every drive."""

import numpy as np
import pytest
import torch

from mariepy import sar, vop

CHANNELS = 4
POINTS = 40
MARGIN = 0.05


def _matrices(device, points=POINTS, seed=0):
    """SAR matrices of random three-component fields, as a body gives them."""
    generator = torch.Generator().manual_seed(seed)
    field = torch.randn(
        CHANNELS, 3, points, dtype=torch.complex128, generator=generator
    )
    conductivity = torch.rand(points, dtype=torch.float64, generator=generator)
    matrices = torch.einsum(
        "iav,jav,v->vij", field.conj(), field, conductivity.to(torch.complex128)
    )
    return matrices.to(device), field, conductivity


def _drives(count, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, CHANNELS, dtype=torch.complex128, generator=generator)


def test_every_matrix_is_dominated_by_the_point_it_is_clustered_into(device):
    matrices, _, _ = _matrices(device)
    points, cluster = vop.compress(matrices, MARGIN)
    assert cluster.shape == (matrices.shape[0],)
    assert int(cluster.max()) == points.shape[0] - 1
    gaps = torch.linalg.eigvalsh(points[cluster] - matrices)[:, 0]
    scale = float(torch.linalg.eigvalsh(matrices)[:, -1].max())
    assert float(gaps.min()) >= -1e-12 * scale


def test_no_drive_sees_less_sar_through_the_points_than_through_the_matrices(device):
    matrices, _, _ = _matrices(device)
    points, _ = vop.compress(matrices, MARGIN)
    drives = _drives(16, 3).to(device)
    true = sar.peak(matrices, drives)
    bound = sar.peak(points, drives)
    assert bool((bound >= true * (1 - 1e-12)).all())


def test_the_overestimate_stays_inside_the_margin_it_was_given(device):
    """A cluster's matrix adds at most the margin's share of the worst eigenvalue."""
    matrices, _, _ = _matrices(device)
    points, _ = vop.compress(matrices, MARGIN)
    allowed = MARGIN * float(torch.linalg.eigvalsh(matrices)[:, -1].max())
    drives = _drives(16, 4).to(device)
    slack = allowed * torch.linalg.vector_norm(drives, dim=-1) ** 2
    assert bool((sar.peak(points, drives) <= sar.peak(matrices, drives) + slack).all())


def test_every_point_is_hermitian(device):
    matrices, _, _ = _matrices(device)
    points, _ = vop.compress(matrices, MARGIN)
    torch.testing.assert_close(points, points.conj().transpose(-2, -1))


def test_a_wider_margin_never_needs_more_points(device):
    matrices, _, _ = _matrices(device)
    counts = [vop.compress(matrices, margin)[0].shape[0] for margin in (0.0, 0.02, 0.2)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] >= 1


def test_a_zero_margin_still_terminates_and_still_bounds(device):
    matrices, _, _ = _matrices(device, points=12)
    points, _ = vop.compress(matrices, 0.0)
    assert points.shape[0] <= matrices.shape[0]
    assert float(vop.dominates(points, matrices).min()) >= -1e-9


def test_a_stack_that_is_not_square_is_refused():
    with pytest.raises(ValueError, match="n_channels"):
        vop.compress(torch.zeros(3, 2, 4, dtype=torch.complex128), 0.05)


def test_a_negative_margin_is_refused():
    with pytest.raises(ValueError, match="fraction"):
        vop.compress(torch.zeros(3, 2, 2, dtype=torch.complex128), -0.1)


def test_the_compression_is_cosimpy_s_own(device):
    """The reference implementation, with the conjugation the two conventions differ by."""
    cosimpy = pytest.importorskip("cosimpy")
    points_count = 24
    matrices, field, conductivity = _matrices(device, points=points_count, seed=2)
    em = cosimpy.EM_Field(
        [128e6],
        [2, 3, 4],
        e_field=field.numpy()[None],
        props={"idxs": np.arange(points_count), "sigma": conductivity.numpy()},
    )
    reference, _ = em.compVOP(128e6, MARGIN, z0_ports=1, elCond_key="sigma")
    points, _ = vop.compress(matrices, MARGIN)
    assert points.shape == reference.shape
    torch.testing.assert_close(
        points.cpu(), torch.from_numpy(np.conj(reference)), atol=1e-10, rtol=1e-8
    )


def _file_arguments():
    return {
        "coil": "loop-8ch-3T",
        "frequency_hz": 127.74e6,
        "drive_unit": "1 sqrt(W) incident at the channel input",
        "channels": [f"ch{index}" for index in range(CHANNELS)],
        "averaging": "10 g, IEC/IEEE 62704-1",
        "bodies": ["duke", "ella"],
        "compression_margin": MARGIN,
        "data_licence": "CC BY 4.0",
    }


def _written(tmp_path, device, **overrides):
    matrices, _, _ = _matrices(device)
    points, _ = vop.compress(matrices, MARGIN)
    whole = torch.stack([matrices.mean(dim=0), matrices[:5].mean(dim=0)])
    arguments = _file_arguments() | overrides
    path = tmp_path / "vops.npz"
    vop.write(path, points, whole, **arguments)
    return path, points, whole, arguments


def test_a_written_file_reads_back_whole(tmp_path, device):
    path, points, whole, arguments = _written(tmp_path, device)
    back = vop.read(path)
    torch.testing.assert_close(back.vops, points.cpu())
    torch.testing.assert_close(back.global_matrix, whole.cpu())
    for key, value in arguments.items():
        assert back.metadata[key] == value
    assert back.metadata["mariepy_version"]


def test_the_file_loads_without_unpickling(tmp_path, device):
    """The metadata is JSON, so the archive never needs pickle to open."""
    path, _, _, _ = _written(tmp_path, device)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"vops", "global_matrix", "metadata"}


def test_the_points_in_the_file_still_bound_the_drive(tmp_path, device):
    path, _, _, _ = _written(tmp_path, device)
    matrices, _, _ = _matrices(device)
    drives = _drives(8, 11).to(device)
    back = vop.read(path, device=matrices.device)
    assert bool((sar.peak(back.vops, drives) >= sar.peak(matrices, drives)).all())


def test_a_channel_count_the_names_do_not_match_is_refused(tmp_path, device):
    with pytest.raises(ValueError, match="channel names"):
        _written(tmp_path, device, channels=["only-one"])


def test_a_body_count_the_names_do_not_match_is_refused(tmp_path, device):
    with pytest.raises(ValueError, match="body names"):
        _written(tmp_path, device, bodies=["duke"])


def test_a_stack_that_is_not_hermitian_is_refused(tmp_path):
    matrices = torch.eye(CHANNELS, dtype=torch.complex128).expand(2, -1, -1).clone()
    matrices[0, 0, 1] = 1j
    with pytest.raises(ValueError, match="not Hermitian"):
        vop.write(tmp_path / "v.npz", matrices, matrices[:2], **_file_arguments())


def test_the_two_stacks_must_agree_on_the_channel_count(tmp_path):
    points = torch.eye(CHANNELS, dtype=torch.complex128)[None]
    whole = torch.eye(CHANNELS + 1, dtype=torch.complex128).expand(2, -1, -1)
    with pytest.raises(ValueError, match="head-average"):
        vop.write(tmp_path / "v.npz", points, whole, **_file_arguments())


def test_an_archive_without_the_contract_s_entries_is_refused(tmp_path):
    path = tmp_path / "bare.npz"
    np.savez(path, vops=np.zeros((1, 2, 2), dtype=np.complex128))
    with pytest.raises(ValueError, match="global_matrix, metadata"):
        vop.read(path)


def test_metadata_the_contract_requires_is_checked_on_reading(tmp_path):
    import json

    path = tmp_path / "thin.npz"
    np.savez(
        path,
        vops=np.zeros((1, 2, 2), dtype=np.complex128),
        global_matrix=np.zeros((1, 2, 2), dtype=np.complex128),
        metadata=np.array(json.dumps({"coil": "x"})),
    )
    with pytest.raises(ValueError, match="frequency_hz"):
        vop.read(path)


def test_a_body_carries_from_its_fields_to_a_file_that_still_bounds_it(tmp_path):
    """The whole chain: fields, matrices, averaging cubes, points, file."""
    from mariepy import averaging

    generator = torch.Generator().manual_seed(12)
    size = 13
    axis = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    tissue = torch.sqrt(x**2 + y**2 + z**2) <= size / 2 - 1.0
    resolution = 0.002
    density = torch.full(tissue.shape, 1000.0, dtype=torch.float64)
    mass = torch.where(tissue, density * resolution**3, 0.0)
    electric = torch.randn(
        CHANNELS, 3, size, size, size, dtype=torch.complex128, generator=generator
    )
    conductivity = 0.5 * tissue.to(torch.float64)

    local = sar.local_matrices(electric, conductivity, density, tissue)
    cubes = averaging.centred_cubes(mass, tissue, 1e-3)
    over_cubes = averaging.averaged_matrices(cubes, local, mass, tissue)
    points, _ = vop.compress(over_cubes, MARGIN)
    whole = sar.average(local, sar.voxel_mass(density, resolution, tissue))

    path = tmp_path / "chain.npz"
    vop.write(
        path,
        points,
        whole[None],
        **(_file_arguments() | {"bodies": ["ball"], "averaging": "1 g, step 1"}),
    )
    back = vop.read(path)
    drives = _drives(8, 13)
    assert bool((sar.peak(back.vops, drives) >= sar.peak(over_cubes, drives)).all())
    assert bool(
        (sar.peak(over_cubes, drives) >= sar.sar(whole[None], drives)[:, 0]).all()
    )
