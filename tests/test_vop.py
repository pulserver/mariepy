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
