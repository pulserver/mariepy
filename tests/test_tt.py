"""Tensor trains and the cross approximation that builds them."""

import numpy as np
import pytest
import torch

from mariepy import tt


def _smooth(sizes):
    """A smooth complex tensor of low tensor-train rank."""
    axes = [torch.linspace(0.1, 1.0, n, dtype=torch.float64) for n in sizes]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return (
        1.0 / (1.0 + grid.sum(-1)) * torch.exp(-3j * grid[..., 0] * grid[..., -1])
    ).to(torch.complex128)


def _random_train(sizes, ranks, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return tt.TensorTrain(
        cores=tuple(
            torch.randn(
                ranks[k],
                sizes[k],
                ranks[k + 1],
                dtype=torch.complex128,
                generator=generator,
            )
            for k in range(len(sizes))
        )
    )


def test_the_matlab_reshape_runs_the_first_index_fastest():
    values = np.arange(24.0).reshape(2, 3, 4, order="F")
    reshaped = tt._reshape(torch.from_numpy(values), (4, 6))
    np.testing.assert_array_equal(reshaped.numpy(), values.reshape(4, 6, order="F"))


def test_a_column_major_index_comes_back_as_its_subscripts():
    sizes = (3, 4, 5)
    for index in range(60):
        expected = np.unravel_index(index, sizes, order="F")
        assert tt._ind2sub(sizes, index) == tuple(int(value) for value in expected)


def test_the_maximum_volume_rows_leave_no_interpolation_coefficient_above_one():
    generator = torch.Generator().manual_seed(1)
    matrix, _ = torch.linalg.qr(
        torch.randn(40, 6, dtype=torch.complex128, generator=generator)
    )
    rows = tt._maxvol(matrix)
    assert len(set(rows)) == 6
    interpolation = torch.linalg.solve(
        matrix[rows].transpose(0, 1), matrix.transpose(0, 1)
    ).transpose(0, 1)
    assert float(interpolation.abs().max()) <= 1.0 + 5e-2


def test_the_rank_chop_keeps_the_discarded_tail_below_the_tolerance():
    values = torch.tensor([1.0, 0.5, 1e-3, 1e-4, 1e-5], dtype=torch.float64)
    kept = tt._chop(values, 2e-3)
    assert kept == 2
    assert float(torch.linalg.vector_norm(values[kept:])) < 2e-3


def test_reorthogonalising_appends_directions_orthogonal_to_the_basis():
    generator = torch.Generator().manual_seed(2)
    basis, _ = torch.linalg.qr(
        torch.randn(12, 3, dtype=torch.complex128, generator=generator)
    )
    extra = torch.randn(12, 4, dtype=torch.complex128, generator=generator)
    grown = tt._reorthogonalise(basis, extra)
    assert grown.shape == (12, 7)
    torch.testing.assert_close(grown.mH @ grown, torch.eye(7, dtype=torch.complex128))
    torch.testing.assert_close(grown[:, :3], basis)


@pytest.mark.parametrize("sizes", [(6, 5, 4, 7), (3, 1, 5, 9)])
def test_the_cross_approximation_reaches_its_tolerance(sizes):
    full = _smooth(sizes)
    train = tt.cross(sizes, lambda s: full[s[:, 0], s[:, 1], s[:, 2], s[:, 3]], 1e-8)
    assert train.shape == sizes
    error = (train.full() - full).abs().max() / full.abs().max()
    assert float(error) <= 1e-7


def _smooth_at(sizes, subscripts):
    """Entries of :func:`_smooth` at some subscripts, without the whole tensor."""
    grid = torch.stack(
        [
            0.1 + 0.9 * subscripts[:, axis].to(torch.float64) / (n - 1)
            for axis, n in enumerate(sizes)
        ],
        dim=-1,
    )
    return (
        1.0 / (1.0 + grid.sum(-1)) * torch.exp(-3j * grid[..., 0] * grid[..., -1])
    ).to(torch.complex128)


def test_the_cross_approximation_samples_a_large_tensor_sparingly():
    """The sweeps cost what the ranks and sizes set, not what the tensor holds.

    The tensor is large enough for that cost to be a small share of it, and
    too large to hold whole, so the train is checked at random entries.
    """
    sizes = (60, 60, 60, 1000)
    total = 60 * 60 * 60 * 1000
    sampled = []

    def entries(subscripts):
        sampled.append(subscripts.shape[0])
        return _smooth_at(sizes, subscripts)

    generator = torch.Generator().manual_seed(0)
    train = tt.cross(sizes, entries, 1e-6, generator=generator)
    assert sum(sampled) < 0.1 * total

    probe = torch.stack(
        [torch.randint(0, n, (20000,), generator=generator) for n in sizes], dim=1
    )
    cores = train.cores
    vector = cores[0][0, probe[:, 0]]  # (probes, r1)
    for axis in range(1, len(sizes)):
        vector = torch.einsum("pa,apb->pb", vector, cores[axis][:, probe[:, axis]])
    expected = _smooth_at(sizes, probe)
    error = torch.linalg.vector_norm(
        vector[:, 0] - expected
    ) / torch.linalg.vector_norm(expected)
    assert float(error) <= 1e-5


def test_applying_a_train_contracts_its_last_index():
    train = _random_train((4, 3, 5, 6), (1, 2, 3, 2, 1))
    vector = torch.randn(6, dtype=torch.complex128)
    expected = torch.einsum("ijkl,l->ijk", train.full(), vector)
    torch.testing.assert_close(train.apply(vector), expected)


def test_the_transposed_application_contracts_the_other_indices():
    train = _random_train((4, 3, 5, 6), (1, 2, 3, 2, 1))
    field = torch.randn(4, 3, 5, dtype=torch.complex128)
    expected = torch.einsum("ijkl,ijk->l", train.full(), field)
    torch.testing.assert_close(train.apply_transpose(field), expected)


def test_the_two_applications_are_transposes_of_each_other():
    train = _random_train((4, 3, 5, 6), (1, 2, 3, 2, 1), seed=3)
    vector = torch.randn(6, dtype=torch.complex128)
    field = torch.randn(4, 3, 5, dtype=torch.complex128)
    left = (field * train.apply(vector)).sum()
    right = (train.apply_transpose(field) * vector).sum()
    torch.testing.assert_close(left, right)
