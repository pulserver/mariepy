"""Tucker compression and the circulant embedding reproduce the dense operator."""

import itertools

import pytest
import torch

from mariepy import tucker

PARITY_BY_COMPONENT = {
    6: (
        (+1, -1, -1, +1, +1, +1),
        (+1, -1, +1, +1, -1, +1),
        (+1, +1, -1, +1, -1, +1),
    ),
    3: (
        (-1, +1, +1),
        (+1, -1, +1),
        (+1, +1, -1),
    ),
}


def _random_tensor(shape, device, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(shape, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float64)
    return torch.complex(real, imaginary).to(device)


def _dense_toeplitz(kernel, signs):
    """Build the three-level Toeplitz operator one kernel component stands for."""
    shape = kernel.shape
    size = shape[0] * shape[1] * shape[2]
    dense = torch.zeros((size, size), device=kernel.device, dtype=kernel.dtype)
    cells = list(itertools.product(*(range(n) for n in shape)))
    for row, observer in enumerate(cells):
        for column, source in enumerate(cells):
            value = kernel[
                tuple(abs(a - b) for a, b in zip(observer, source, strict=True))
            ]
            for axis, (a, b) in enumerate(zip(observer, source, strict=True)):
                if a < b:
                    value = value * signs[axis]
            dense[row, column] = value
    return dense


def _apply_symbol(symbol, vector, shape):
    """Apply the compressed operator to one vector through the doubled grid."""
    padded = torch.zeros(symbol.shape, device=vector.device, dtype=vector.dtype)
    padded[: shape[0], : shape[1], : shape[2]] = vector.reshape(shape)
    product = torch.fft.fftn(padded) * symbol.expand()
    return torch.fft.ifftn(product)[: shape[0], : shape[1], : shape[2]].reshape(-1)


def test_mode_product_contracts_the_named_mode(device):
    tensor = _random_tensor((3, 4, 5), device)
    matrix = _random_tensor((4, 7), device, seed=1)
    got = tucker.mode_product(tensor, matrix, 1)
    assert got.shape == (3, 7, 5)
    assert torch.allclose(got, torch.einsum("ijk,jm->imk", tensor, matrix))


def test_hosvd_reconstructs_a_tensor_it_does_not_truncate(device):
    tensor = _random_tensor((4, 3, 5), device)
    core, *factors = tucker.hosvd(tensor)
    assert torch.allclose(tucker.to_full(core, *factors), tensor, atol=1e-12)


def test_hosvd_keeps_full_rank_when_given_no_tolerance(device):
    tensor = _random_tensor((4, 3, 5), device)
    core, *_ = tucker.hosvd(tensor)
    assert core.shape == (4, 3, 5)


def test_hosvd_keeps_one_vector_past_the_rank_it_needs(device):
    """MARIE's rule includes the first singular value at or below the threshold.

    A tensor of mode rank 2 therefore compresses to a 3x3x3 core, one vector
    wider than it has to be, which costs memory and never accuracy.
    """
    factors = [_random_tensor((n, 2), device, seed=n) for n in (6, 5, 4)]
    tensor = torch.einsum(
        "ia,jb,kc,abc->ijk", *factors, _random_tensor((2, 2, 2), device)
    )
    core, *reconstructed = tucker.hosvd(tensor, tol=1e-10)
    assert core.shape == (3, 3, 3)
    assert torch.allclose(tucker.to_full(core, *reconstructed), tensor, atol=1e-10)


def test_hosvd_reconstructs_within_the_tolerance_it_was_given(device):
    tensor = _random_tensor((8, 7, 6), device)
    tol = 1e-3
    core, *factors = tucker.hosvd(tensor, tol=tol)
    error = torch.linalg.vector_norm(tucker.to_full(core, *factors) - tensor)
    assert error.item() <= tol * torch.linalg.vector_norm(tensor).item()


def test_hosvd_rejects_a_tensor_that_is_not_third_order():
    with pytest.raises(ValueError, match="third-order tensor"):
        tucker.hosvd(torch.zeros(3, 4))


@pytest.mark.parametrize("n_components", [6, 3])
def test_circulant_embedding_reproduces_the_dense_toeplitz_product(
    n_components, device
):
    shape = (3, 4, 2)
    kernel = _random_tensor((*shape, n_components), device)
    symbols = tucker.circulant_tucker(kernel, tol=None)
    assert len(symbols) == n_components

    vector = _random_tensor((shape[0] * shape[1] * shape[2],), device, seed=9)
    for component, symbol in enumerate(symbols):
        signs = [
            PARITY_BY_COMPONENT[n_components][axis][component] for axis in range(3)
        ]
        dense = _dense_toeplitz(kernel[..., component], signs)
        assert torch.allclose(
            _apply_symbol(symbol, vector, shape), dense @ vector, atol=1e-11
        )


def test_circulant_embedding_doubles_every_axis(device):
    shape = (3, 4, 2)
    kernel = _random_tensor((*shape, 6), device)
    for symbol in tucker.circulant_tucker(kernel, tol=None):
        assert symbol.shape == tuple(2 * n for n in shape)


def test_circulant_tucker_rejects_a_component_count_it_has_no_parity_for(device):
    kernel = _random_tensor((2, 2, 2, 4), device)
    with pytest.raises(ValueError, match="no parity assignment"):
        tucker.circulant_tucker(kernel)


def test_circulant_tucker_rejects_a_kernel_that_is_not_fourth_order():
    with pytest.raises(ValueError, match="four axes"):
        tucker.circulant_tucker(torch.zeros(2, 2, 2))
