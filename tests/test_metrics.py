"""SNR, transmit efficiency and g-factor maps against their definitions."""

import math

import pytest
import torch

from mariepy import metrics
from mariepy.constants import Medium


def _random(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(*shape, generator=generator, dtype=torch.float64),
        torch.randn(*shape, generator=generator, dtype=torch.float64),
    )


def _covariance(n, seed):
    base = _random((n, n), seed)
    return base @ base.conj().T + n * torch.eye(n, dtype=torch.complex128)


def test_water_protons_at_three_tesla_have_the_textbook_magnetisation():
    """``N (gamma hbar)^2 B0 / (4 k T)`` for spin one half, worked by hand."""
    gamma = 2 * math.pi * 42.577478518e6
    hbar = 6.626e-34 / (2 * math.pi)
    by_hand = 6.691e28 * (gamma * hbar) ** 2 * 3.0 / (4 * 1.3806503e-23 * 310.0)
    got = metrics.equilibrium_magnetisation(Medium(3.0))
    assert got == pytest.approx(by_hand, rel=1e-12)
    assert got == pytest.approx(9.3e-3, rel=0.02)


def test_a_nucleus_without_spin_data_is_refused():
    with pytest.raises(KeyError, match="31P"):
        metrics.equilibrium_magnetisation(Medium(3.0, "31P"))


def test_the_noise_covariance_is_the_conductivity_weighted_field_product():
    field = _random((2, 3, 2, 2, 1), seed=1)
    sigma = torch.rand(2, 2, 1, dtype=torch.float64)
    mask = torch.tensor([[[True], [False]], [[True], [True]]])
    resolution = 0.01
    got = metrics.noise_covariance(field, sigma, mask, resolution)
    weight = (resolution**3 * sigma * mask).to(torch.complex128)
    expected = torch.einsum("pcxyz,qcxyz,xyz->pq", field, field.conj(), weight)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(got, got.conj().T)


def test_the_coil_adds_its_resistance_between_every_pair_of_channels():
    field = torch.zeros((2, 3, 1, 1, 1), dtype=torch.complex128)
    coil = _random((2, 4), seed=2)
    loss = torch.diag(torch.tensor([1.0, 2.0, 0.5, 0.0])).to(torch.complex128)
    got = metrics.noise_covariance(
        field, torch.zeros(1, 1, 1), torch.ones(1, 1, 1), 0.01, coil=coil, loss=loss
    )
    expected = coil @ loss.real.to(coil.dtype) @ coil.conj().T
    torch.testing.assert_close(got, expected)


def test_one_channel_s_snr_is_its_sensitivity_over_the_root_of_its_noise():
    medium = Medium(3.0)
    b1 = _random((1, 2, 2, 2), seed=3)
    psi = torch.tensor([[4.0 + 0j]], dtype=torch.complex128)
    scale = (
        0.01**3
        * medium.angular_frequency
        * metrics.equilibrium_magnetisation(medium)
        / math.sqrt(4 * metrics.BOLTZMANN * metrics.BODY_TEMPERATURE)
    )
    got = metrics.snr(b1, psi, medium, 0.01)
    torch.testing.assert_close(got, scale * b1[0].abs() / 2.0)


def test_combined_snr_does_not_change_when_the_channels_are_recombined():
    """Any invertible recombination of the receivers carries the same information."""
    medium = Medium(3.0)
    b1 = _random((3, 2, 2, 1), seed=4)
    psi = _covariance(3, seed=5)
    mixing = _covariance(3, seed=6)
    recombined = torch.einsum("pq,qxyz->pxyz", mixing, b1)
    psi_recombined = mixing @ psi @ mixing.conj().T
    torch.testing.assert_close(
        metrics.snr(recombined, psi_recombined, medium, 0.01),
        metrics.snr(b1, psi, medium, 0.01),
    )


def test_transmit_efficiency_is_the_largest_eigenvalue_it_names():
    b1 = _random((3, 2, 1, 1), seed=7)
    psi = _covariance(3, seed=8)
    got = metrics.transmit_efficiency(b1, psi)
    inverse = torch.linalg.inv(psi).conj()
    for x in range(2):
        row = b1[:, x, 0, 0]
        matrix = inverse @ torch.outer(row.conj(), row)
        largest = torch.linalg.eigvals(matrix).real.max()
        torch.testing.assert_close(got[x, 0, 0], largest)


def test_without_acceleration_the_g_factor_is_one():
    sensitivity = _random((4, 6, 4), seed=9)
    mask = torch.ones(6, 4, dtype=torch.bool)
    mask[0, 0] = False
    g, inverse = metrics.g_factor(_covariance(4, seed=10), sensitivity, mask, 1)
    torch.testing.assert_close(g[mask], torch.ones(23, dtype=torch.float64))
    assert float(g[0, 0]) == 0.0 and float(inverse[0, 0]) == 0.0


def test_the_g_factor_of_two_folded_pixels_is_the_sense_formula():
    """Two channels, two pixels half a field of view apart, identity noise."""
    sensitivity = torch.tensor([[[1.0], [0.3]], [[0.2], [1.0]]], dtype=torch.complex128)
    mask = torch.ones(2, 1, dtype=torch.bool)
    g, _ = metrics.g_factor(torch.eye(2, dtype=torch.complex128), sensitivity, mask, 2)
    s = sensitivity[:, :, 0]
    gram = s.conj().T @ s
    expected = torch.sqrt(
        (torch.diagonal(torch.linalg.inv(gram)) * torch.diagonal(gram)).real
    )
    torch.testing.assert_close(g[:, 0], expected)
