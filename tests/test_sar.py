"""Local SAR written as a matrix in the channel drives."""

import pytest
import torch

from mariepy import sar

CHANNELS = 4
SHAPE = (3, 4, 2)
RESOLUTION = 0.002


def _case(device, components=3, seed=0):
    """Random channel fields over a small grid, with half the voxels tissue."""
    generator = torch.Generator().manual_seed(seed)
    electric = torch.randn(
        CHANNELS, components, *SHAPE, dtype=torch.complex128, generator=generator
    ).to(device)
    conductivity = torch.rand(SHAPE, dtype=torch.float64, generator=generator).to(
        device
    )
    density = (
        900.0 + 300.0 * torch.rand(SHAPE, dtype=torch.float64, generator=generator)
    ).to(device)
    mask = torch.zeros(SHAPE, dtype=torch.bool, device=device)
    mask.reshape(-1)[: mask.numel() // 2] = True
    return electric, conductivity, density, mask


def test_the_sar_of_a_drive_is_the_ohmic_loss_of_the_field_it_makes(device):
    """v^H Q v is sigma |sum_c v_c E_c|^2 / 2 rho, the definition it stands for."""
    electric, conductivity, density, mask = _case(device)
    matrices = sar.local_matrices(electric, conductivity, density, mask)
    generator = torch.Generator().manual_seed(5)
    drive = torch.randn(CHANNELS, dtype=torch.complex128, generator=generator).to(
        device
    )

    combined = torch.einsum("c,caxyz->axyz", drive, electric)
    direct = (
        conductivity * (combined.abs() ** 2).sum(dim=0) / (2.0 * density)
    ).reshape(-1)[mask.reshape(-1)]
    torch.testing.assert_close(sar.sar(matrices, drive), direct)


def test_every_local_matrix_is_hermitian(device):
    electric, conductivity, density, mask = _case(device)
    matrices = sar.local_matrices(electric, conductivity, density, mask)
    assert matrices.shape == (int(mask.sum()), CHANNELS, CHANNELS)
    torch.testing.assert_close(matrices, matrices.conj().transpose(-2, -1))


def test_no_drive_dissipates_negative_power(device):
    """The matrices are positive semi-definite, being sums of outer products."""
    electric, conductivity, density, mask = _case(device)
    matrices = sar.local_matrices(electric, conductivity, density, mask)
    values = torch.linalg.eigvalsh(matrices)
    assert float(values.min()) >= -1e-12 * float(values.max())


def test_the_linear_basis_is_read_at_the_cell_centres(device):
    electric, conductivity, density, mask = _case(device, components=12)
    linear = sar.local_matrices(electric, conductivity, density, mask)
    constant = sar.local_matrices(electric[:, 0::4], conductivity, density, mask)
    torch.testing.assert_close(linear, constant)


def test_a_field_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="n_channels"):
        sar.local_matrices(
            torch.zeros(2, 5, 3, 4, 2, dtype=torch.complex128),
            torch.zeros(SHAPE, dtype=torch.float64),
            1000.0,
        )


def test_the_whole_body_matrix_is_its_power_over_its_mass(device):
    """The mass-weighted mean of the voxel matrices is the region's own matrix."""
    electric, conductivity, density, mask = _case(device)
    matrices = sar.local_matrices(electric, conductivity, density, mask)
    mass = sar.voxel_mass(density, RESOLUTION, mask)
    whole = sar.average(matrices, mass)

    generator = torch.Generator().manual_seed(6)
    drive = torch.randn(CHANNELS, dtype=torch.complex128, generator=generator).to(
        device
    )
    combined = torch.einsum("c,caxyz->axyz", drive, electric)
    power = (
        (0.5 * conductivity * (combined.abs() ** 2).sum(dim=0) * RESOLUTION**3)
        .reshape(-1)[mask.reshape(-1)]
        .sum()
    )
    torch.testing.assert_close(sar.sar(whole[None], drive)[0], power / mass.sum())


def test_a_uniform_density_gives_every_voxel_the_same_mass(device):
    _, _, _, mask = _case(device)
    mass = sar.voxel_mass(1000.0, RESOLUTION, mask)
    torch.testing.assert_close(mass, torch.full_like(mass, 1000.0 * RESOLUTION**3))


def test_several_drives_are_evaluated_at_once(device):
    electric, conductivity, density, mask = _case(device)
    matrices = sar.local_matrices(electric, conductivity, density, mask)
    generator = torch.Generator().manual_seed(7)
    drives = torch.randn(3, CHANNELS, dtype=torch.complex128, generator=generator).to(
        device
    )
    together = sar.sar(matrices, drives)
    assert together.shape == (3, int(mask.sum()))
    for index, drive in enumerate(drives):
        torch.testing.assert_close(together[index], sar.sar(matrices, drive))
    torch.testing.assert_close(sar.peak(matrices, drives), together.max(dim=-1).values)


def test_a_region_of_no_mass_has_no_average(device):
    electric, conductivity, density, mask = _case(device)
    matrices = sar.local_matrices(electric, conductivity, density, mask)
    with pytest.raises(ValueError, match="non-zero mass"):
        sar.average(matrices, torch.zeros(matrices.shape[0], dtype=torch.float64))
