"""The circular components of a coil's field over a body, written as a file and read back."""

import numpy as np
import pytest
import torch

from mariepy import maps

META = {
    "coil": "two loops",
    "channels": ["a", "b"],
    "frequency_hz": 127.7e6,
    "drive_unit": "1 A into the port, the others open",
    "origin": (-0.01, -0.02, -0.03),
    "resolution": 0.005,
    "frame": "the body's own",
    "bodies": ["ball"],
    "data_licence": "none",
}


def _maps():
    generator = torch.Generator().manual_seed(5)
    shape = (2, 3, 4, 5)
    real, imaginary = torch.randn((2, 2, *shape), generator=generator)
    mask = torch.zeros(shape[1:], dtype=torch.bool)
    mask[1:, 1:3, 2:] = True
    return (
        torch.complex(real[0], imaginary[0]),
        torch.complex(real[1], imaginary[1]),
        mask,
    )


def test_a_map_file_reads_back_what_was_written(tmp_path):
    plus, minus, mask = _maps()
    path = tmp_path / "maps.npz"

    maps.write(path, plus, minus, mask, **META)
    back = maps.read(path)

    torch.testing.assert_close(back.plus, plus.to(torch.complex64))
    torch.testing.assert_close(back.minus, minus.to(torch.complex64))
    assert torch.equal(back.mask, mask)
    assert back.metadata["channels"] == ["a", "b"]
    assert back.metadata["origin"] == [-0.01, -0.02, -0.03]
    assert set(maps.METADATA_KEYS) <= set(back.metadata)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["plus"].dtype == np.complex64


def test_maps_the_names_or_the_mask_do_not_match_are_refused(tmp_path):
    plus, minus, mask = _maps()
    with pytest.raises(ValueError, match="alike"):
        maps.write(tmp_path / "m.npz", plus, minus[:1], mask, **META)
    with pytest.raises(ValueError, match="the mask"):
        maps.write(tmp_path / "m.npz", plus, minus, mask[1:], **META)
    with pytest.raises(ValueError, match="channel names"):
        maps.write(tmp_path / "m.npz", plus, minus, mask, **{**META, "channels": ["a"]})


def test_an_archive_missing_an_entry_is_refused(tmp_path):
    path = tmp_path / "m.npz"
    np.savez(path, plus=np.zeros((1, 2, 2, 2), np.complex64))
    with pytest.raises(ValueError, match="no mask, metadata, minus"):
        maps.read(path)
