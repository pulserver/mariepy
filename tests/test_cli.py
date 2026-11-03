"""The ``mariepy`` command writes a VOP file that reads back."""

import pytest
import torch

from mariepy import vop
from mariepy.cli import main


def test_vops_on_the_sphere_writes_a_file_vop_read_accepts(tmp_path):
    path = tmp_path / "vops.npz"
    main(
        [
            "vops",
            "--sphere",
            "--out",
            str(path),
            "--safety-factor",
            "1.5",
            "--safety-basis",
            "test",
            "--transmit",
            "loop/2/0x0",
        ]
    )
    back = vop.read(path)
    assert back.vops.shape[1:] == (2, 2)
    assert torch.allclose(back.vops, back.vops.mH)
    assert back.metadata["safety_factor"] == 1.5
    assert back.metadata["transmit"] == "loop/2/0x0"


def test_vops_refuses_a_file_without_a_safety_factor(tmp_path):
    with pytest.raises(SystemExit):
        main(["vops", "--sphere", "--out", str(tmp_path / "v.npz")])


def test_vops_refuses_both_a_case_and_the_sphere(tmp_path):
    with pytest.raises(SystemExit):
        main(
            [
                "vops",
                "case.json",
                "--sphere",
                "--out",
                str(tmp_path / "v.npz"),
                "--safety-factor",
                "1",
                "--safety-basis",
                "test",
            ]
        )


def test_vops_writes_the_global_safety_factor_it_is_given(tmp_path):
    path = tmp_path / "vops.npz"
    main(
        [
            "vops",
            "--sphere",
            "--out",
            str(path),
            "--safety-factor",
            "1.5",
            "--safety-basis",
            "test",
            "--global-safety-factor",
            "1.35",
            "--global-safety-basis",
            "vendor cross-check",
        ]
    )
    back = vop.read(path).metadata
    assert back["global_safety_factor"] == 1.35
    assert back["global_safety_basis"] == "vendor cross-check"
