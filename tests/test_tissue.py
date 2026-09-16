"""Tissue properties, and the body a labelled volume makes."""

import cmath

import pytest
import torch

from mariepy import tissue
from mariepy.constants import Medium

# A dispersion of the shape the 1996 paper prints, in its own units: four
# relaxations spread over ps, ns, us and ms, and a static ionic conductivity.
EXAMPLE = {
    "ef": 4.0,
    "del1": 50.0,
    "tau1": 7.23,
    "alf1": 0.1,
    "del2": 7000.0,
    "tau2": 353.68,
    "alf2": 0.1,
    "del3": 1.2e6,
    "tau3": 318.31,
    "alf3": 0.1,
    "del4": 2.5e7,
    "tau4": 2.274,
    "alf4": 0.0,
    "sig": 0.2,
}


def _dispersion(**changes):
    values = EXAMPLE | changes
    row = [values[column] for column in tissue.GABRIEL_COLUMNS]
    for term, scale in enumerate(tissue._TIME_SCALE):
        row[2 + 3 * term] *= scale
    return torch.tensor(row, dtype=torch.float64)


def _by_hand(row, frequency):
    """The four-Cole-Cole sum written out term by term."""
    omega = 2.0 * cmath.pi * frequency
    value = complex(row[0])
    for term in range(4):
        strength, relaxation, broadening = (
            float(row[1 + 3 * term]),
            float(row[2 + 3 * term]),
            float(row[3 + 3 * term]),
        )
        value += strength / (1.0 + (1j * omega * relaxation) ** (1.0 - broadening))
    vacuum = Medium(1.0).permittivity
    value -= 1j * float(row[13]) / (omega * vacuum)
    return value.real, -omega * vacuum * value.imag


@pytest.mark.parametrize("frequency", [1e3, 1e6, 63.87e6, 127.74e6, 1e9])
def test_the_model_is_the_sum_of_the_terms_it_is_written_as(frequency, device):
    row = _dispersion().to(device)
    permittivity, conductivity = tissue.dielectric(row, frequency)
    want_permittivity, want_conductivity = _by_hand(row.cpu(), frequency)
    assert float(permittivity) == pytest.approx(want_permittivity, rel=1e-12)
    assert float(conductivity) == pytest.approx(want_conductivity, rel=1e-12)


def test_a_stack_of_tissues_is_evaluated_at_once(device):
    rows = torch.stack([_dispersion(), _dispersion(sig=0.5), _dispersion(ef=10.0)])
    permittivity, conductivity = tissue.dielectric(rows.to(device), 127.74e6)
    assert permittivity.shape == (3,)
    for index, row in enumerate(rows):
        one = tissue.dielectric(row.to(device), 127.74e6)
        assert float(permittivity[index]) == pytest.approx(float(one[0]))
        assert float(conductivity[index]) == pytest.approx(float(one[1]))


def test_far_below_every_relaxation_only_the_ionic_conductivity_is_left(device):
    """The dispersions are lossless once the frequency is well under their poles."""
    row = _dispersion(alf1=0.0, alf2=0.0, alf3=0.0, alf4=0.0).to(device)
    _, conductivity = tissue.dielectric(row, 1e-6)
    assert float(conductivity) == pytest.approx(EXAMPLE["sig"], rel=1e-6)


def test_far_above_every_relaxation_the_permittivity_is_the_one_left_over(device):
    row = _dispersion(alf1=0.0, alf2=0.0, alf3=0.0, alf4=0.0).to(device)
    permittivity, _ = tissue.dielectric(row, 1e18)
    assert float(permittivity) == pytest.approx(EXAMPLE["ef"], rel=1e-6)


def test_the_dispersions_only_ever_add_loss(device):
    row = _dispersion().to(device)
    for frequency in (1e2, 1e5, 1e8, 1e11):
        _, conductivity = tissue.dielectric(row, frequency)
        assert float(conductivity) >= EXAMPLE["sig"]


def test_a_row_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="14 parameters"):
        tissue.dielectric(torch.zeros(6, dtype=torch.float64), 1e6)


def test_a_frequency_of_zero_is_refused():
    with pytest.raises(ValueError, match="positive frequency"):
        tissue.dielectric(_dispersion(), 0.0)


def _write(path, rows):
    columns = ["label", "name", "density", *tissue.GABRIEL_COLUMNS]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n")
    return path


def _row(label, name, density=1050.0, **changes):
    return {"label": label, "name": name, "density": density, **(EXAMPLE | changes)}


def test_a_table_reads_its_relaxation_times_in_the_paper_s_units(tmp_path):
    path = _write(tmp_path / "t.csv", [_row(1, "muscle")])
    table = tissue.read_table(path)
    assert set(table) == {1}
    assert table[1].name == "muscle"
    assert table[1].density == 1050.0
    torch.testing.assert_close(table[1].dispersion, _dispersion())


def test_a_table_missing_a_column_is_refused(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("label,name,density\n1,muscle,1050\n")
    with pytest.raises(ValueError, match="del1"):
        tissue.read_table(path)


def test_a_table_naming_a_label_twice_is_refused(tmp_path):
    path = _write(tmp_path / "t.csv", [_row(1, "muscle"), _row(1, "fat")])
    with pytest.raises(ValueError, match="appears twice"):
        tissue.read_table(path)


def test_a_table_that_names_nothing_is_refused(tmp_path):
    path = _write(tmp_path / "t.csv", [])
    with pytest.raises(ValueError, match="names no tissue"):
        tissue.read_table(path)


def _labels(device):
    labels = torch.zeros((4, 4, 4), dtype=torch.long, device=device)
    labels[1:3, 1:3, 1:3] = 1
    labels[0, 0, 0] = 2
    labels[3, 3, 3] = 7  # a label that means free space here
    return labels


def test_a_label_volume_becomes_a_body_at_the_working_frequency(tmp_path, device):
    path = _write(
        tmp_path / "t.csv",
        [_row(1, "muscle", density=1050.0), _row(2, "fat", density=950.0, sig=0.05)],
    )
    table = tissue.read_table(path)
    medium = Medium(3.0)
    built = tissue.build(_labels(device), table, medium, 0.002, background=(0, 7))

    assert built.body.resolution == 0.002
    assert int(built.body.mask.sum()) == 9
    assert not bool(built.body.mask[3, 3, 3])
    assert float(built.density[1, 1, 1]) == 1050.0
    assert float(built.density[0, 0, 0]) == 950.0
    assert float(built.density[3, 3, 3]) == 0.0

    want = tissue.dielectric(table[1].dispersion, medium.frequency)
    assert float(built.body.permittivity[1, 1, 1]) == pytest.approx(float(want[0]))
    assert float(built.body.conductivity[1, 1, 1]) == pytest.approx(float(want[1]))
    assert float(built.body.permittivity[3, 3, 3]) == 1.0
    assert float(built.body.conductivity[3, 3, 3]) == 0.0


def test_a_body_is_centred_on_the_grid_unless_told_otherwise(tmp_path, device):
    path = _write(tmp_path / "t.csv", [_row(1, "muscle"), _row(2, "fat")])
    table = tissue.read_table(path)
    built = tissue.build(_labels(device), table, Medium(3.0), 0.002, background=(0, 7))
    assert built.body.origin == (-0.003, -0.003, -0.003)
    moved = tissue.build(
        _labels(device),
        table,
        Medium(3.0),
        0.002,
        origin=(0.0, 0.0, 0.0),
        background=(0, 7),
    )
    assert moved.body.origin == (0.0, 0.0, 0.0)


def test_a_volume_with_no_named_label_is_refused(tmp_path, device):
    path = _write(tmp_path / "t.csv", [_row(5, "muscle")])
    table = tissue.read_table(path)
    with pytest.raises(ValueError, match="no voxel carries"):
        tissue.build(
            _labels(device), table, Medium(3.0), 0.002, background=(0, 1, 2, 7)
        )


def test_a_label_the_table_leaves_out_is_refused(tmp_path, device):
    """A tissue missing from the table would quietly become air."""
    path = _write(tmp_path / "t.csv", [_row(1, "muscle"), _row(2, "fat")])
    table = tissue.read_table(path)
    with pytest.raises(ValueError, match="does not name: 7"):
        tissue.build(_labels(device), table, Medium(3.0), 0.002)


# Muscle, white matter and cortical bone, as the 1996 appendix gives them, with
# the relaxation times in the paper's own units. These three carry the same
# properties in Gabriel's fit and in the IT'IS database, so the values MARIE's
# Hugo head at 3 T stores for them check the model rather than the table.
VERIFIED = {
    "muscle": (
        (
            4.0,
            50.0,
            7.23,
            0.1,
            7000.0,
            353.68,
            0.1,
            1.2e6,
            318.31,
            0.1,
            2.5e7,
            2.274,
            0.0,
            0.2,
        ),
        (63.5, 0.717),
    ),
    "white matter": (
        (
            4.0,
            32.0,
            7.958,
            0.1,
            100.0,
            7.958,
            0.1,
            4.0e4,
            53.052,
            0.3,
            3.5e7,
            7.958,
            0.02,
            0.02,
        ),
        (52.5, 0.339),
    ),
    "bone cortical": (
        (
            2.5,
            10.0,
            13.26,
            0.2,
            180.0,
            79.577,
            0.2,
            5.0e3,
            159.155,
            0.2,
            1.0e5,
            15.915,
            0.0,
            0.02,
        ),
        (14.7, 0.0670),
    ),
}

LARMOR_3T = 127.74e6


@pytest.mark.parametrize("name", sorted(VERIFIED))
def test_the_dispersion_gives_the_properties_a_head_model_carries(name):
    """The model, on published parameters, lands on the numbers a head model uses."""
    row, (want_permittivity, want_conductivity) = VERIFIED[name]
    values = list(row)
    for term, scale in enumerate(tissue._TIME_SCALE):
        values[2 + 3 * term] *= scale
    permittivity, conductivity = tissue.dielectric(
        torch.tensor(values, dtype=torch.float64), LARMOR_3T
    )
    assert float(permittivity) == pytest.approx(want_permittivity, rel=5e-3)
    assert float(conductivity) == pytest.approx(want_conductivity, rel=1e-2)
