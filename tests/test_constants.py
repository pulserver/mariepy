"""The electromagnetic constants agree with their published values."""

import math

import pytest

from mariepy.constants import NUCLEI, Medium


def test_the_proton_larmor_frequency_at_three_tesla():
    assert Medium(3.0).frequency == pytest.approx(127.73e6, rel=1e-4)


@pytest.mark.parametrize(
    ("field_strength", "megahertz"),
    [(1.5, 63.87), (3.0, 127.73), (7.0, 298.04), (11.7, 498.16)],
)
def test_the_proton_larmor_frequency_scales_with_the_field(field_strength, megahertz):
    assert Medium(field_strength).frequency / 1e6 == pytest.approx(megahertz, rel=1e-3)


def test_the_permittivity_of_free_space():
    assert Medium(3.0).permittivity == pytest.approx(8.8541878128e-12, rel=1e-9)


def test_the_impedance_of_free_space():
    assert Medium(3.0).impedance == pytest.approx(376.730313, rel=1e-6)


def test_the_wavenumber_is_two_pi_over_the_wavelength():
    medium = Medium(7.0)
    assert medium.wavenumber == pytest.approx(2.0 * math.pi / medium.wavelength)


def test_the_electric_scaling_is_the_displacement_current_factor():
    medium = Medium(3.0)
    assert medium.electric_scaling == pytest.approx(
        1j * medium.angular_frequency * medium.permittivity
    )


def test_the_skin_depth_of_copper_shrinks_as_the_field_rises():
    assert Medium(7.0).skin_depth < Medium(1.5).skin_depth
    # A few micrometres at proton Larmor frequencies.
    assert 1e-6 < Medium(3.0).skin_depth < 1e-5


def test_the_surface_resistance_is_set_by_the_skin_depth():
    medium = Medium(3.0)
    assert medium.surface_resistance == pytest.approx(
        1.0 / (5.96e7 * medium.skin_depth)
    )


def test_a_nucleus_with_a_negative_moment_still_has_a_positive_frequency():
    assert NUCLEI["15N"] < 0
    assert Medium(3.0, "15N").frequency > 0


def test_the_nucleus_name_is_read_without_regard_to_case():
    assert Medium(3.0, "1h").frequency == Medium(3.0, "1H").frequency


def test_an_untabulated_nucleus_is_refused():
    with pytest.raises(KeyError, match="no gyromagnetic ratio"):
        _ = Medium(3.0, "42XX").frequency
