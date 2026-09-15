"""Electromagnetic constants at a field strength and nucleus.

Ported from MARIE 3.0's ``src_physics/em_constants.m``. The nuclear spins and
number densities that file also carries describe thermal magnetisation, which
enters the SNR maps rather than the solver, and are added when those are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["NUCLEI", "Medium"]

# Gyromagnetic ratio over 2 pi, in Hz/T. Negative for a nucleus whose moment is
# antiparallel to its spin; the Larmor frequency takes the magnitude.
NUCLEI: dict[str, float] = {
    "1H": 42.577478518e6,
    "2H": 6.536e6,
    "3HE": 32.434e6,
    "13C": 10.705e6,
    "7LI": 16.546e6,
    "15N": -4.316e6,
    "17O": -5.772e6,
    "23NA": 11.262e6,
    "25MG": -2.614e6,
    "31P": 17.235e6,
    "35CL": 4.171e6,
    "39K": 1.250e6,
    "59CO": 10.054e6,
    "79BR": 10.666e6,
    "87RB": 13.932e6,
    "129XE": 11.777e6,
}

SPEED_OF_LIGHT = 299792458.0
VACUUM_PERMEABILITY = 4.0e-7 * math.pi
COPPER_CONDUCTIVITY = 5.96e7


@dataclass(frozen=True)
class Medium:
    """Free-space and conductor constants at one field strength and nucleus.

    Parameters
    ----------
    field_strength
        Static field in tesla.
    nucleus
        Key of :data:`NUCLEI`, case-insensitive.

    Raises
    ------
    KeyError
        The nucleus is not tabulated.

    Examples
    --------
    >>> medium = Medium(3.0)
    >>> round(medium.frequency / 1e6, 3)
    127.732
    """

    field_strength: float
    nucleus: str = "1H"

    @property
    def gyromagnetic_ratio(self) -> float:
        """Return the nucleus's gyromagnetic ratio over 2 pi, in Hz/T."""
        key = self.nucleus.upper()
        if key not in NUCLEI:
            available = ", ".join(sorted(NUCLEI))
            raise KeyError(
                f"no gyromagnetic ratio for {self.nucleus!r}; have {available}"
            )
        return NUCLEI[key]

    @property
    def frequency(self) -> float:
        """Return the Larmor frequency in Hz."""
        return abs(self.gyromagnetic_ratio) * self.field_strength

    @property
    def angular_frequency(self) -> float:
        """Return the Larmor frequency in rad/s."""
        return 2.0 * math.pi * self.frequency

    @property
    def permittivity(self) -> float:
        """Return the permittivity of free space in F/m."""
        return 1.0 / (SPEED_OF_LIGHT**2 * VACUUM_PERMEABILITY)

    @property
    def permeability(self) -> float:
        """Return the permeability of free space in H/m."""
        return VACUUM_PERMEABILITY

    @property
    def wavelength(self) -> float:
        """Return the free-space wavelength in m."""
        return SPEED_OF_LIGHT / self.frequency

    @property
    def wavenumber(self) -> float:
        """Return the free-space wavenumber in rad/m."""
        return self.angular_frequency / SPEED_OF_LIGHT

    @property
    def impedance(self) -> float:
        """Return the impedance of free space in ohm."""
        return VACUUM_PERMEABILITY * SPEED_OF_LIGHT

    @property
    def electric_scaling(self) -> complex:
        """Return ``j omega epsilon_0``, MARIE's ``ce``."""
        return 1j * self.angular_frequency * self.permittivity

    @property
    def magnetic_scaling(self) -> complex:
        """Return ``j omega mu_0``, MARIE's ``cm``."""
        return 1j * self.angular_frequency * VACUUM_PERMEABILITY

    @property
    def skin_depth(self) -> float:
        """Return the skin depth of copper at this frequency, in m."""
        return math.sqrt(2.0) / math.sqrt(
            self.angular_frequency * VACUUM_PERMEABILITY * COPPER_CONDUCTIVITY
        )

    @property
    def surface_resistance(self) -> float:
        """Return the surface resistance of copper in ohm per square."""
        return 1.0 / (COPPER_CONDUCTIVITY * self.skin_depth)
