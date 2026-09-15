"""Scattering by a perfectly conducting sphere, from the Mie series.

The coefficients are Bohren and Huffman, *Absorption and Scattering of Light by
Small Particles*, section 4.3, in the perfect-conductor limit: the tangential
electric field vanishes on the surface, which leaves

``a_n = [x j_n(x)]' / [x h_n(x)]'``  and  ``b_n = j_n(x) / h_n(x)``

for the transverse-magnetic and transverse-electric partial waves, with
``x = k a``. A perfect conductor absorbs nothing, so its extinction is its
scattering, and the cross section is the one quantity a surface-integral solve
delivers without evaluating a far field: the power the incident field does on
the induced current.
"""

import math

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def scattering_cross_section(size: float, terms: int = 40) -> float:
    """Return the scattering cross section of a conducting sphere, over ``k ** 2``.

    Parameters
    ----------
    size
        Size parameter ``k a``.
    terms
        Number of partial waves summed.

    Returns
    -------
    float
        ``sigma_sca * k ** 2``, which is dimensionless; the caller divides by
        the square of its own wavenumber.
    """
    order = np.arange(1, terms + 1)
    bessel = spherical_jn(order, size)
    bessel_prime = spherical_jn(order, size, derivative=True)
    neumann = spherical_yn(order, size)
    neumann_prime = spherical_yn(order, size, derivative=True)
    hankel = bessel + 1j * neumann
    hankel_prime = bessel_prime + 1j * neumann_prime

    magnetic = bessel / hankel
    electric = (bessel + size * bessel_prime) / (hankel + size * hankel_prime)
    weight = 2 * order + 1
    return (
        2.0
        * math.pi
        * float(np.sum(weight * (np.abs(electric) ** 2 + np.abs(magnetic) ** 2)))
    )
