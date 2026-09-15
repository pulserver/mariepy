"""The analytic Mie series for a plane wave on a homogeneous sphere.

Written from the standard expansion (Bohren and Huffman, *Absorption and
Scattering of Light by Small Particles*, chapter 4), not ported from MARIE:
the point of it is to be an independent answer.

The wave travels along z and is polarised along x. Inside a sphere of complex
refractive index ``m`` the field is the series

    E = sum_n E_n ( c_n M_o1n - j d_n N_e1n )

with ``E_n = j**n E_0 (2n + 1) / (n (n + 1))``, and the vector spherical
harmonics evaluated with the spherical Bessel function of the first kind, which
is the one regular at the origin.
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def _riccati_j(n, x):
    """Return psi_n(x) = x j_n(x) and its derivative."""
    return x * spherical_jn(n, x), spherical_jn(n, x) + x * spherical_jn(
        n, x, derivative=True
    )


def _riccati_h(n, x):
    """Return xi_n(x) = x h_n(x) and its derivative, for the outgoing wave."""
    h = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    dh = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    return x * h, h + x * dh


def internal_coefficients(index, size, orders):
    """Return the internal coefficients c_n and d_n.

    Parameters
    ----------
    index
        Complex refractive index of the sphere, relative to the surroundings.
    size
        Size parameter ``k a`` with ``k`` the wavenumber outside.
    orders
        Array of multipole orders, starting at 1.

    Returns
    -------
    c : ndarray
        Coefficient of the magnetic multipole inside.
    d : ndarray
        Coefficient of the electric multipole inside.
    """
    x = size
    mx = index * size
    psi_x, dpsi_x = _riccati_j(orders, x)
    psi_mx, dpsi_mx = _riccati_j(orders, mx)
    xi_x, dxi_x = _riccati_h(orders, x)

    c = (psi_x * dxi_x - dpsi_x * xi_x) / (psi_mx * dxi_x - index * dpsi_mx * xi_x)
    d = (index * (psi_x * dxi_x - dpsi_x * xi_x)) / (
        index * psi_mx * dxi_x - dpsi_mx * xi_x
    )
    return c, d


def _angular(orders, cos_theta):
    """Return pi_n and tau_n at one set of angles, by upward recurrence."""
    pi = np.zeros((orders.size, cos_theta.size))
    tau = np.zeros((orders.size, cos_theta.size))
    previous = np.zeros(cos_theta.size)
    current = np.ones(cos_theta.size)
    for row, n in enumerate(orders):
        pi[row] = current
        tau[row] = n * cos_theta * current - (n + 1) * previous
        previous, current = (
            current,
            ((2 * n + 1) * cos_theta * current - (n + 1) * previous) / n,
        )
    return pi, tau


def internal_field(points, radius, index, wavenumber, n_terms=None):
    """Return the electric field inside a sphere lit by a plane wave.

    The wave travels along +z, is polarised along +x and has unit amplitude.

    Parameters
    ----------
    points
        Shape ``(n, 3)``, positions in metres, inside the sphere.
    radius
        Sphere radius in metres.
    index
        Complex refractive index of the sphere relative to free space.
    wavenumber
        Free-space wavenumber in rad/m.
    n_terms
        Multipole orders to keep. The Wiscombe criterion by default.

    Returns
    -------
    ndarray
        Shape ``(n, 3)``, the complex electric field in Cartesian components,
        in mariepy's time convention.

    Notes
    -----
    Bohren and Huffman expand ``exp(+i k z)`` with a time dependence of
    ``exp(-i omega t)``. mariepy's Green function is ``exp(-j k R) / (4 pi R)``,
    which is the other convention, so the series is conjugated on the way out
    and the incident wave this returns the response to is ``exp(-j k z)``.
    """
    size = wavenumber * radius
    if n_terms is None:
        n_terms = int(
            np.ceil(abs(index) * size + 4.05 * (abs(index) * size) ** (1 / 3) + 10)
        )
    orders = np.arange(1, n_terms + 1)

    c, d = internal_coefficients(index, size, orders)

    r = np.linalg.norm(points, axis=1)
    safe = np.where(r == 0, 1.0, r)
    cos_theta = np.clip(points[:, 2] / safe, -1.0, 1.0)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = np.arctan2(points[:, 1], points[:, 0])

    rho = index * wavenumber * r
    pi, tau = _angular(orders, cos_theta)

    field_r = np.zeros(points.shape[0], dtype=complex)
    field_theta = np.zeros_like(field_r)
    field_phi = np.zeros_like(field_r)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)

    for row, n in enumerate(orders):
        prefactor = 1j**n * (2 * n + 1) / (n * (n + 1))
        bessel = spherical_jn(n, rho)
        derivative = spherical_jn(n, rho, derivative=True)

        # j_n(rho)/rho and [rho j_n(rho)]'/rho both have a finite limit at the
        # origin, which j_n ~ rho**n / (2n+1)!! gives as 1/3 and 2/3 for the
        # dipole and zero for every higher order.
        at_origin = rho == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            bessel_over_rho = np.where(at_origin, 0.0, bessel / rho)
            riccati_over_rho = np.where(
                at_origin, 0.0, (bessel + rho * derivative) / rho
            )
        if n == 1:
            bessel_over_rho = np.where(at_origin, 1.0 / 3.0, bessel_over_rho)
            riccati_over_rho = np.where(at_origin, 2.0 / 3.0, riccati_over_rho)

        # Bohren and Huffman 4.50, with z_n the spherical Bessel function.
        m_theta = cos_phi * pi[row] * bessel
        m_phi = -sin_phi * tau[row] * bessel
        n_r = cos_phi * n * (n + 1) * sin_theta * pi[row] * bessel_over_rho
        n_theta = cos_phi * tau[row] * riccati_over_rho
        n_phi = -sin_phi * pi[row] * riccati_over_rho

        field_r += prefactor * (-1j * d[row] * n_r)
        field_theta += prefactor * (c[row] * m_theta - 1j * d[row] * n_theta)
        field_phi += prefactor * (c[row] * m_phi - 1j * d[row] * n_phi)

    cartesian = np.empty_like(points, dtype=complex)
    cartesian[:, 0] = (
        field_r * sin_theta * cos_phi
        + field_theta * cos_theta * cos_phi
        - field_phi * sin_phi
    )
    cartesian[:, 1] = (
        field_r * sin_theta * sin_phi
        + field_theta * cos_theta * sin_phi
        + field_phi * cos_phi
    )
    cartesian[:, 2] = field_r * cos_theta - field_theta * sin_theta
    return np.conjugate(cartesian)
