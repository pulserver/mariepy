"""The analytic Mie series for a plane wave on a homogeneous or layered sphere.

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

    # Bohren and Huffman's 4.53 in terms of Riccati functions: the internal
    # field is expanded in j_n(m x), so the magnetic coefficient carries m.
    c = (
        index
        * (psi_x * dxi_x - dpsi_x * xi_x)
        / (psi_mx * dxi_x - index * dpsi_mx * xi_x)
    )
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


def layered_coefficients(radii, indices, wavenumber, orders):
    """Solve the layered sphere's expansion coefficients, order by order.

    In layer ``l`` the radial functions are ``A j_n + B y_n`` of ``m_l k r``,
    with ``B = 0`` in the core. Writing ``u = rho z_n(rho)`` for them, the
    tangential fields match across the interface at ``x = k r_l`` as

    - magnetic multipoles: ``c u / m`` and ``c u'`` continuous,
    - electric multipoles: ``d u' / m`` and ``d u`` continuous,

    and outside the sphere ``c u = psi - b xi`` and ``d u = psi - a xi``
    (Bohren and Huffman, section 8.1, extended to any number of layers).

    Parameters
    ----------
    radii
        Outer radius of each layer, increasing, in metres.
    indices
        Complex refractive index of each layer, in Bohren and Huffman's
        convention.
    wavenumber
        Free-space wavenumber.
    orders
        Multipole orders.

    Returns
    -------
    list of tuple
        Per layer ``(cj, cy, dj, dy)``, each an array over the orders.
    """
    layers = len(radii)
    out = [
        tuple(np.zeros(orders.size, dtype=complex) for _ in range(4))
        for _ in range(layers)
    ]

    def functions(n, rho):
        j, dj = spherical_jn(n, rho), spherical_jn(n, rho, derivative=True)
        y, dy = spherical_yn(n, rho), spherical_yn(n, rho, derivative=True)
        return (rho * j, j + rho * dj), (rho * y, y + rho * dy)

    for row, n in enumerate(orders):
        for magnetic in (True, False):
            size = 2 * layers
            matrix = np.zeros((size, size), dtype=complex)
            rhs = np.zeros(size, dtype=complex)

            # Unknowns: core A, then (A, B) of each outer layer, then the
            # scattering coefficient.
            def column(layer, kind):
                if layer == 0:
                    return 0
                return 2 * layer - 1 + kind

            for interface in range(layers):
                x = wavenumber * radii[interface]
                inner = interface
                m_in = indices[inner]
                (psi, dpsi), (chi, dchi) = functions(n, m_in * x)
                first, second = 2 * interface, 2 * interface + 1
                pairs = (
                    [(psi, dpsi, 0)] if inner == 0 else [(psi, dpsi, 0), (chi, dchi, 1)]
                )
                for u, du, kind in pairs:
                    col = column(inner, kind)
                    if magnetic:
                        matrix[first, col] += u / m_in
                        matrix[second, col] += du
                    else:
                        matrix[first, col] += du / m_in
                        matrix[second, col] += u
                if interface + 1 < layers:
                    m_out = indices[interface + 1]
                    (psi, dpsi), (chi, dchi) = functions(n, m_out * x)
                    for u, du, kind in ((psi, dpsi, 0), (chi, dchi, 1)):
                        col = column(interface + 1, kind)
                        if magnetic:
                            matrix[first, col] -= u / m_out
                            matrix[second, col] -= du
                        else:
                            matrix[first, col] -= du / m_out
                            matrix[second, col] -= u
                else:
                    (psi, dpsi), (chi, dchi) = functions(n, x)
                    xi, dxi = psi + 1j * chi, dpsi + 1j * dchi
                    last = size - 1
                    # outside: psi - s xi, with index one
                    if magnetic:
                        matrix[first, last] += xi
                        matrix[second, last] += dxi
                        rhs[first] += psi
                        rhs[second] += dpsi
                    else:
                        matrix[first, last] += dxi
                        matrix[second, last] += xi
                        rhs[first] += dpsi
                        rhs[second] += psi
            solution = np.linalg.solve(matrix, rhs)
            for layer in range(layers):
                a = solution[column(layer, 0)]
                b = 0.0 if layer == 0 else solution[column(layer, 1)]
                slot = (0, 1) if magnetic else (2, 3)
                out[layer][slot[0]][row] = a
                out[layer][slot[1]][row] = b
    return out


def layered_internal_field(points, radii, indices, wavenumber, n_terms=None):
    """Return the electric field inside a layered sphere lit by a plane wave.

    As :func:`internal_field`, for concentric layers.

    Parameters
    ----------
    points
        Shape ``(n, 3)``, positions in metres, inside the outer radius.
    radii
        Outer radius of each layer, increasing, in metres.
    indices
        Complex refractive index of each layer relative to free space, in
        Bohren and Huffman's convention.
    wavenumber
        Free-space wavenumber in rad/m.
    n_terms
        Multipole orders to keep.

    Returns
    -------
    ndarray
        Shape ``(n, 3)``, complex, in mariepy's time convention.
    """
    radii = np.asarray(radii, dtype=float)
    indices = np.asarray(indices, dtype=complex)
    size = wavenumber * radii[-1]
    if n_terms is None:
        largest = np.abs(indices).max() * size
        n_terms = int(np.ceil(largest + 4.05 * largest ** (1 / 3) + 10))
    orders = np.arange(1, n_terms + 1)
    coefficients = layered_coefficients(radii, indices, wavenumber, orders)

    r = np.linalg.norm(points, axis=1)
    layer = np.searchsorted(radii, r, side="left")
    layer = np.minimum(layer, len(radii) - 1)
    safe = np.where(r == 0, 1.0, r)
    cos_theta = np.clip(points[:, 2] / safe, -1.0, 1.0)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = np.arctan2(points[:, 1], points[:, 0])
    rho = indices[layer] * wavenumber * r
    pi, tau = _angular(orders, cos_theta)

    field_r = np.zeros(points.shape[0], dtype=complex)
    field_theta = np.zeros_like(field_r)
    field_phi = np.zeros_like(field_r)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    at_origin = rho == 0
    safe_rho = np.where(at_origin, 1.0, rho)
    for row, n in enumerate(orders):
        prefactor = 1j**n * (2 * n + 1) / (n * (n + 1))
        cj = np.array([coefficients[i][0][row] for i in range(len(radii))])[layer]
        cy = np.array([coefficients[i][1][row] for i in range(len(radii))])[layer]
        dj = np.array([coefficients[i][2][row] for i in range(len(radii))])[layer]
        dy = np.array([coefficients[i][3][row] for i in range(len(radii))])[layer]
        j = spherical_jn(n, safe_rho)
        dj_ = spherical_jn(n, safe_rho, derivative=True)
        with np.errstate(all="ignore"):
            y = np.where(cy != 0, spherical_yn(n, safe_rho), 0.0)
            dy_ = np.where(cy != 0, spherical_yn(n, safe_rho, derivative=True), 0.0)
        magnetic = cj * j + cy * y
        magnetic_over = (cj * j + cy * y) / safe_rho
        electric_over = (
            dj * (j + safe_rho * dj_) + dy * (y + safe_rho * dy_)
        ) / safe_rho
        electric_plain_over = (dj * j + dy * y) / safe_rho
        if n == 1:
            magnetic_over = np.where(at_origin, cj / 3.0, magnetic_over)
            electric_over = np.where(at_origin, dj * 2.0 / 3.0, electric_over)
            electric_plain_over = np.where(at_origin, dj / 3.0, electric_plain_over)
        else:
            electric_over = np.where(at_origin, 0.0, electric_over)
            electric_plain_over = np.where(at_origin, 0.0, electric_plain_over)
        magnetic = np.where(at_origin, 0.0, magnetic)
        del magnetic_over

        field_r += prefactor * (
            -1j * n * (n + 1) * cos_phi * sin_theta * pi[row] * electric_plain_over
        )
        field_theta += prefactor * (
            cos_phi * pi[row] * magnetic - 1j * cos_phi * tau[row] * electric_over
        )
        field_phi += prefactor * (
            -sin_phi * tau[row] * magnetic + 1j * sin_phi * pi[row] * electric_over
        )

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
