"""The volume integral operator on the body grid.

Ported from MARIE 3.0's ``src_integral_equations/src_vie/src_operators_vie/``:
``cubatures/{G_N,G_K,VV_Nop,VV_Kop,weights_points}.m`` for the kernel, and
``src_solver/src_mvp/mvp_vie/`` for the products.

Two operators act on the polarisation current in the voxels. N is the
double-curl dyadic that gives the electric field, symmetric with six distinct
components in the order xx, xy, xz, yy, yz, zz. K is the curl that gives the
magnetic field, with three components in the order x, y, z. Both depend only on
the offset between two voxels, so each is stored once per offset and applied by
FFT through :mod:`mariepy.tucker`.

Voxels that touch need the surface-surface treatment of MARIE's
``surface_surface_kernels_*.m`` and its singular integrals; the quadrature here
is the volume-volume rule MARIE uses at every other offset, and it diverges as
the offset goes to zero.
"""

from __future__ import annotations

import torch

from mariepy.quadrature import gauss_legendre_1d
from mariepy.tucker import CirculantSymbol

__all__ = [
    "apply_g",
    "apply_inverse_g",
    "apply_k",
    "apply_n",
    "green_k",
    "green_n",
    "volume_volume_k",
    "volume_volume_n",
]

# Which of the six stored components of N carries each (row, column) of the
# dyad, given the order xx, xy, xz, yy, yz, zz.
_DYADIC_INDEX = ((0, 1, 2), (1, 3, 4), (2, 4, 5))

# Which component of K carries each (row, column) of the curl, and with which
# sign. The diagonal is empty, as a curl has no diagonal.
_CURL_INDEX = ((None, 2, 1), (2, None, 0), (1, 0, None))
_CURL_SIGN = ((0.0, -1.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 1.0, 0.0))

# How many kernel evaluations to hold at once. The six-dimensional rule has
# order**6 points, so the offsets are taken in chunks that keep this bounded.
_CHUNK_ELEMENTS = 1 << 22


def green_n(separation: torch.Tensor, wavenumber: float) -> torch.Tensor:
    """Return the six distinct components of the double-curl dyadic Green kernel.

    Parameters
    ----------
    separation
        Shape ``(..., 3)``, the vector from source point to observation point.
    wavenumber
        Free-space wavenumber in rad/m.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 6)``, complex, in the order xx, xy, xz, yy, yz, zz.
    """
    x, y, z = separation[..., 0], separation[..., 1], separation[..., 2]
    r = torch.linalg.vector_norm(separation, dim=-1)
    k = wavenumber

    envelope = torch.exp(-1j * k * r) / (4.0 * torch.pi)
    r2, r3, r4, r5 = r**2, r**3, r**4, r**5

    def diagonal(component: torch.Tensor) -> torch.Tensor:
        return envelope * (
            3.0 * component**2 / r5
            + 3j * k * component**2 / r4
            - (1.0 + k**2 * component**2) / r3
            - 1j * k / r2
        )

    def off_diagonal(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        product = first * second
        return envelope * (
            3.0 * product / r5 + 3j * k * product / r4 - k**2 * product / r3
        )

    g_xx, g_yy, g_zz = diagonal(x), diagonal(y), diagonal(z)
    return torch.stack(
        (
            -g_yy - g_zz,
            off_diagonal(x, y),
            off_diagonal(x, z),
            -g_xx - g_zz,
            off_diagonal(y, z),
            -g_xx - g_yy,
        ),
        dim=-1,
    )


def green_k(separation: torch.Tensor, wavenumber: float) -> torch.Tensor:
    """Return the three components of the curl Green kernel.

    Parameters
    ----------
    separation
        Shape ``(..., 3)``, the vector from source point to observation point.
    wavenumber
        Free-space wavenumber in rad/m.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)``, complex, in the order x, y, z.
    """
    r = torch.linalg.vector_norm(separation, dim=-1)
    envelope = torch.exp(-1j * r * wavenumber) / (4.0 * torch.pi)
    radial = (1.0 / r**3 + 1j * wavenumber / r**2).unsqueeze(-1)
    return envelope.unsqueeze(-1) * (-separation) * radial


def volume_volume_n(
    offsets: torch.Tensor, resolution: float, wavenumber: float, order: int = 4
) -> torch.Tensor:
    """Integrate the N kernel over two voxels separated by each offset.

    Parameters
    ----------
    offsets
        Shape ``(n_offsets, 3)``, the vector from the source voxel's centre to
        the observation voxel's centre, in metres.
    resolution
        Voxel pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Gauss-Legendre points per axis; the rule has ``order ** 6`` points.

    Returns
    -------
    torch.Tensor
        Shape ``(n_offsets, 6)``, complex.

    Raises
    ------
    ValueError
        An offset is zero, where the kernel is not integrable by this rule.
    """
    return _volume_volume(offsets, resolution, wavenumber, order, green_n, 6)


def volume_volume_k(
    offsets: torch.Tensor, resolution: float, wavenumber: float, order: int = 4
) -> torch.Tensor:
    """Integrate the K kernel over two voxels separated by each offset.

    Parameters
    ----------
    offsets
        Shape ``(n_offsets, 3)``, in metres.
    resolution
        Voxel pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Gauss-Legendre points per axis.

    Returns
    -------
    torch.Tensor
        Shape ``(n_offsets, 3)``, complex.

    Raises
    ------
    ValueError
        An offset is zero.
    """
    return _volume_volume(offsets, resolution, wavenumber, order, green_k, 3)


def _volume_volume(offsets, resolution, wavenumber, order, kernel, n_components):
    """Run the six-dimensional rule for one kernel, in chunks over the offsets."""
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(f"offsets must have shape (n, 3), got {tuple(offsets.shape)}")
    if torch.any(torch.linalg.vector_norm(offsets, dim=-1) == 0):
        raise ValueError(
            "the volume-volume rule diverges at a zero offset; voxels that touch "
            "need the surface-surface treatment"
        )

    weights, nodes = gauss_legendre_1d(
        order, device=offsets.device, dtype=offsets.dtype
    )
    # Six nested axes: three over the observation voxel, three over the source.
    grids = torch.meshgrid(*([nodes] * 6), indexing="ij")
    weight = torch.ones_like(grids[0])
    for axis in range(6):
        shape = [1] * 6
        shape[axis] = order
        weight = weight * weights.reshape(shape)
    weight = weight.reshape(-1)

    # The kernel sees only the separation, so the two triples enter as their
    # difference: the observation node less the source node, scaled to the voxel.
    separation_nodes = (
        resolution
        / 2.0
        * torch.stack(
            (
                grids[0] - grids[3],
                grids[1] - grids[4],
                grids[2] - grids[5],
            ),
            dim=-1,
        ).reshape(-1, 3)
    )

    jacobian = (resolution / 2.0) ** 6
    n_points = weight.shape[0]
    chunk = max(1, _CHUNK_ELEMENTS // n_points)

    result = torch.zeros(
        (offsets.shape[0], n_components),
        device=offsets.device,
        dtype=torch.complex128,
    )
    for start in range(0, offsets.shape[0], chunk):
        block = offsets[start : start + chunk]
        separation = block.unsqueeze(1) + separation_nodes.unsqueeze(0)
        values = kernel(separation, wavenumber)
        result[start : start + chunk] = jacobian * torch.einsum(
            "q,bqc->bc", weight.to(values.dtype), values
        )
    return result


def apply_n(
    symbols: tuple[CirculantSymbol, ...], current: torch.Tensor
) -> torch.Tensor:
    """Apply the N operator to a current on the grid.

    Parameters
    ----------
    symbols
        Six compressed symbols from :func:`mariepy.tucker.circulant_tucker`, in
        the order xx, xy, xz, yy, yz, zz.
    current
        Shape ``(..., 3, n1, n2, n3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, n1, n2, n3)``.

    Raises
    ------
    ValueError
        The wrong number of symbols was given.
    """
    if len(symbols) != 6:
        raise ValueError(f"the N operator takes six symbols, got {len(symbols)}")
    return _apply(symbols, current, _DYADIC_INDEX, None)


def apply_k(
    symbols: tuple[CirculantSymbol, ...], current: torch.Tensor
) -> torch.Tensor:
    """Apply the K operator to a current on the grid.

    Parameters
    ----------
    symbols
        Three compressed symbols, in the order x, y, z.
    current
        Shape ``(..., 3, n1, n2, n3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, n1, n2, n3)``.

    Raises
    ------
    ValueError
        The wrong number of symbols was given.
    """
    if len(symbols) != 3:
        raise ValueError(f"the K operator takes three symbols, got {len(symbols)}")
    return _apply(symbols, current, _CURL_INDEX, _CURL_SIGN)


def _apply(symbols, current, index, sign):
    """Convolve each component of the current with the symbols that reach it."""
    if current.ndim < 4 or current.shape[-4] != 3:
        raise ValueError(
            f"a current must end in (3, n1, n2, n3), got {tuple(current.shape)}"
        )
    grid = current.shape[-3:]
    padded = symbols[0].shape

    transformed = torch.fft.fftn(current, s=padded, dim=(-3, -2, -1))
    expanded = [symbol.expand() for symbol in symbols]

    out = torch.zeros_like(transformed)
    for row in range(3):
        for column in range(3):
            which = index[row][column]
            if which is None:
                continue
            scale = 1.0 if sign is None else sign[row][column]
            out[..., row, :, :, :] = out[..., row, :, :, :] + scale * (
                expanded[which] * transformed[..., column, :, :, :]
            )

    field = torch.fft.ifftn(out, dim=(-3, -2, -1))
    return field[..., : grid[0], : grid[1], : grid[2]]


def apply_g(current: torch.Tensor, resolution: float) -> torch.Tensor:
    """Apply the Galerkin mass matrix of the piecewise-constant basis.

    Parameters
    ----------
    current
        Any shape.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        ``current`` scaled by the voxel volume.
    """
    return resolution**3 * current


def apply_inverse_g(current: torch.Tensor, resolution: float) -> torch.Tensor:
    """Invert :func:`apply_g`.

    Parameters
    ----------
    current
        Any shape.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        ``current`` divided by the voxel volume.
    """
    return current / resolution**3
