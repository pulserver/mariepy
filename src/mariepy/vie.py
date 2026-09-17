"""The volume integral operator on the body grid.

Ported from MARIE 3.0's ``src_integral_equations/src_vie/src_operators_vie/``:
``cubatures/{G_N,G_K,VV_Nop,VV_Kop,weights_points}.m`` and the
``surface_surface_*`` family for the kernel, and ``src_solver/src_mvp/mvp_vie/``
for the products.

A cell carries either the piecewise-constant basis or the piecewise-linear one.
The linear basis has four scalar functions per cell, ``1``, ``x/dx``, ``y/dx`` and
``z/dx`` about the cell's centre, times each of the three Cartesian directions:
twelve per cell, ordered direction-major as ``4 * direction + function``. Its
kernel is stored for the ten (test, source) pairs of :data:`PAIRS`; the other
six follow from them with the sign of :data:`PAIR_OF`.

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

import functools

import torch

from mariepy import _accelerators
from mariepy.quadrature import gauss_legendre_1d

__all__ = [
    "PAIRS",
    "PAIR_OF",
    "apply_g",
    "apply_inverse_g",
    "apply_k",
    "apply_n",
    "green_k",
    "green_n",
    "mass",
    "unit_responses",
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

# The (test, source) pairs of scalar basis functions the linear kernel stores,
# in MARIE's order: 0 is the constant, 1 to 3 the functions linear in x, y, z.
PAIRS = ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

# For every (test, source) pair, the stored pair that carries it and the sign
# it carries it with, as MARIE's ``mvp_N_pwl_tucker.m`` tabulates them. Swapping
# the constant with a linear function is odd; swapping two linear functions is
# even.
PAIR_OF = (
    ((0, +1), (4, +1), (5, +1), (6, +1)),
    ((4, -1), (1, +1), (7, +1), (8, +1)),
    ((5, -1), (7, +1), (2, +1), (9, +1)),
    ((6, -1), (8, +1), (9, +1), (3, +1)),
)

# The Galerkin mass of each scalar basis function over a cell, in units of the
# cell volume: the constant integrates to one, each linear function squared to
# one twelfth.
_MASS = (1.0, 1.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0)

# How many kernel evaluations to hold at once. A volume rule has thousands of
# points, so the offsets are taken in chunks that keep this bounded.
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
    offsets: torch.Tensor,
    resolution: float,
    wavenumber: float,
    order: int = 4,
    *,
    linear: bool = False,
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
        Gauss-Legendre points per axis of MARIE's product rule; the overlap
        rule used here reaches the same polynomial degree with
        ``(2 * order + 2) ** 3`` points.
    linear
        Weight the integral by each of the :data:`PAIRS` of linear basis
        functions, rather than by the constant alone.

    Returns
    -------
    torch.Tensor
        Shape ``(n_offsets, 6)``, or ``(n_offsets, 10, 6)`` when ``linear``,
        complex.

    Raises
    ------
    ValueError
        An offset is zero, where the kernel is not integrable by this rule.
    """
    return _volume_volume(offsets, resolution, wavenumber, order, green_n, 6, linear)


def volume_volume_k(
    offsets: torch.Tensor,
    resolution: float,
    wavenumber: float,
    order: int = 4,
    *,
    linear: bool = False,
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
    linear
        As in :func:`volume_volume_n`.

    Returns
    -------
    torch.Tensor
        Shape ``(n_offsets, 3)``, or ``(n_offsets, 10, 3)`` when ``linear``,
        complex.

    Raises
    ------
    ValueError
        An offset is zero.
    """
    return _volume_volume(offsets, resolution, wavenumber, order, green_k, 3, linear)


def _overlap_rule(order, device, dtype):
    """Return the one-dimensional rule for the difference of two cell coordinates.

    With ``u`` and ``v`` the observer's and the source's coordinate over one
    cell pitch, each in ``[-1/2, 1/2]``, a kernel of ``d = u - v`` integrates
    over both as ``integral h(d) f(d) dd``, where ``h`` is the length of the
    overlap weighted by the basis functions: ``1``, ``u``, ``v`` or ``u v``.
    Each weight is a polynomial of degree up to three on either side of
    ``d = 0``. A Gauss rule of ``order + 1`` points on each half integrates
    it with a polynomial ``f`` to the degree the ``order``-point product rule
    reaches for the same weight.

    Returns
    -------
    nodes : torch.Tensor
        Shape ``(2 * order + 2,)``, the values of ``d``.
    weights : torch.Tensor
        Shape ``(4, 2 * order + 2)``: the rule for the weights ``1``, ``u``,
        ``v`` and ``u v``, in that order.
    """
    base_weights, base_nodes = gauss_legendre_1d(order + 1, device=device, dtype=dtype)
    half = (base_nodes + 1.0) / 2.0  # on [0, 1]
    nodes = torch.cat([-half.flip(0), half])
    gauss = torch.cat([base_weights.flip(0), base_weights]) / 2.0
    positive = nodes >= 0
    low = torch.where(positive, nodes - 0.5, torch.full_like(nodes, -0.5))
    high = torch.where(positive, torch.full_like(nodes, 0.5), nodes + 0.5)
    constant = high - low
    first = (high**2 - low**2) / 2.0
    second = first - nodes * constant
    both = (high**3 - low**3) / 3.0 - nodes * first
    return nodes, gauss * torch.stack([constant, first, second, both])


def _volume_volume(
    offsets, resolution, wavenumber, order, kernel, n_components, linear=False
):
    """Integrate a kernel over two cells, reduced to the difference of their points.

    The kernel depends on the separation alone, so the six-dimensional
    integral MARIE takes with an ``order ** 6`` product rule is, exactly, a
    three-dimensional one weighted by the cells' overlap; this takes it with
    ``(2 order + 2) ** 3`` points and the same polynomial exactness.
    """
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(f"offsets must have shape (n, 3), got {tuple(offsets.shape)}")
    if torch.any(torch.linalg.vector_norm(offsets, dim=-1) == 0):
        raise ValueError(
            "the volume-volume rule diverges at a zero offset; voxels that touch "
            "need the surface-surface treatment"
        )

    nodes, weights = _overlap_rule(order, offsets.device, offsets.dtype)
    grid = torch.cartesian_prod(nodes, nodes, nodes)  # (m, 3)
    separation_nodes = resolution * grid

    def along(axis_weights):
        """Combine one weight per axis into the three-dimensional rule."""
        return (
            axis_weights[0][:, None, None]
            * axis_weights[1][None, :, None]
            * axis_weights[2][None, None, :]
        ).reshape(-1)

    pairs = PAIRS if linear else ((0, 0),)
    rows = []
    for test, basis in pairs:
        per_axis = []
        for axis in range(3):
            slot = (1 if test == axis + 1 else 0) + (2 if basis == axis + 1 else 0)
            per_axis.append(weights[slot])
        rows.append(along(per_axis))
    pair_weight = torch.stack(rows)

    jacobian = resolution**6
    if offsets.device.type == "cpu" and kernel in (green_n, green_k):
        compiled = _accelerators.require("volume_volume")
        values = compiled(
            offsets.detach().numpy(),
            separation_nodes.detach().numpy(),
            pair_weight.detach().numpy(),
            wavenumber,
            kernel is green_n,
        )
        result = jacobian * torch.from_numpy(values)
        return result if linear else result[:, 0]
    return _volume_volume_torch(
        offsets,
        separation_nodes,
        pair_weight,
        jacobian,
        wavenumber,
        kernel,
        n_components,
        linear,
    )


def _volume_volume_torch(
    offsets,
    separation_nodes,
    pair_weight,
    jacobian,
    wavenumber,
    kernel,
    n_components,
    linear,
):
    """Contract the kernel with the rule in torch, on whichever device holds it.

    The compiled rule in ``_ext`` is checked against this.
    """
    chunk = max(1, _CHUNK_ELEMENTS // separation_nodes.shape[0])
    result = torch.zeros(
        (offsets.shape[0], pair_weight.shape[0], n_components),
        device=offsets.device,
        dtype=torch.complex128,
    )
    for start in range(0, offsets.shape[0], chunk):
        block = offsets[start : start + chunk]
        separation = block.unsqueeze(1) + separation_nodes.unsqueeze(0)
        values = kernel(separation, wavenumber)
        result[start : start + chunk] = jacobian * torch.einsum(
            "pq,bqc->bpc", pair_weight.to(values.dtype), values
        )
    return result if linear else result[:, 0]


def _volume_volume_product(
    offsets, resolution, wavenumber, order, kernel, n_components, linear=False
):
    """Run the six-dimensional product rule, as MARIE does; kept as the reference."""
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

    if linear:
        # Each scalar basis function is its local coordinate over the pitch,
        # half the Gauss node, so the pair weight is a product of node halves.
        observer = [torch.ones_like(grids[0])] + [grids[a] / 2.0 for a in range(3)]
        source = [torch.ones_like(grids[0])] + [grids[a] / 2.0 for a in range(3, 6)]
        pair_weight = torch.stack(
            [
                (observer[test] * source[basis]).reshape(-1) * weight
                for test, basis in PAIRS
            ]
        )
    else:
        pair_weight = weight.unsqueeze(0)

    result = torch.zeros(
        (offsets.shape[0], pair_weight.shape[0], n_components),
        device=offsets.device,
        dtype=torch.complex128,
    )
    for start in range(0, offsets.shape[0], chunk):
        block = offsets[start : start + chunk]
        separation = block.unsqueeze(1) + separation_nodes.unsqueeze(0)
        values = kernel(separation, wavenumber)
        result[start : start + chunk] = jacobian * torch.einsum(
            "pq,bqc->bpc", pair_weight.to(values.dtype), values
        )
    return result if linear else result[:, 0]


def apply_n(symbols, current: torch.Tensor) -> torch.Tensor:
    """Apply the N operator to a current on the grid.

    Parameters
    ----------
    symbols
        From :func:`mariepy.tucker.circulant_tucker`: six symbols in the order
        xx, xy, xz, yy, yz, zz for the constant basis, or ten such sextuples,
        one per pair of :data:`PAIRS`, for the linear basis.
    current
        Shape ``(..., 3, n1, n2, n3)`` for the constant basis, or
        ``(..., 12, n1, n2, n3)`` for the linear basis.

    Returns
    -------
    torch.Tensor
        The same shape as ``current``.

    Raises
    ------
    ValueError
        The symbols do not match the basis the current is written in.
    """
    if _is_linear(current):
        _check_linear_symbols(symbols, 6)
        return _apply_linear(symbols, current, _DYADIC_INDEX, None)
    if len(symbols) != 6:
        raise ValueError(f"the N operator takes six symbols, got {len(symbols)}")
    return _apply(symbols, current, _DYADIC_INDEX, None)


def apply_k(symbols, current: torch.Tensor) -> torch.Tensor:
    """Apply the K operator to a current on the grid.

    Parameters
    ----------
    symbols
        Three symbols in the order x, y, z, or ten such triples for the linear
        basis.
    current
        Shape ``(..., 3, n1, n2, n3)`` or ``(..., 12, n1, n2, n3)``.

    Returns
    -------
    torch.Tensor
        The same shape as ``current``.

    Raises
    ------
    ValueError
        The symbols do not match the basis the current is written in.
    """
    if _is_linear(current):
        _check_linear_symbols(symbols, 3)
        return _apply_linear(symbols, current, _CURL_INDEX, _CURL_SIGN)
    if len(symbols) != 3:
        raise ValueError(f"the K operator takes three symbols, got {len(symbols)}")
    return _apply(symbols, current, _CURL_INDEX, _CURL_SIGN)


def _is_linear(current: torch.Tensor) -> bool:
    """Say whether a current is written in the linear basis."""
    if current.ndim < 4 or current.shape[-4] not in (3, 12):
        raise ValueError(
            "a current must end in (3, n1, n2, n3) or (12, n1, n2, n3), got "
            f"{tuple(current.shape)}"
        )
    return current.shape[-4] == 12


def _check_linear_symbols(symbols, n_components: int) -> None:
    """Raise unless the symbols are one tuple per stored pair of linear functions."""
    if len(symbols) != len(PAIRS) or any(len(row) != n_components for row in symbols):
        raise ValueError(
            f"the linear basis takes {len(PAIRS)} tuples of {n_components} symbols"
        )


def _apply_linear(symbols, current, index, sign):
    """Convolve a linear-basis current, expanding one stored symbol at a time.

    Ported from MARIE's ``mvp_N_pwl_tucker.m`` and ``mvp_K_pwl_tucker.m``. Each
    output entry ``4 * p + l`` gathers every input entry ``4 * q + l'`` through
    the stored pair and component that carry ``(l, l')`` and ``(p, q)``.
    """
    if _compiled(current, symbols[0][0].shape):
        return _apply_compiled(*_product_terms(symbols, index, sign, True), current)

    grid = current.shape[-3:]
    padded = symbols[0][0].shape
    transformed = torch.fft.fftn(current, s=padded, dim=(-3, -2, -1))
    out = torch.zeros_like(transformed)

    for (pair, which), entries in _linear_uses(index, sign).items():
        expanded = symbols[pair][which].expand()
        for row, column, scale in entries:
            # Fused, because the padded grid is large and this runs 144 times:
            # writing it as a sum would allocate two of them per term.
            out[..., row, :, :, :].addcmul_(
                expanded, transformed[..., column, :, :, :], value=scale
            )

    field = torch.fft.ifftn(out, dim=(-3, -2, -1))
    return field[..., : grid[0], : grid[1], : grid[2]]


def _apply(symbols, current, index, sign):
    """Convolve each component of the current with the symbols that reach it."""
    if current.ndim < 4 or current.shape[-4] != 3:
        raise ValueError(
            f"a current must end in (3, n1, n2, n3), got {tuple(current.shape)}"
        )
    if _compiled(current, symbols[0].shape):
        return _apply_compiled(*_product_terms(symbols, index, sign, False), current)

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
            out[..., row, :, :, :].addcmul_(
                expanded[which], transformed[..., column, :, :, :], value=scale
            )

    field = torch.fft.ifftn(out, dim=(-3, -2, -1))
    return field[..., : grid[0], : grid[1], : grid[2]]


@functools.cache
def _linear_uses(index, sign):
    """Group the 144 terms of a linear-basis product by the symbol they read.

    Returns a mapping from ``(pair, component)`` to the ``(row, column, sign)``
    entries that symbol weighs.
    """
    uses = {}
    for p in range(3):
        for q in range(3):
            which = index[p][q]
            if which is None:
                continue
            component_sign = 1.0 if sign is None else sign[p][q]
            for test in range(4):
                for basis in range(4):
                    pair, pair_sign = PAIR_OF[test][basis]
                    uses.setdefault((pair, which), []).append(
                        (4 * p + test, 4 * q + basis, component_sign * pair_sign)
                    )
    return uses


def _product_terms(symbols, index, sign, linear: bool):
    """List a product's symbols flat, and its terms as (row, column, symbol, sign)."""
    if not linear:
        terms = [
            (
                row,
                column,
                index[row][column],
                1.0 if sign is None else sign[row][column],
            )
            for row in range(3)
            for column in range(3)
            if index[row][column] is not None
        ]
        return list(symbols), terms
    per_pair = len(symbols[0])
    terms = [
        (row, column, pair * per_pair + which, scale)
        for (pair, which), entries in _linear_uses(index, sign).items()
        for row, column, scale in entries
    ]
    return [symbol for row in symbols for symbol in row], terms


def unit_responses(
    symbols, offsets: torch.Tensor, shape, *, curl: bool = False
) -> torch.Tensor:
    """Apply N, or K, to a unit current at each of several cells.

    The product is a convolution on the extended grid, so the field of a unit
    current is the inverse transform of a symbol, shifted to the cell. That
    gives every response from one transform per symbol.

    Parameters
    ----------
    symbols
        As :func:`apply_n` takes them, or as :func:`apply_k` does when
        ``curl``.
    offsets
        Shape ``(m, 3)``, integer: the cells carrying the unit currents.
    shape
        The grid the currents and the fields are on.
    curl
        Apply K rather than N.

    Returns
    -------
    torch.Tensor
        Shape ``(c * m, c, n1 * n2 * n3)``, with ``c`` 3 or 12 by the symbols'
        basis: entry ``(k * m + i, l)`` is component ``l`` of the field of a
        unit current in component ``k`` at ``offsets[i]``, as :func:`apply_n`
        or :func:`apply_k` gives it.
    """
    linear = not hasattr(symbols[0], "expand")
    index, sign = (_CURL_INDEX, _CURL_SIGN) if curl else (_DYADIC_INDEX, None)
    flat, terms = _product_terms(symbols, index, sign, linear)
    n_components = 12 if linear else 3
    padded = flat[0].shape
    device = offsets.device
    kernels = torch.stack(
        [torch.fft.ifftn(symbol.expand()).reshape(-1) for symbol in flat]
    )
    axes = [torch.arange(extent, device=device) for extent in shape]
    cells = torch.cartesian_prod(*axes)
    lengths = torch.tensor(padded, device=device)
    shifted = (cells[None, :, :] - offsets[:, None, :]) % lengths
    where = (shifted[..., 0] * padded[1] + shifted[..., 1]) * padded[2] + shifted[
        ..., 2
    ]

    n_offsets = offsets.shape[0]
    out = torch.zeros(
        (n_components, n_offsets, n_components, cells.shape[0]),
        dtype=kernels.dtype,
        device=device,
    )
    for row, column, which, scale in terms:
        out[column, :, row, :] += scale * kernels[which][where]
    return out.reshape(n_components * n_offsets, n_components, -1)


# Below this many cells on the extended grid, expanding the symbols costs less
# than starting the kernel's threads, and torch's batched product is faster.
_COMPILED_MIN_CELLS = 64**3


def _compiled(current: torch.Tensor, padded) -> bool:
    """Say whether a product runs in the compiled kernel.

    It does for complex128 on the CPU, on an extended grid large enough that
    the expanded symbols, not the threads, set the cost.
    """
    return (
        current.device.type == "cpu"
        and current.dtype == torch.complex128
        and padded[0] * padded[1] * padded[2] >= _COMPILED_MIN_CELLS
    )


def _apply_compiled(symbols, terms, current: torch.Tensor) -> torch.Tensor:
    """Convolve a current through :func:`mariepy._ext.multiply_symbols`.

    The multiply runs in place on one buffer of the extended grid, and each
    output component is transformed back and cropped on its own, so the product
    never holds more than that buffer and one extra component of it.
    """
    multiply = _accelerators.require("multiply_symbols")
    grid = current.shape[-3:]
    padded = symbols[0].shape
    batch = current.shape[:-4]
    n_components = current.shape[-4]
    flat = current.reshape(-1, n_components, *grid)
    result = torch.empty_like(flat)
    cores = [symbol.core.resolve_conj().contiguous().numpy() for symbol in symbols]
    factors = [
        factor.resolve_conj().contiguous().numpy()
        for symbol in symbols
        for factor in symbol.factors
    ]
    rows, columns, which, scales = (list(column) for column in zip(*terms, strict=True))
    buffer = torch.empty((n_components, *padded), dtype=current.dtype)
    for item in range(flat.shape[0]):
        for component in range(n_components):
            buffer[component] = torch.fft.fftn(
                flat[item, component], s=padded, dim=(-3, -2, -1)
            )
        multiply(
            buffer.numpy(),
            cores,
            factors,
            rows,
            columns,
            which,
            [float(scale) for scale in scales],
            torch.get_num_threads(),
        )
        for component in range(n_components):
            result[item, component] = torch.fft.ifftn(buffer[component])[
                : grid[0], : grid[1], : grid[2]
            ]
    return result.reshape(*batch, n_components, *grid)


def mass(n_components: int, resolution: float) -> torch.Tensor:
    """Return the diagonal Galerkin mass matrix of one cell's basis functions.

    Ported from MARIE's ``mvp_G_pwc.m`` and ``mvp_G_pwl.m``.

    Parameters
    ----------
    n_components
        3 for the constant basis, 12 for the linear one.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        Shape ``(n_components,)``, float64.
    """
    per_direction = _MASS if n_components == 12 else _MASS[:1]
    return resolution**3 * torch.tensor(per_direction * 3, dtype=torch.float64)


def apply_g(current: torch.Tensor, resolution: float) -> torch.Tensor:
    """Apply the Galerkin mass matrix of the basis a current is written in.

    Parameters
    ----------
    current
        Shape ``(..., 3, n1, n2, n3)`` or ``(..., 12, n1, n2, n3)``.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        ``current`` with each basis function scaled by its mass.
    """
    weights = mass(current.shape[-4], resolution).to(current.device)
    return current * weights.to(current.dtype)[:, None, None, None]


def apply_inverse_g(current: torch.Tensor, resolution: float) -> torch.Tensor:
    """Invert :func:`apply_g`.

    Parameters
    ----------
    current
        Shape ``(..., 3, n1, n2, n3)`` or ``(..., 12, n1, n2, n3)``.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        ``current`` with each basis function divided by its mass.
    """
    weights = mass(current.shape[-4], resolution).to(current.device)
    return current / weights.to(current.dtype)[:, None, None, None]


# The eight corners of a cell of side `resolution` centred on the origin, in
# MARIE's order, and the four corners of each face as a cycle around it. Faces
# run -x, +x, -y, +y, -z, +z.
_CORNERS = (
    (-1, -1, -1),
    (+1, -1, -1),
    (+1, +1, -1),
    (-1, +1, -1),
    (-1, -1, +1),
    (+1, -1, +1),
    (+1, +1, +1),
    (-1, +1, +1),
)
_FACE_CYCLES = (
    (0, 3, 7, 4),
    (1, 2, 6, 5),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 1, 2, 3),
    (4, 5, 6, 7),
)
_FACE_NORMALS = (
    (-1.0, 0.0, 0.0),
    (+1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, +1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, +1.0),
)

# The six distinct components of the symmetric dyad, as the pair of Cartesian
# directions each one tests.
_COMPONENT_AXES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def face_cycle(face: int, centre, resolution: float):
    """Return the four corners of one face of a cell, as a cycle around it.

    Parameters
    ----------
    face
        Which face, in the order -x, +x, -y, +y, -z, +z.
    centre
        The cell's centre, a sequence of three floats.
    resolution
        Cell pitch in metres.

    Returns
    -------
    list of tuple
        Four corners, each a triple of floats.
    """
    half = resolution / 2.0
    return [
        tuple(centre[axis] + half * _CORNERS[corner][axis] for axis in range(3))
        for corner in _FACE_CYCLES[face]
    ]


def face_adjacency(
    offset_cells, face_observer: int, face_source: int, resolution: float
):
    """Return how two faces of two cells touch, and the points DIRECTFN wants.

    The singular routines depend on which faces touch, on the adjacency, and on
    the shared feature sitting in the slot each routine reads it from. They do
    not depend on the winding of either square, nor on the order within a shared
    edge, so the ordering is derived here rather than tabulated.

    Parameters
    ----------
    offset_cells
        Whole-cell offset from the source cell to the observation cell.
    face_observer, face_source
        Faces of the observation and source cells.
    resolution
        Cell pitch in metres.

    Returns
    -------
    kind : str or None
        ``"self"``, ``"edge"``, ``"vertex"``, or ``None`` when the faces do not
        touch.
    points : list of tuple or None
        The four, six or seven ordered points, or ``None``.
    """
    source = face_cycle(face_source, (0.0, 0.0, 0.0), resolution)
    observer = face_cycle(
        face_observer,
        tuple(resolution * value for value in offset_cells),
        resolution,
    )

    tolerance = 1e-9 * resolution

    def same(first, second):
        return all(abs(a - b) <= tolerance for a, b in zip(first, second, strict=True))

    shared = [
        index
        for index, point in enumerate(source)
        if any(same(point, other) for other in observer)
    ]

    if len(shared) == 4:
        return "self", list(source)
    if len(shared) == 2:
        rotated = _rotate_to(source, shared, (2, 3))
        following = _cycle_from(observer, rotated[3], rotated[2])
        return "edge", [*rotated, following[3], following[2]]
    if len(shared) == 1:
        rotated = _rotate_to(source, shared, (2,))
        following = _cycle_from(observer, rotated[2], None)
        return "vertex", [*rotated, *following[1:]]
    return None, None


def _rotate_to(cycle, shared, slots):
    """Rotate a four-cycle, either way round, until the shared points land."""
    for direction in (cycle, cycle[::-1]):
        for turn in range(4):
            candidate = direction[turn:] + direction[:turn]
            landed = {candidate.index(cycle[index]) for index in shared}
            if landed == set(slots):
                return candidate
    raise ValueError("the shared points do not sit on one face of the cell")


def _cycle_from(cycle, first, second):
    """Rotate a four-cycle to start at one point, optionally reaching a second."""
    start = next(
        index
        for index, point in enumerate(cycle)
        if all(abs(a - b) <= 1e-12 for a, b in zip(point, first, strict=True))
    )
    forward = cycle[start:] + cycle[:start]
    if second is None:
        return forward
    if all(abs(a - b) <= 1e-12 for a, b in zip(forward[1], second, strict=True)):
        return forward
    return [forward[0], *forward[1:][::-1]]


def surface_surface_coefficient(
    face_observer: int, face_source: int, component: int
) -> float:
    """Return the constant the surface-surface integral of one face pair carries.

    Ported from MARIE's ``coefficients_Nop.m``. For the piecewise-constant basis
    only its first kernel survives: the other three each carry the gradient of
    the scalar basis term, which is zero when the basis is constant.

    Parameters
    ----------
    face_observer, face_source
        Faces of the observation and source cells.
    component
        Which of the six components of the symmetric dyad, in the order xx, xy,
        xz, yy, yz, zz.

    Returns
    -------
    float
        ``(n x e_p) . (n' x e_q)`` for that face pair and component.
    """
    import numpy as _np

    p, q = _COMPONENT_AXES[component]
    normal = _np.array(_FACE_NORMALS[face_observer])
    normal_source = _np.array(_FACE_NORMALS[face_source])
    return float(
        _np.dot(
            _np.cross(normal, _np.eye(3)[p]),
            _np.cross(normal_source, _np.eye(3)[q]),
        )
    )


def surface_surface_n(
    offset_cells,
    resolution: float,
    wavenumber: float,
    order: int = 15,
    *,
    linear: bool = False,
) -> torch.Tensor:
    """Integrate the N kernel over two cells by their faces.

    Voxels that touch make the volume-volume rule diverge. MARIE reduces the
    double-curl dyadic to integrals over the twelve faces of the two cells,
    where the remaining singularity is weak enough for DIRECTFN to handle
    exactly; face pairs that do not touch take a plain four-dimensional rule.
    Ported from ``surface_surface_kernels_Nop.m``, ``surface_surface_coeff_Nop.m``,
    ``coefficients_Nop.m`` and ``kernels_Nop.m``.

    Parameters
    ----------
    offset_cells
        Whole-cell offset from the source cell to the observation cell.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Points per axis, for the four-dimensional rule and for DIRECTFN alike.
    linear
        Return every pair of :data:`PAIRS` rather than the constant pair alone.

    Returns
    -------
    torch.Tensor
        Shape ``(6,)``, or ``(10, 6)`` when ``linear``, complex, in the order
        xx, xy, xz, yy, yz, zz.
    """
    return _surface_surface(
        offset_cells, resolution, wavenumber, order, linear, _N_REDUCTION
    )


def _surface_surface(
    offset_cells, resolution, wavenumber, order, linear, reduction, every_pair=False
):
    """Sum the four reduced face kernels of one operator over the 36 face pairs.

    Each reduced kernel is an integral times a constant. The integral depends on
    the test function, the source function, both or neither, as ``reduction``
    records; the constant carries whatever dependence the integral does not.
    """
    import numpy as _np

    from mariepy import _accelerators

    directfn = _accelerators.require(module="mariepy._directfn")
    weights, nodes = gauss_legendre_1d(order)
    weights, nodes = weights.numpy(), nodes.numpy()

    n_functions = 4 if linear else 1
    n_components = len(reduction.axes)
    centre_observer = _np.array([resolution * value for value in offset_cells])
    centre_source = _np.zeros(3)
    total = _np.zeros((n_functions, n_functions, n_components), dtype=complex)
    singular = []

    for face_observer in range(6):
        for face_source in range(6):
            normal_observer = _np.array(_FACE_NORMALS[face_observer])
            normal_source = _np.array(_FACE_NORMALS[face_source])
            coefficients = _face_coefficients(
                reduction.coefficients,
                face_observer,
                face_source,
                resolution,
                wavenumber,
                n_functions,
            )
            if not _np.any(coefficients):
                continue

            kind, points = face_adjacency(
                offset_cells, face_observer, face_source, resolution
            )
            if kind is None:
                observer, source, weight = _face_pair_points(
                    centre_observer + normal_observer * resolution / 2.0,
                    centre_source + normal_source * resolution / 2.0,
                    face_observer // 2,
                    face_source // 2,
                    resolution,
                    nodes,
                    weights,
                )
            else:
                vertices = _np.array(points, dtype=float)
                routine, observer_centre = _singular_call(kind, vertices, directfn)

            for term in range(4):
                if not _np.any(coefficients[term]):
                    continue
                uses_test, uses_source = reduction.depends[term]
                for test in range(n_functions if uses_test else 1):
                    for basis in range(n_functions if uses_source else 1):
                        if kind is None:
                            integrand = reduction.integrand(
                                term,
                                observer,
                                source,
                                centre_observer,
                                centre_source,
                                test,
                                basis,
                                normal_observer,
                                normal_source,
                                resolution,
                                wavenumber,
                            )
                            value = (resolution / 2.0) ** 4 * _np.sum(
                                weight * integrand
                            )
                        tests = [test] if uses_test else range(n_functions)
                        sources = [basis] if uses_source else range(n_functions)
                        if kind is not None:
                            singular.append(
                                (
                                    routine,
                                    (
                                        vertices,
                                        (vertices[0] + vertices[2]) / 2.0,
                                        observer_centre,
                                        normal_source,
                                        normal_observer,
                                        wavenumber,
                                        resolution,
                                        order,
                                        reduction.directfn_type[term],
                                        test,
                                        basis,
                                    ),
                                    coefficients[term],
                                    tests,
                                    sources,
                                )
                            )
                            continue
                        for row in tests:
                            for column in sources:
                                total[row, column] += (
                                    coefficients[term][:, row, column] * value
                                )

    # The singular integrals are independent and release the GIL; take them
    # on every core, and add them in the order they were set up.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as pool:
        values = list(pool.map(lambda job: job[0](*job[1]), singular))
    for (_, _, coefficient, tests, sources), value in zip(
        singular, values, strict=True
    ):
        for row in tests:
            for column in sources:
                total[row, column] += coefficient[:, row, column] * value

    if every_pair:
        return torch.tensor(total, dtype=torch.complex128)
    stored = (
        _np.stack([total[row, column] for row, column in PAIRS])
        if linear
        else total[0, 0]
    )
    return torch.tensor(stored, dtype=torch.complex128)


def _step(function: int, resolution: float):
    """Return the gradient of one scalar basis function: zero for the constant."""
    import numpy as _np

    gradient = _np.zeros(3)
    if function:
        gradient[function - 1] = 1.0 / resolution
    return gradient


def _n_coefficients(normal, normal_source, resolution, wavenumber, n_functions):
    """Return MARIE's ``coefficients_Nop.m`` for one face pair, every term.

    The result is four arrays of shape ``(6, n_functions, n_functions)``, one
    per reduced kernel, indexed by component, test function and source function.
    """
    import numpy as _np

    del wavenumber
    unit = _np.eye(3)
    out = _np.zeros((4, 6, n_functions, n_functions), dtype=complex)
    for component, (p, q) in enumerate(_COMPONENT_AXES):
        e_p, e_q = unit[p], unit[q]
        out[0, component] = _np.dot(
            _np.cross(normal, e_p), _np.cross(normal_source, e_q)
        )
        for test in range(n_functions):
            h = _step(test, resolution)
            out[1, component, test, :] = _np.dot(
                _np.cross(_np.cross(h, e_p), e_q), normal
            )
            for basis in range(n_functions):
                hp = _step(basis, resolution)
                crossed = _np.cross(h, e_p)
                out[3, component, test, basis] = sum(
                    _np.dot(_np.cross(crossed, unit[a]), normal)
                    * _np.dot(hp * e_q[a], normal_source)
                    for a in range(3)
                )
        for basis in range(n_functions):
            hp = _step(basis, resolution)
            out[2, component, :, basis] = _np.dot(
                _np.cross(e_p, _np.cross(hp, e_q)), normal
            )
    return out


@functools.lru_cache(maxsize=512)
def _face_coefficients(
    coefficients, face_observer, face_source, resolution, wavenumber, n_functions
):
    """Return one reduction's constants for a face pair; they depend on the normals only."""
    import numpy as _np

    return coefficients(
        _np.array(_FACE_NORMALS[face_observer]),
        _np.array(_FACE_NORMALS[face_source]),
        resolution,
        wavenumber,
        n_functions,
    )


# The integrands of one face pair share its Green functions. The entry holds the
# point arrays, so an identity match cannot come from a recycled object.
_last_greens: list = [None]


def _face_greens(observer, source, wavenumber):
    """Return the separation, its length, and the dynamic and static Green functions."""
    import numpy as _np

    cached = _last_greens[0]
    if (
        cached is not None
        and cached[0] is observer
        and cached[1] is source
        and cached[2] == wavenumber
    ):
        return cached[3]
    separation = observer - source
    distance = _np.sqrt(_np.einsum("ij,ij->i", separation, separation))
    green = _np.exp(-1j * wavenumber * distance) / (4.0 * _np.pi * distance)
    static = 1.0 / (4.0 * _np.pi * distance)
    result = separation, distance, green, static
    _last_greens[0] = (observer, source, wavenumber, result)
    return result


def _radial_field(separation, distance, green, static, wavenumber):
    """Return the vector ``F`` of MARIE's reduced kernels, one row per point."""
    radial = (
        -1j * wavenumber * green / distance - green / distance**2 + static / distance**2
    ) / (1j * wavenumber) ** 2
    return separation * radial[:, None]


def _scalar_function(points, centre, function, resolution):
    """Evaluate one scalar basis function about a cell centre."""
    import numpy as _np

    if function == 0:
        return _np.ones(points.shape[0])
    axis = function - 1
    return (points[:, axis] - centre[axis]) / resolution


def _n_integrand(
    term,
    observer,
    source,
    centre_observer,
    centre_source,
    test,
    basis,
    normal,
    normal_source,
    resolution,
    wavenumber,
):
    """Return MARIE's ``kernels_Nop.m`` at every point of the face-pair rule."""
    del normal
    separation, distance, green, static = _face_greens(observer, source, wavenumber)
    if term == 0:
        return (
            _scalar_function(observer, centre_observer, test, resolution)
            * _scalar_function(source, centre_source, basis, resolution)
            * green
        )
    if term == 3:
        return (green - static) / (1j * wavenumber) ** 2
    projected = (
        _radial_field(separation, distance, green, static, wavenumber) @ normal_source
    )
    if term == 1:
        return _scalar_function(source, centre_source, basis, resolution) * projected
    return _scalar_function(observer, centre_observer, test, resolution) * projected


def _face_pair_points(
    centre_observer,
    centre_source,
    axis_observer,
    axis_source,
    resolution,
    nodes,
    weights,
):
    """Return the points of the four-dimensional rule on each face, and its weights."""
    import numpy as _np

    def place(centre, axis, first, second):
        free = [value for value in range(3) if value != axis]
        point = _np.empty((first.size, 3))
        point[:, axis] = centre[axis]
        point[:, free[0]] = centre[free[0]] + resolution / 2.0 * first
        point[:, free[1]] = centre[free[1]] + resolution / 2.0 * second
        return point

    a, b, c, d = (
        grid.reshape(-1) for grid in _np.meshgrid(*([nodes] * 4), indexing="ij")
    )
    weight = _np.ones(a.size)
    for grid in _np.meshgrid(*([weights] * 4), indexing="ij"):
        weight = weight * grid.reshape(-1)

    return (
        place(centre_observer, axis_observer, a, b),
        place(centre_source, axis_source, c, d),
        weight,
    )


def _singular_call(kind, vertices, directfn):
    """Pick the DIRECTFN routine and the observation centre for one adjacency.

    The centres are the ones MARIE forms in ``singular_ST_lin.m``,
    ``singular_EA_lin.m`` and ``singular_VA_lin.m``: each is the midpoint of a
    diagonal of its own square, which the ordering puts in a different slot for
    each adjacency.
    """
    if kind == "self":
        return directfn.voxel_self, (vertices[0] + vertices[2]) / 2.0
    if kind == "edge":
        return directfn.voxel_edge, (vertices[3] + vertices[5]) / 2.0
    return directfn.voxel_vertex, (vertices[2] + vertices[5]) / 2.0


def kernel_n(
    shape: tuple[int, int, int],
    resolution: float,
    wavenumber: float,
    *,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
    linear: bool = False,
) -> torch.Tensor:
    """Assemble the N kernel at every offset of a grid.

    Three regimes, as in MARIE's ``assembly_N.m``: a coarse volume-volume rule
    far out, a finer one over the first ``medium_order`` offsets along each
    axis, and the surface-surface reduction over the two-by-two-by-two block
    where the cells touch and the volume rule diverges.

    Parameters
    ----------
    shape
        Grid shape; the kernel is stored at every non-negative offset.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    far_order, medium_order
        Points per axis for the two volume-volume passes.
    near_order
        Points per axis for the surface-surface pass.
    linear
        Assemble every pair of :data:`PAIRS` of the linear basis.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3, 6)``, or ``(n1, n2, n3, 10, 6)`` when ``linear``,
        complex, in the order xx, xy, xz, yy, yz, zz.
    """
    return _assemble(
        shape,
        resolution,
        wavenumber,
        far_order,
        medium_order,
        near_order,
        volume_volume_n,
        surface_surface_n,
        6,
        linear,
    )


def _assemble(
    shape,
    resolution,
    wavenumber,
    far_order,
    medium_order,
    near_order,
    by_volume,
    by_surface,
    n_components,
    linear=False,
):
    """Fill every offset of a grid from the regime that is valid there."""
    import itertools as _itertools

    trailing = (len(PAIRS), n_components) if linear else (n_components,)
    kernel = torch.zeros((*shape, *trailing), dtype=torch.complex128)
    cells = torch.stack(
        torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij"), dim=-1
    ).reshape(-1, 3)
    touching = (cells < 2).all(dim=1)
    offsets = cells[~touching]

    if offsets.numel():
        far = by_volume(
            resolution * offsets.to(torch.float64),
            resolution,
            wavenumber,
            far_order,
            linear=linear,
        )
        kernel[offsets[:, 0], offsets[:, 1], offsets[:, 2]] = far

        medium = offsets[(offsets < medium_order).all(dim=1)]
        if medium.numel():
            refined = by_volume(
                resolution * medium.to(torch.float64),
                resolution,
                wavenumber,
                medium_order,
                linear=linear,
            )
            kernel[medium[:, 0], medium[:, 1], medium[:, 2]] = refined

    near = tuple(_itertools.product(*(range(min(2, n)) for n in shape)))
    for cell in near:
        kernel[cell] = by_surface(
            cell, resolution, wavenumber, near_order, linear=linear
        )

    return kernel


# The three components of the curl, as the pair of Cartesian directions each one
# tests. Anti-symmetry leaves three distinct interactions rather than six.
_CURL_COMPONENT_AXES = ((2, 1), (0, 2), (1, 0))


def curl_surface_coefficient(face_source: int, component: int) -> float:
    """Return the constant the first reduced curl kernel of one face pair carries.

    Ported from MARIE's ``coefficients_Kop.m``. It depends on the source face
    alone, and it is the only constant that survives the piecewise-constant
    basis.

    Parameters
    ----------
    face_source
        Face of the source cell.
    component
        Which component of the curl, in the order x, y, z.

    Returns
    -------
    float
        ``(e_p x e_q) . n'`` for that face and component.
    """
    import numpy as _np

    p, q = _CURL_COMPONENT_AXES[component]
    return float(
        _np.dot(
            _np.cross(_np.eye(3)[p], _np.eye(3)[q]),
            _np.array(_FACE_NORMALS[face_source]),
        )
    )


def _k_coefficients(normal, normal_source, resolution, wavenumber, n_functions):
    """Return MARIE's ``coefficients_Kop.m`` for one face pair, every term."""
    import numpy as _np

    unit = _np.eye(3)
    out = _np.zeros((4, 3, n_functions, n_functions), dtype=complex)
    for component, (p, q) in enumerate(_CURL_COMPONENT_AXES):
        e_p, e_q = unit[p], unit[q]
        out[0, component] = _np.dot(_np.cross(e_p, e_q), normal_source)
        for test in range(n_functions):
            h = _step(test, resolution)
            for basis in range(n_functions):
                hp = _step(basis, resolution)
                out[1, component, test, basis] = (
                    _np.dot(normal_source, _np.dot(_np.cross(e_q, e_p), hp) * h)
                    / (1j * wavenumber) ** 2
                )
            out[3, component, test, :] = sum(
                _np.dot(_np.cross(e_q, unit[a]), normal_source)
                * _np.dot(h * _np.dot(e_p, unit[a]), normal)
                for a in range(3)
            )
        for basis in range(n_functions):
            hp = _step(basis, resolution)
            out[2, component, :, basis] = _np.dot(normal, normal_source) * _np.dot(
                hp, _np.cross(e_p, e_q)
            )
    return out


def _k_integrand(
    term,
    observer,
    source,
    centre_observer,
    centre_source,
    test,
    basis,
    normal,
    normal_source,
    resolution,
    wavenumber,
):
    """Return MARIE's ``kernels_Kop.m`` at every point of the face-pair rule."""
    del normal_source
    separation, distance, green, static = _face_greens(observer, source, wavenumber)
    if term == 0:
        return (
            _scalar_function(observer, centre_observer, test, resolution)
            * _scalar_function(source, centre_source, basis, resolution)
            * (_radial_field(separation, distance, green, static, wavenumber) @ normal)
        )
    if term == 1:
        field = _radial_field(separation, distance, green, static, wavenumber)
        return (field - separation / 2.0 * static[:, None]) @ normal
    difference = (green - static) / (1j * wavenumber) ** 2
    if term == 2:
        return (
            _scalar_function(observer, centre_observer, test, resolution) * difference
        )
    return _scalar_function(source, centre_source, basis, resolution) * difference


class _Reduction:
    """How one operator's four reduced face kernels are built.

    Attributes
    ----------
    axes
        The Cartesian pair each stored component tests.
    depends
        For each kernel, whether its integral depends on the test function and
        on the source function.
    directfn_type
        The reduced kernel DIRECTFN evaluates for each term; ``Kernels.cpp``
        branches on 1 through 8.
    """

    def __init__(self, axes, depends, directfn_type, coefficients, integrand):
        self.axes = axes
        self.depends = depends
        self.directfn_type = directfn_type
        self.coefficients = coefficients
        self.integrand = integrand


_N_REDUCTION = _Reduction(
    axes=_COMPONENT_AXES,
    depends=((True, True), (False, True), (True, False), (False, False)),
    directfn_type=(1, 2, 3, 4),
    coefficients=_n_coefficients,
    integrand=_n_integrand,
)

_K_REDUCTION = _Reduction(
    axes=_CURL_COMPONENT_AXES,
    depends=((True, True), (False, False), (True, False), (False, True)),
    directfn_type=(5, 6, 7, 8),
    coefficients=_k_coefficients,
    integrand=_k_integrand,
)


def surface_surface_k(
    offset_cells,
    resolution: float,
    wavenumber: float,
    order: int = 15,
    *,
    linear: bool = False,
) -> torch.Tensor:
    """Integrate the K kernel over two cells by their faces.

    The counterpart of :func:`surface_surface_n` for the curl operator, ported
    from ``surface_surface_kernels_Kop.m`` and its companions. Its first kernel
    contracts the separation with the *observation* face normal where the
    dyadic operator's contracts with the source's.

    Parameters
    ----------
    offset_cells
        Whole-cell offset from the source cell to the observation cell.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Points per axis, for the four-dimensional rule and for DIRECTFN alike.
    linear
        Return every pair of :data:`PAIRS` rather than the constant pair alone.

    Returns
    -------
    torch.Tensor
        Shape ``(3,)``, or ``(10, 3)`` when ``linear``, complex, in the order
        x, y, z.
    """
    return _surface_surface(
        offset_cells, resolution, wavenumber, order, linear, _K_REDUCTION
    )


def kernel_k(
    shape: tuple[int, int, int],
    resolution: float,
    wavenumber: float,
    *,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
    linear: bool = False,
) -> torch.Tensor:
    """Assemble the K kernel at every offset of a grid.

    The three regimes of :func:`kernel_n`, for the curl operator.

    Parameters
    ----------
    shape
        Grid shape; the kernel is stored at every non-negative offset.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    far_order, medium_order
        Points per axis for the two volume-volume passes.
    near_order
        Points per axis for the surface-surface pass.
    linear
        Assemble every pair of :data:`PAIRS` of the linear basis.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3, 3)``, or ``(n1, n2, n3, 10, 3)`` when ``linear``,
        complex, in the order x, y, z.
    """
    return _assemble(
        shape,
        resolution,
        wavenumber,
        far_order,
        medium_order,
        near_order,
        volume_volume_k,
        surface_surface_k,
        3,
        linear,
    )
