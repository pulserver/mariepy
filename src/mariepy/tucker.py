"""Tucker compression of the three-level circulant body kernel.

The volume integral operator couples every voxel to every other through a
kernel that depends only on the offset between them, so each Cartesian
component of it is a three-level Toeplitz tensor. Embedding that tensor in a
circulant of at least twice the extent along each axis turns its application
into three FFTs, and a truncated higher-order SVD stores the embedded tensor in
a fraction of the memory the full grid would take.

Ported from MARIE 3.0's ``src_mathematics/src_numerical_linear_algebra/
src_tucker/{hosvd,hosvd_to_full,nmp}.m`` and
``src_integral_equations/src_vie/src_operators_vie/
assembly_fft_circ_tucker_pwc.m``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "CirculantSymbol",
    "circulant_tucker",
    "hosvd",
    "mode_product",
    "pair_parity",
    "to_full",
    "transform_length",
]

# Parity of each component of the symmetric dyadic kernel under reflection of
# the offset in x, y and z. The component order is xx, xy, xz, yy, yz, zz, and
# the mirrored half of the circulant embedding carries these signs.
_PARITY_XX_TO_ZZ = (
    (+1, -1, -1, +1, +1, +1),
    (+1, -1, +1, +1, -1, +1),
    (+1, +1, -1, +1, -1, +1),
)

# Parity of the three components of a vector kernel, used where the operator
# has one component per axis rather than the six of a symmetric dyadic.
_PARITY_X_TO_Z = (
    (-1, +1, +1),
    (+1, -1, +1),
    (+1, +1, -1),
)


@dataclass(frozen=True)
class CirculantSymbol:
    """One component of the kernel, compressed and transformed.

    Attributes
    ----------
    core
        Tucker core, shape ``(r1, r2, r3)``.
    factors
        Three factor matrices, of shapes ``(r1, L1)``, ``(r2, L2)`` and
        ``(r3, L3)`` with ``Li = transform_length(ni)``. Each is the circulant
        extension of a Tucker factor along its axis, already Fourier
        transformed, so expanding the core through them gives the operator's
        symbol on the extended grid.
    """

    core: torch.Tensor
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the symbol this expands to."""
        return tuple(factor.shape[1] for factor in self.factors)  # type: ignore[return-value]

    def expand(self) -> torch.Tensor:
        """Return the symbol on the extended grid, shape :attr:`shape`."""
        return to_full(self.core, *self.factors)


def mode_product(tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
    """Contract one mode of a tensor with a matrix.

    Parameters
    ----------
    tensor
        Tensor of any order; mode ``mode`` has extent ``k``.
    matrix
        Shape ``(k, m)``. Its first axis is contracted with the named mode.
    mode
        Axis of ``tensor`` to contract, counted from zero.

    Returns
    -------
    torch.Tensor
        ``tensor`` with mode ``mode`` replaced by an axis of extent ``m``.
    """
    contracted = torch.tensordot(tensor, matrix, dims=([mode], [0]))
    return torch.movedim(contracted, -1, mode)


def to_full(
    core: torch.Tensor,
    factor_1: torch.Tensor,
    factor_2: torch.Tensor,
    factor_3: torch.Tensor,
) -> torch.Tensor:
    """Expand a Tucker core through its three factors.

    Parameters
    ----------
    core
        Shape ``(r1, r2, r3)``.
    factor_1, factor_2, factor_3
        Shapes ``(r1, n1)``, ``(r2, n2)`` and ``(r3, n3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3)``.
    """
    expanded = mode_product(core, factor_1, 0)
    expanded = mode_product(expanded, factor_2, 1)
    return mode_product(expanded, factor_3, 2)


def hosvd(
    tensor: torch.Tensor, tol: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress a third-order tensor by a truncated higher-order SVD.

    Each mode is truncated at the first singular value at or below
    ``tol * s[0] / sqrt(3)``, that value included, which is the rule MARIE
    applies. Exact zeros do not trigger the truncation.

    Parameters
    ----------
    tensor
        Shape ``(n1, n2, n3)``.
    tol
        Relative tolerance. ``None`` keeps every mode at full rank.

    Returns
    -------
    core : torch.Tensor
        Shape ``(r1, r2, r3)``.
    factor_1, factor_2, factor_3 : torch.Tensor
        Shapes ``(r1, n1)``, ``(r2, n2)`` and ``(r3, n3)``, so that
        :func:`to_full` reconstructs the tensor.

    Raises
    ------
    ValueError
        ``tensor`` is not third order.
    """
    if tensor.ndim != 3:
        raise ValueError(f"hosvd takes a third-order tensor, got {tensor.ndim} axes")

    factors = []
    for mode in range(3):
        unfolding = torch.movedim(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        left, singular_values, _ = torch.linalg.svd(unfolding, full_matrices=False)
        rank = _truncation_rank(singular_values, tol)
        factors.append(left[:, :rank])

    core = tensor
    for mode, factor in enumerate(factors):
        core = mode_product(core, factor.conj(), mode)

    return core, *(factor.transpose(0, 1) for factor in factors)


def _truncation_rank(singular_values: torch.Tensor, tol: float | None) -> int:
    """Return how many singular values to keep, by MARIE's rule."""
    count = singular_values.numel()
    if tol is None or count == 0:
        return count

    threshold = tol * singular_values[0].item() / math.sqrt(3.0)
    below = (singular_values <= threshold) & (singular_values != 0)
    indices = torch.nonzero(below, as_tuple=False)
    if indices.numel() == 0:
        return count
    return int(indices[0].item()) + 1


def pair_parity(test: int, source: int) -> tuple[int, int, int]:
    """Return the parity of a pair of linear basis functions under reflection.

    Ported as a rule from the ``bt_n`` tables of MARIE's
    ``assembly_fft_circ_tucker_pwl.m``: reflecting the offset along an axis
    flips the sign of every basis function linear in that axis, so a pair
    changes sign once for each of its two functions that is.

    Parameters
    ----------
    test, source
        Scalar basis functions: 0 for the constant, 1 to 3 for x, y, z.

    Returns
    -------
    tuple of int
        The sign along each of the three axes.
    """
    return tuple(
        (-1) ** ((test == axis + 1) + (source == axis + 1)) for axis in range(3)
    )


def circulant_tucker(kernel: torch.Tensor, tol: float | None = None):
    """Embed each component of a Toeplitz kernel in a circulant and compress it.

    Ported from MARIE's ``assembly_fft_circ_tucker_pwc.m`` and
    ``assembly_fft_circ_tucker_pwl.m``.

    Parameters
    ----------
    kernel
        Shape ``(n1, n2, n3, n_components)`` for the constant basis, or
        ``(n1, n2, n3, 10, n_components)`` for the linear basis with one entry
        per pair of :data:`mariepy.vie.PAIRS`, holding the kernel at
        non-negative offsets. ``n_components`` is 6 for a symmetric dyadic
        operator, in the order xx, xy, xz, yy, yz, zz, or 3 for a vector
        operator, in the order x, y, z; the parity of each component under
        reflection of the offset follows from that order.
    tol
        Relative tolerance passed to :func:`hosvd`.

    Returns
    -------
    tuple
        One :class:`CirculantSymbol` per component, or for the linear basis one
        such tuple per pair.

    Raises
    ------
    ValueError
        ``kernel`` has neither four nor five axes, or carries a component count
        with no parity assignment.
    """
    if kernel.ndim == 5:
        from mariepy.vie import PAIRS

        if kernel.shape[3] != len(PAIRS):
            raise ValueError(
                f"a linear kernel carries {len(PAIRS)} pairs, got {kernel.shape[3]}"
            )
        return tuple(
            _circulant_tucker(kernel[..., index, :], tol, pair_parity(*pair))
            for index, pair in enumerate(PAIRS)
        )
    if kernel.ndim != 4:
        raise ValueError(f"the kernel needs four or five axes, got {kernel.ndim}")
    return _circulant_tucker(kernel, tol, (1, 1, 1))


def _circulant_tucker(kernel, tol, pair_sign):
    """Compress every component of a four-axis kernel under one pair parity."""
    n_components = kernel.shape[3]
    if n_components == 6:
        parity = _PARITY_XX_TO_ZZ
    elif n_components == 3:
        parity = _PARITY_X_TO_Z
    else:
        raise ValueError(
            f"a kernel of {n_components} components has no parity assignment; "
            f"expected 6 for a symmetric dyadic or 3 for a vector operator"
        )

    symbols = []
    for component in range(n_components):
        core, *factors = hosvd(kernel[..., component], tol)
        transformed = tuple(
            _circulant_extension(factor, parity[axis][component] * pair_sign[axis])
            for axis, factor in enumerate(factors)
        )
        symbols.append(CirculantSymbol(core=core, factors=transformed))
    return tuple(symbols)


_SMOOTH_PRIMES = (2, 3, 5, 7)


def transform_length(n: int) -> int:
    """Give the transform length the circulant embedding of ``n`` offsets uses.

    A Toeplitz block of ``n`` rows sits inside any circulant of ``2 * n - 1``
    rows or more, so the length is free above that, and an FFT of a length whose
    prime factors are all small is several times faster than one of a length
    with a large prime factor. The shortest length with no prime factor above
    seven is taken.

    Parameters
    ----------
    n
        Offsets along the axis, the body's extent in voxels.

    Returns
    -------
    int
        The transform length, at least ``2 * n - 1``.
    """
    least = max(1, 2 * n - 1)
    length = least
    while True:
        rest = length
        for prime in _SMOOTH_PRIMES:
            while rest % prime == 0:
                rest //= prime
        if rest == 1:
            return length
        length += 1


def _circulant_extension(factor: torch.Tensor, sign: int) -> torch.Tensor:
    """Mirror one Tucker factor into a circulant and transform it.

    Parameters
    ----------
    factor
        Shape ``(rank, n)``, one factor of the Tucker decomposition.
    sign
        Parity of this component under reflection of the offset along this
        axis.

    Returns
    -------
    torch.Tensor
        Shape ``(rank, transform_length(n))``, Fourier transformed along the
        extended axis.
    """
    rank, n = factor.shape
    length = transform_length(n)
    extended = torch.zeros((rank, length), device=factor.device, dtype=factor.dtype)
    extended[:, :n] = factor
    if n > 1:
        extended[:, length - n + 1 :] = sign * torch.flip(factor[:, 1:], dims=(1,))
    return torch.fft.fft(extended, dim=1)
