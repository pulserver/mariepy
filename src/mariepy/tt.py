"""Tensor trains, and the cross approximation that builds one from its entries.

Ported from MARIE 3.0's copy of TT-Toolbox (MIT, Oseledets and co-authors):
``TT_QTT/cross/dmrg_cross_gpu.m`` and the helpers it calls,
``core/maxvol2.m``, ``core/my_chop2.m``, ``core/reort.m`` and
``core/tt_ind2sub.m``, and from MARIE's ``mvp_TT_core.m``,
``mvp_transpose_TT_core.m`` and ``full_tt.m``.

A tensor train of order ``d`` is a list of cores, core ``k`` of shape
``(r_k, n_k, r_{k+1})`` with ``r_0 = r_d = 1``; the entry at ``(i_1, ..., i_d)``
is the product of the matrices ``core_k[:, i_k, :]``. The cross approximation
samples the tensor only along index sets chosen by the maximum-volume rule,
so a tensor far too large to hold is built from a few of its entries.

The cross approximation works in MATLAB's column-major index order throughout,
so its reshapes are written as :func:`_reshape`, which reproduces MATLAB's.
Indices handed to the entry function are zero-based.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

__all__ = ["TensorTrain", "cross"]


@dataclass(frozen=True)
class TensorTrain:
    """A tensor in tensor-train format.

    Attributes
    ----------
    cores
        Core ``k`` has shape ``(r_k, n_k, r_{k+1})``, the outer ranks one.
    """

    cores: tuple[torch.Tensor, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the size of each index."""
        return tuple(int(core.shape[1]) for core in self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """Return the ``d + 1`` ranks, the outer two ones."""
        return (1, *(int(core.shape[2]) for core in self.cores))

    def full(self) -> torch.Tensor:
        """Contract every core into the full tensor.

        Returns
        -------
        torch.Tensor
            Shape :attr:`shape`.
        """
        result = self.cores[0]
        for core in self.cores[1:]:
            result = torch.tensordot(result, core, dims=1)
        return result[0, ..., 0]

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        """Contract the last index with a vector.

        Ported from MARIE's ``mvp_TT_core.m`` for a train of order four.

        Parameters
        ----------
        vector
            Shape ``(n_last,)``.

        Returns
        -------
        torch.Tensor
            The first ``d - 1`` indices, shape :attr:`shape` without its last.
        """
        last = self.cores[-1][:, :, 0]
        result = last @ vector.to(last)
        for core in reversed(self.cores[:-1]):
            result = torch.tensordot(core, result, dims=1)
        return result[0]

    def apply_transpose(self, field: torch.Tensor) -> torch.Tensor:
        """Contract the first ``d - 1`` indices with a field over them.

        Ported from MARIE's ``mvp_transpose_TT_core.m``.

        Parameters
        ----------
        field
            Shape :attr:`shape` without its last.

        Returns
        -------
        torch.Tensor
            Shape ``(n_last,)``.
        """
        first = self.cores[0][0]
        result = torch.tensordot(field.to(first), first, dims=([0], [0]))
        for core in self.cores[1:-1]:
            result = torch.tensordot(result, core, dims=([0, -1], [1, 0]))
        return torch.tensordot(result, self.cores[-1][:, :, 0], dims=([0], [0]))

    def to(self, device: torch.device | str) -> TensorTrain:
        """Return the same train on another device.

        Parameters
        ----------
        device
            Where the cores go.

        Returns
        -------
        TensorTrain
            The moved train.
        """
        return TensorTrain(cores=tuple(core.to(device) for core in self.cores))


def _reshape(tensor: torch.Tensor, shape) -> torch.Tensor:
    """Reshape as MATLAB does, first index fastest."""
    shape = tuple(int(size) for size in shape)
    reversed_axes = tuple(range(tensor.ndim - 1, -1, -1))
    flipped = tensor.permute(reversed_axes).reshape(shape[::-1])
    return flipped.permute(tuple(range(len(shape) - 1, -1, -1)))


def _left_divide(matrix: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Solve ``matrix @ x = right``, in the least-squares sense when not square."""
    if matrix.shape[0] == matrix.shape[1]:
        return torch.linalg.solve(matrix, right)
    return torch.linalg.lstsq(matrix, right).solution


def _right_divide(left: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Solve ``x @ matrix = left``, in the least-squares sense when not square."""
    return _left_divide(matrix.transpose(0, 1), left.transpose(0, 1)).transpose(0, 1)


def _ind2sub(sizes, index: int) -> tuple[int, ...]:
    """Return the zero-based subscripts of a zero-based column-major index.

    Ported from ``tt_ind2sub.m``.
    """
    subscripts = []
    for size in sizes:
        subscripts.append(index % size)
        index //= size
    return tuple(subscripts)


def _maxvol(matrix: torch.Tensor, *, tolerance: float = 5e-2, iterations: int = 100):
    """Return rows whose square submatrix has close to the largest volume.

    Ported from ``maxvol2.m``. The search starts from the pivots of a
    partially pivoted LU factorisation and swaps rows while an entry of the
    interpolation matrix exceeds ``1 + tolerance`` in size. MARIE's copy never
    advances its iteration count, so it can loop without end; here the count
    advances, and the rows found by then are returned.

    Returns
    -------
    list of int
        Zero-based row indices, one per column.
    """
    n, r = matrix.shape
    if n <= r:
        return list(range(n))
    permutation, _, _ = torch.linalg.lu(matrix)
    order = permutation.real.transpose(0, 1).argmax(dim=1)
    rows = [int(value) for value in order[:r]]
    interpolation = _right_divide(matrix, matrix[rows])
    for _ in range(iterations + 1):
        # Column-major position of the largest entry, as MATLAB's max(b(:)).
        magnitude = interpolation.abs().transpose(0, 1).reshape(-1)
        largest = int(magnitude.argmax())
        if float(magnitude[largest]) <= 1.0 + tolerance:
            return sorted(rows)
        column, row = divmod(largest, n)
        current = rows[column]
        interpolation = interpolation + torch.outer(
            interpolation[:, column],
            (interpolation[current] - interpolation[row]) / interpolation[row, column],
        )
        rows[column] = row
    return rows


def _chop(singular_values: torch.Tensor, tolerance: float) -> int:
    """Return how many singular values keep the discarded tail below a tolerance.

    Ported from ``my_chop2.m``.
    """
    if float(torch.linalg.vector_norm(singular_values)) == 0.0:
        return 1
    if tolerance <= 0:
        return int(singular_values.numel())
    tail = torch.cumsum(singular_values.flip(0) ** 2, dim=0)
    below = torch.nonzero(tail < tolerance**2).flatten()
    if below.numel() == 0:
        return int(singular_values.numel())
    return int(singular_values.numel()) - (int(below[-1]) + 1)


def _reorthogonalise(basis: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
    """Append new directions to an orthonormal basis, orthogonal to it.

    Ported from ``reort.m``. The test for a second pass compares each new
    direction's squared modulus with its start's; MARIE's copy squares without
    conjugating and so compares real parts on complex data. Either way the
    directions returned are orthonormal and orthogonal to the basis.
    """
    if extra.shape[1] == 0 or basis.shape[0] == basis.shape[1]:
        return basis
    if basis.shape[1] + extra.shape[1] >= basis.shape[0]:
        extra = extra[:, : basis.shape[0] - basis.shape[1]]
    new = extra - basis @ (basis.mH @ extra)
    for _ in range(20):
        new_norm = (new.abs() ** 2).sum(dim=0)
        extra_norm = (extra.abs() ** 2).sum(dim=0)
        again = bool((new_norm <= 0.25 * extra_norm).any())
        new, _ = torch.linalg.qr(new)
        if not again:
            return torch.cat([basis, new], dim=1)
        extra = new
        new = new - basis @ (basis.mH @ new)
    combined, _ = torch.linalg.qr(torch.cat([basis, new], dim=1))
    return combined


def _rank_bounds(sizes) -> list[int]:
    """Return the largest rank each bond can carry."""
    d = len(sizes)
    return [
        min(math.prod(sizes[:bond]), math.prod(sizes[bond:])) for bond in range(d + 1)
    ]


def cross(
    sizes,
    entries: Callable[[torch.Tensor], torch.Tensor],
    tolerance: float,
    *,
    kick: int = 5,
    sweeps: int = 5,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.complex128,
) -> TensorTrain:
    """Build a tensor train from its entries by the DMRG cross method.

    Ported from MARIE's ``dmrg_cross_gpu.m``.

    Parameters
    ----------
    sizes
        The size of each index.
    entries
        Given an integer array of shape ``(m, d)`` of zero-based subscripts,
        returns the ``m`` entries there.
    tolerance
        Relative accuracy of each local truncation, and of the sweep error at
        which the method stops.
    kick
        Random directions added to each local basis, which lets the ranks
        grow.
    sweeps
        Most sweeps to make.
    generator
        Source of the random start and the random kicks.
    dtype
        Entry type.

    Returns
    -------
    TensorTrain
        The approximation.
    """
    n = [int(size) for size in sizes]
    d = len(n)
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64).to(dtype)

    bounds = _rank_bounds(n)
    ry = [min(2, bounds[bond]) for bond in range(d + 1)]
    ry[0] = ry[d] = 1
    y = [random(ry[k], n[k], ry[k + 1]) for k in range(d)]

    rmat = [None] * (d + 1)
    rmat[0] = torch.ones((1, 1), dtype=dtype)
    rmat[d] = torch.ones((1, 1), dtype=dtype)
    index = [None] * (d + 1)
    index[d] = torch.zeros((0, ry[d]), dtype=torch.int64)
    index[0] = torch.zeros((ry[0], 0), dtype=torch.int64)
    r1 = torch.ones((1, 1), dtype=dtype)

    # Right-to-left: orthogonalise each core and pick its right index set.
    for i in range(d - 1, 0, -1):
        core = _reshape(y[i], (ry[i] * n[i], ry[i + 1])) @ r1
        core = _reshape(core, (ry[i], n[i] * ry[i + 1])).transpose(0, 1)
        core, rm = torch.linalg.qr(core)
        rows = _maxvol(core)
        old = index[i + 1]
        new_rank = min(n[i] * ry[i + 1], ry[i])
        fresh = torch.zeros((d - i, new_rank), dtype=torch.int64)
        for s in range(new_rank):
            rank_index, position = _ind2sub((ry[i + 1], n[i]), rows[s])
            fresh[:, s] = torch.cat([torch.tensor([position]), old[:, rank_index]])
        index[i] = fresh
        r1 = core[rows]
        core = _right_divide(core, r1)
        r1 = (r1 @ rm).transpose(0, 1)
        core = core.transpose(0, 1)
        y[i] = _reshape(core, (ry[i], n[i], ry[i + 1]))
        core = _reshape(core, (ry[i] * n[i], ry[i + 1])) @ rmat[i + 1]
        core = _reshape(core, (ry[i], n[i] * ry[i + 1])).transpose(0, 1)
        _, rm = torch.linalg.qr(core)
        rmat[i] = rm

    core = _reshape(y[0], (ry[0] * n[0], ry[1]))
    y[0] = _reshape(core @ r1, (ry[0], n[0], ry[1]))

    direction = 1
    i = 0
    sweep = 1
    worst = 0.0
    while sweep < sweeps:
        left, right = index[i], index[i + 2]
        count = ry[i] * n[i] * n[i + 1] * ry[i + 2]
        a = torch.arange(count)
        left_rank = a % ry[i]
        first = (a // ry[i]) % n[i]
        second = (a // (ry[i] * n[i])) % n[i + 1]
        right_rank = a // (ry[i] * n[i] * n[i + 1])
        subscripts = torch.cat(
            [
                left[left_rank],
                first[:, None],
                second[:, None],
                right[:, right_rank].transpose(0, 1),
            ],
            dim=1,
        )
        score = entries(subscripts).to(dtype)

        score = rmat[i] @ _reshape(score, (ry[i], n[i] * n[i + 1] * ry[i + 2]))
        ry[i] = score.shape[0]
        score = _reshape(score, (ry[i] * n[i] * n[i + 1], ry[i + 2])) @ rmat[i + 2]
        ry[i + 2] = score.shape[1]
        score = _reshape(score, (ry[i] * n[i], n[i + 1] * ry[i + 2]))

        u, s, vh = torch.linalg.svd(score, full_matrices=False)
        v = vh.mH
        rank = _chop(
            s, float(torch.linalg.vector_norm(s)) * tolerance / math.sqrt(d - 1)
        )
        u, v, s = u[:, :rank], v[:, :rank], s[:rank].to(dtype)

        if direction == 1:
            v = v * s.conj()[None, :]
            u = _reorthogonalise(u, random(u.shape[0], kick))
            added = u.shape[1] - rank
            if added > 0:
                v = torch.cat([v, torch.zeros((v.shape[0], added), dtype=dtype)], dim=1)
        else:
            u = u * s[None, :]
            v = _reorthogonalise(v, random(v.shape[0], kick))
            added = v.shape[1] - rank
            if added > 0:
                u = torch.cat([u, torch.zeros((u.shape[0], added), dtype=dtype)], dim=1)
        rank += added
        v = v.mH

        approximation = _reshape(y[i], (-1, ry[i + 1])) @ _reshape(
            y[i + 1], (ry[i + 1], -1)
        )
        approximation = rmat[i] @ _reshape(
            approximation, (ry[i], n[i] * n[i + 1] * ry[i + 2])
        )
        approximation = (
            _reshape(approximation, (ry[i] * n[i] * n[i + 1], ry[i + 2])) @ rmat[i + 2]
        )
        local = float(
            torch.linalg.vector_norm(score.reshape(-1) - approximation.reshape(-1))
            / torch.linalg.vector_norm(score.reshape(-1))
        )
        worst = max(worst, local)
        ry[i + 1] = rank

        u = _left_divide(rmat[i], _reshape(u, (ry[i], n[i] * rank)))
        v = _reshape(v, (rank * n[i + 1], ry[i + 2]))
        u = _reshape(u, (ry[i] * n[i], ry[i + 1]))
        v = _right_divide(v, rmat[i + 2])
        v = _reshape(v, (rank, n[i + 1] * ry[i + 2]))

        if direction == 1:
            u, rm = torch.linalg.qr(u)
            rows = _maxvol(u)
            r1 = u[rows]
            u = _right_divide(u, r1)
            y[i] = _reshape(u, (ry[i], n[i], ry[i + 1]))
            r1 = r1 @ rm
            v = r1 @ v
            y[i + 1] = _reshape(v, (ry[i + 1], n[i + 1], ry[i + 2]))
            u1 = rmat[i] @ _reshape(u, (ry[i], n[i] * ry[i + 1]))
            _, rm = torch.linalg.qr(_reshape(u1, (ry[i] * n[i], ry[i + 1])))
            rmat[i + 1] = rm
            old = index[i]
            fresh = torch.zeros((ry[i + 1], i + 1), dtype=torch.int64)
            for s_index in range(ry[i + 1]):
                rank_index, position = _ind2sub((ry[i], n[i]), rows[s_index])
                fresh[s_index] = torch.cat([old[rank_index], torch.tensor([position])])
            index[i + 1] = fresh
            if i == d - 2:
                direction = -direction
            else:
                i += 1
        else:
            v = v.transpose(0, 1)
            v, rm = torch.linalg.qr(v)
            rows = _maxvol(v)
            r1 = v[rows]
            v = _right_divide(v, r1)
            y[i + 1] = _reshape(v, (n[i + 1], ry[i + 2], ry[i + 1])).permute(2, 0, 1)
            r1 = (r1 @ rm).transpose(0, 1)
            u = u @ r1
            y[i] = _reshape(u, (ry[i], n[i], ry[i + 1]))
            v = _reshape(v.transpose(0, 1), (ry[i + 1] * n[i + 1], ry[i + 2]))
            v = _reshape(v @ rmat[i + 2], (ry[i + 1], n[i + 1] * ry[i + 2]))
            _, rm = torch.linalg.qr(v.transpose(0, 1))
            rmat[i + 1] = rm
            old = index[i + 2]
            fresh = torch.zeros((d - i - 1, ry[i + 1]), dtype=torch.int64)
            for s_index in range(ry[i + 1]):
                position, rank_index = _ind2sub((n[i + 1], ry[i + 2]), rows[s_index])
                fresh[:, s_index] = torch.cat(
                    [torch.tensor([position]), old[:, rank_index]]
                )
            index[i + 1] = fresh
            if i == 0:
                direction = -direction
                sweep += 1
                if worst < tolerance:
                    break
                worst = 0.0
            else:
                i -= 1

    return TensorTrain(cores=tuple(y))
