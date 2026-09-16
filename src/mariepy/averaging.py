"""Spatial-average SAR over the cubes of IEC/IEEE 62704-1.

The standard averages local SAR over a cube of tissue of a target mass, 10 g for
the head and trunk. A cube is grown about a voxel until it holds that mass, so
its outermost shell is entered only partly: the cube's face is a fraction ``k``
of a voxel into that shell, which gives the shell's face voxels weight ``k``,
its edge voxels ``k**2`` and its corner voxels ``k**3``, and fixes ``k`` as the
root in ``(0, 1]`` of a cubic in those three masses.

Averaging is linear in whatever is averaged, so a cube is a set of weights and
nothing about a drive enters here. :func:`averaged` carries any voxel field
through the cubes, and :mod:`mariepy.sar`'s matrices through it give one SAR
matrix per cube. The peak spatial-average SAR of a drive is then the largest
``v^H Q v`` over the cubes, which is what the standard's two steps assign to
voxels one at a time.

A voxel that no valid cube encloses is averaged over a second family instead:
six cubes with the voxel on a face, grown outward and sideways at once, of which
the smallest few count. Both families end up in the same pool.

Ported from the Apache-2.0 reference implementation of the standard,
<https://github.com/umbertozanovello/IEC-IEEE-62704-1-spatial-average-SAR>:
``core/avgSARStep1.c``, ``core/avgSARStep2.c`` and the mass and usage helpers of
``core/auxFuncs.c``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

__all__ = [
    "Cubes",
    "averaged",
    "averaged_matrices",
    "centred_cubes",
    "cube_pool",
    "face_cubes",
]

MASS_TOLERANCE = 1e-6
"""How far a cube's mass may sit from the target before its shell is cut back."""

BACKGROUND_SHARE = 0.1
"""How much of a valid cube's volume may be background."""

VOLUME_SPREAD = 1.05
"""How much larger than the smallest of a voxel's six cubes another may be."""

WHOLE_VOXEL = 0.999
"""The share of a voxel above which a cube counts as enclosing it."""


@dataclass(frozen=True)
class Cubes:
    """Averaging volumes over one body, each holding the target mass.

    A cube is an axis-aligned box in voxel coordinates, where voxel ``j`` spans
    ``j - 0.5`` to ``j + 0.5``. A voxel's share of a cube is its overlap with
    the box along each axis, multiplied together, so a cube whose face falls
    part way through a voxel takes that fraction of it.

    Attributes
    ----------
    lower, upper
        Shape ``(n_cubes, 3)``, the box in voxel coordinates.
    target_mass
        The mass each cube holds, in kilograms.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    target_mass: float

    def __len__(self) -> int:
        """Count the cubes."""
        return int(self.lower.shape[0])

    @property
    def side(self) -> torch.Tensor:
        """Give each cube's side along each axis, in voxels."""
        return self.upper - self.lower

    @property
    def volume(self) -> torch.Tensor:
        """Give each cube's volume, in voxels."""
        return self.side.prod(dim=-1)

    def join(self, other: Cubes) -> Cubes:
        """Put two families of cubes into one pool.

        Raises
        ------
        ValueError
            The two average over different masses.
        """
        if self.target_mass != other.target_mass:
            raise ValueError(
                f"one family averages over {self.target_mass} kg and the other "
                f"over {other.target_mass} kg"
            )
        return Cubes(
            lower=torch.cat([self.lower, other.lower]),
            upper=torch.cat([self.upper, other.upper]),
            target_mass=self.target_mass,
        )


def _boxed(centre: torch.Tensor, extent: torch.Tensor, target_mass: float) -> Cubes:
    """Build cubes from their centres and half-sides."""
    return Cubes(
        lower=centre - extent, upper=centre + extent, target_mass=float(target_mass)
    )


def _cube_root(value: float) -> float:
    """Give the cube root of a positive number.

    Spelled as a power rather than with ``math.cbrt``, which needs Python 3.11
    where this package supports 3.10.
    """
    return value ** (1.0 / 3.0)


def _prefix(grid: torch.Tensor) -> torch.Tensor:
    """Give the inclusive prefix sum of a grid, with a zero plane on each low face."""
    padded = torch.zeros(
        tuple(size + 1 for size in grid.shape), dtype=torch.float64, device=grid.device
    )
    padded[1:, 1:, 1:] = grid
    return padded.cumsum(0).cumsum(1).cumsum(2)


def _windowed(prefix: torch.Tensor, low, high, window) -> torch.Tensor:
    """Sum a grid over one box about every centre in a window.

    Parameters
    ----------
    prefix
        From :func:`_prefix`.
    low, high
        The box, as offsets from the centre: it spans ``centre + low`` up to but
        not including ``centre + high``, per axis.
    window
        Per axis, the first and last-plus-one centre to read. Every offset must
        keep the box inside the grid over that range.

    Returns
    -------
    torch.Tensor
        One value per centre in the window.
    """
    total = None
    for corner in itertools.product((0, 1), repeat=3):
        offsets = [high[axis] if corner[axis] else low[axis] for axis in range(3)]
        cut = tuple(
            slice(window[axis][0] + offsets[axis], window[axis][1] + offsets[axis])
            for axis in range(3)
        )
        term = (-1.0) ** (3 - sum(corner)) * prefix[cut]
        total = term if total is None else total + term
    return total


def _centred(half: int, shape):
    """Give the window of centres a cube of this half-width fits about."""
    return [(half, size - half) for size in shape]


def _shell_sums(prefix: torch.Tensor, half: int) -> tuple[torch.Tensor, ...]:
    """Sum a grid over the pieces one cube's outermost shell splits into.

    Returns
    -------
    tuple
        The whole cube, the cube one shell smaller, and the shell's corner, edge
        and face parts, each about every centre clear of ``half``.
    """
    window = _centred(half, [size - 1 for size in prefix.shape])
    whole = _windowed(prefix, (-half,) * 3, (half + 1,) * 3, window)
    inner = _windowed(prefix, (-half + 1,) * 3, (half,) * 3, window)

    corners = torch.zeros_like(whole)
    for signs in itertools.product((-1, 1), repeat=3):
        low = [sign * half for sign in signs]
        corners = corners + _windowed(prefix, low, [value + 1 for value in low], window)

    faces = torch.zeros_like(whole)
    for axis, sign in itertools.product(range(3), (-1, 1)):
        low = [-half + 1] * 3
        high = [half] * 3
        low[axis] = sign * half
        high[axis] = sign * half + 1
        faces = faces + _windowed(prefix, low, high, window)

    return whole, inner, corners, whole - inner - corners - faces, faces


def _cube_faces_hold_tissue(prefix: torch.Tensor, half: int) -> torch.Tensor:
    """Say whether every face of the cube touches or cuts tissue."""
    shape = [size - 1 for size in prefix.shape]
    holds = None
    for axis, sign in itertools.product(range(3), (-1, 1)):
        low = [-half] * 3
        high = [half + 1] * 3
        low[axis] = sign * half
        high[axis] = sign * half + 1
        face = _windowed(prefix, low, high, _centred(half, shape)) > 0.5
        holds = face if holds is None else (holds & face)
    return holds


def _fill(
    corner: torch.Tensor, edge: torch.Tensor, face: torch.Tensor, gap: torch.Tensor
):
    """Solve ``corner k^3 + edge k^2 + face k + gap = 0`` for the root in ``(0, 1]``.

    ``gap`` is how far the cube one shell smaller falls short of the target, so
    it is negative and the polynomial rises through zero exactly once in the
    unit interval.
    """
    low = torch.zeros_like(gap)
    high = torch.ones_like(gap)
    for _ in range(60):
        middle = 0.5 * (low + high)
        value = ((corner * middle + edge) * middle + face) * middle + gap
        low = torch.where(value < 0, middle, low)
        high = torch.where(value < 0, high, middle)
    return 0.5 * (low + high)


def _inner(half: int, shape) -> tuple[slice, ...]:
    """Give the slice of centres a cube of this half-width fits about."""
    return tuple(slice(half, size - half) for size in shape)


def centred_cubes(
    mass: torch.Tensor, tissue: torch.Tensor, target_mass: float
) -> Cubes:
    """Build the averaging cube of every voxel the standard finds one for.

    This is step 1 of IEC/IEEE 62704-1: a cube is grown evenly about each tissue
    voxel until it holds ``target_mass``. It is kept when it stays inside the
    grid, every one of its faces touches or cuts tissue, and no more than a
    tenth of its volume is background.

    Parameters
    ----------
    mass
        Each voxel's mass in kilograms, shape ``(n1, n2, n3)``, zero where there
        is no tissue.
    tissue
        Where the body is, the same shape.
    target_mass
        The mass to average over, in kilograms.

    Returns
    -------
    Cubes
        One cube per voxel the standard accepts one for.

    Raises
    ------
    ValueError
        The grid holds no tissue, or no voxel is heavy enough to reach the
        target mass within the grid.
    """
    if not bool(tissue.any()):
        raise ValueError("a body with no tissue has nothing to average over")
    shape = tuple(mass.shape)
    device = mass.device
    heaviest = float(mass[tissue].max())
    start = max(0, math.floor((_cube_root(target_mass / heaviest) - 1) / 2))

    mass_prefix = _prefix(mass)
    tissue_prefix = _prefix(tissue.to(torch.float64))

    settled = ~tissue
    kept = torch.zeros(shape, dtype=torch.bool, device=device)
    half_of = torch.zeros(shape, dtype=torch.long, device=device)
    fill_of = torch.ones(shape, dtype=torch.float64, device=device)

    half = start
    while not bool(settled.all()) and 2 * half + 1 <= min(shape):
        inner = _inner(half, shape)
        outside = torch.ones(shape, dtype=torch.bool, device=device)
        outside[inner] = False
        settled |= outside  # the cube has left the grid, so this voxel has none

        if half > start or start == 0:
            faces = torch.zeros(shape, dtype=torch.bool, device=device)
            faces[inner] = _cube_faces_hold_tissue(tissue_prefix, half)
            settled |= ~faces  # a face that sees only background ends the growth

        whole = torch.zeros(shape, dtype=torch.float64, device=device)
        whole[inner] = _windowed(
            mass_prefix, (-half,) * 3, (half + 1,) * 3, _centred(half, shape)
        )
        reached = (whole - target_mass) / target_mass >= -MASS_TOLERANCE
        found = reached & ~settled
        if bool(found.any()):
            half_of[found] = half
            fill_of[found] = 1.0
            over = found & ((whole - target_mass) / target_mass > MASS_TOLERANCE)
            if bool(over.any()):
                fill_of[over] = _partial_fill(
                    mass_prefix, half, target_mass, shape, over
                )
            kept[found] = _within_background_share(
                tissue_prefix, half, fill_of, shape, found
            )
            settled |= found
        half += 1

    if not bool(kept.any()):
        raise ValueError(
            f"no voxel reaches {target_mass * 1e3:.3g} g inside a grid of "
            f"{shape[0]}x{shape[1]}x{shape[2]} voxels"
        )
    extent = (half_of[kept].to(torch.float64) - 0.5 + fill_of[kept]).unsqueeze(-1)
    return _boxed(kept.nonzero().to(torch.float64), extent, target_mass)


def _partial_fill(mass_prefix, half, target_mass, shape, where):
    """Give the fill of the cubes that overshoot the target at this half-width."""
    device = mass_prefix.device
    inner = _inner(half, shape)
    if half == 0:
        # A single voxel already heavier than the target is used in proportion.
        centre = torch.zeros(shape, dtype=torch.float64, device=device)
        centre[inner] = _windowed(mass_prefix, (0,) * 3, (1,) * 3, _centred(0, shape))
        return (target_mass / centre)[where]
    whole, smaller, corner, edge, face = _shell_sums(mass_prefix, half)
    del whole
    spread = []
    for part in (corner, edge, face, smaller):
        full = torch.zeros(shape, dtype=torch.float64, device=device)
        full[inner] = part
        spread.append(full[where])
    return _fill(spread[0], spread[1], spread[2], spread[3] - target_mass)


def _within_background_share(tissue_prefix, half, fill_of, shape, where):
    """Say whether background takes no more than its share of each cube's volume."""
    device = tissue_prefix.device
    if half == 0:
        return torch.ones_like(where[where], dtype=torch.bool)
    inner = _inner(half, shape)
    _, smaller, corner, edge, face = _shell_sums(tissue_prefix, half)
    spread = []
    for part in (corner, edge, face, smaller):
        full = torch.zeros(shape, dtype=torch.float64, device=device)
        full[inner] = part
        spread.append(full[where])
    k = fill_of[where]
    held = spread[3] + k * spread[2] + k**2 * spread[1] + k**3 * spread[0]
    side = 2 * half - 1
    volume = side**3 + k * 6 * side**2 + k**2 * 12 * side + k**3 * 8
    return (volume - held) / volume <= BACKGROUND_SHARE


def _padded_prefix(grid: torch.Tensor) -> torch.Tensor:
    """Prefix-sum a grid with one empty voxel added on every side.

    An entry at index ``p`` along an axis sums the grid below original voxel
    ``p - 1``, so a box that reaches one voxel outside the grid still reads a
    valid entry.
    """
    padded = torch.zeros(
        tuple(size + 2 for size in grid.shape), dtype=grid.dtype, device=grid.device
    )
    padded[1:-1, 1:-1, 1:-1] = grid
    zero = torch.zeros(
        tuple(size + 3 for size in grid.shape), dtype=grid.dtype, device=grid.device
    )
    zero[1:, 1:, 1:] = padded
    return zero.cumsum(0).cumsum(1).cumsum(2)


def _reading(cubes: Cubes):
    """Give the prefix positions and weights each cube reads along each axis.

    A voxel's overlap with the box is one over a run of whole voxels and a
    fraction at each end, so a cube's weighted sum over an axis is four entries
    of the prefix sum with fixed weights. Three axes multiply out to the block
    :func:`_weighted` contracts.

    Returns
    -------
    tuple
        The four prefix positions per axis, shape ``(n_cubes, 3, 4)``, and their
        weights, the same shape.
    """
    first = torch.ceil(cubes.lower + 0.5)
    last = torch.floor(cubes.upper - 0.5)
    low = (first - 0.5 - cubes.lower).clamp(0.0, 1.0)
    high = (cubes.upper - last - 0.5).clamp(0.0, 1.0)
    base = first.to(torch.long)
    top = last.to(torch.long)
    # A prefix entry at p sums the grid below original voxel p - 1.
    position = torch.stack([base, base + 1, top + 2, top + 3], dim=-1)
    weight = torch.stack([-low, low - 1.0, 1.0 - high, high], dim=-1)
    return position, weight


def _weighted(prefix: torch.Tensor, cubes: Cubes, chunk: int = 1 << 16):
    """Sum a grid over every cube, each voxel weighted by its share of it."""
    position, weight = _reading(cubes)
    # Beyond the grid the prefix sum stops rising, so clamping a box that reaches
    # outside it reads exactly the tissue it covers.
    limits = torch.tensor(
        [size - 1 for size in prefix.shape], device=position.device
    ).reshape(1, 3, 1)
    position = position.clamp(torch.zeros_like(limits), limits)
    out = torch.empty(len(cubes), dtype=prefix.dtype, device=prefix.device)
    for start in range(0, len(cubes), chunk):
        cut = slice(start, start + chunk)
        index, factor = position[cut], weight[cut].to(prefix.dtype)
        block = prefix[
            index[:, 0, :, None, None],
            index[:, 1, None, :, None],
            index[:, 2, None, None, :],
        ]
        out[cut] = torch.einsum(
            "ni,nj,nk,nijk->n", factor[:, 0], factor[:, 1], factor[:, 2], block
        )
    return out


def averaged(cubes: Cubes, field: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Average a voxel field over every cube, weighting each voxel by its mass.

    Parameters
    ----------
    cubes
        From :func:`centred_cubes` or :func:`face_cubes`.
    field
        The value at every voxel, shape ``(..., n1, n2, n3)``.
    mass
        Each voxel's mass in kilograms, shape ``(n1, n2, n3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n_cubes)``, each cube's mass-weighted mean over its own
        target mass.
    """
    weighted = (field * mass).reshape(-1, *mass.shape)
    out = torch.stack([_weighted(_padded_prefix(plane), cubes) for plane in weighted])
    return (out / cubes.target_mass).reshape(*field.shape[:-3], len(cubes))


def averaged_matrices(
    cubes: Cubes,
    matrices: torch.Tensor,
    mass: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average a body's SAR matrices over every cube.

    Each cube holds the target mass, so its matrix is the spatial-average SAR
    matrix of that volume: ``v^H Q v`` is what the standard averages for a
    drive, and the largest over the cubes is its peak.

    Parameters
    ----------
    cubes
        From :func:`centred_cubes`.
    matrices
        Shape ``(n_voxels, n_channels, n_channels)``, as
        :func:`mariepy.sar.local_matrices` gives them, over the voxels ``mask``
        selects.
    mass
        Each voxel's mass in kilograms, shape ``(n1, n2, n3)``.
    mask
        Which voxels the matrices cover, the same shape.

    Returns
    -------
    torch.Tensor
        Shape ``(n_cubes, n_channels, n_channels)``, complex, Hermitian.
    """
    channels = matrices.shape[-1]
    flat = mask.reshape(-1)
    out = torch.empty(
        (len(cubes), channels, channels),
        dtype=matrices.dtype,
        device=matrices.device,
    )
    for row in range(channels):
        grid = torch.zeros(
            (channels, mass.numel()), dtype=matrices.dtype, device=matrices.device
        )
        grid[:, flat] = matrices[:, row, :].transpose(0, 1)
        out[:, row, :] = averaged(
            cubes, grid.reshape(channels, *mass.shape), mass
        ).transpose(0, 1)
    return 0.5 * (out + out.conj().transpose(-2, -1))


def _enclosed(cubes: Cubes, shape) -> torch.Tensor:
    """Mark the voxels some cube takes all but a thousandth of.

    The standard averages a voxel over its own cube only when no cube it lies
    inside already covers it, and counts a voxel as covered when its share of
    that cube is above :data:`WHOLE_VOXEL`.
    """
    first = torch.ceil(cubes.lower + 0.5)
    last = torch.floor(cubes.upper - 0.5)
    low = (first - 0.5 - cubes.lower).clamp(0.0, 1.0)
    high = (cubes.upper - last - 0.5).clamp(0.0, 1.0)
    start = (first - (low > WHOLE_VOXEL).to(first.dtype)).to(torch.long)
    stop = (last + (high > WHOLE_VOXEL).to(last.dtype)).to(torch.long) + 1
    counts = torch.zeros(
        tuple(size + 1 for size in shape),
        dtype=torch.float64,
        device=cubes.lower.device,
    )
    ones = torch.ones(len(cubes), dtype=torch.float64, device=cubes.lower.device)
    for corner in itertools.product((0, 1), repeat=3):
        index = tuple(
            (stop if corner[axis] else start)[:, axis].clamp(0, shape[axis])
            for axis in range(3)
        )
        counts.index_put_(index, (-1.0) ** sum(corner) * ones, accumulate=True)
    inside = counts.cumsum(0).cumsum(1).cumsum(2)
    return inside[: shape[0], : shape[1], : shape[2]] > 0.5


def _face_box(centre: torch.Tensor, axis: int, sign: int, side: torch.Tensor):
    """Give the box of a cube with ``centre`` on one of its faces."""
    half = 0.5 * side.unsqueeze(-1)
    lower, upper = centre - half, centre + half
    reach = side if sign > 0 else -side
    near = centre[:, axis] - 0.5 * sign
    lower = lower.clone()
    upper = upper.clone()
    lower[:, axis] = torch.minimum(near, near + reach)
    upper[:, axis] = torch.maximum(near, near + reach)
    return lower, upper


def _fits(lower: torch.Tensor, upper: torch.Tensor, shape) -> torch.Tensor:
    """Say whether a box stays inside the grid."""
    limit = torch.tensor(
        [size - 0.5 for size in shape], dtype=torch.float64, device=lower.device
    )
    return ((lower >= -0.5) & (upper <= limit)).all(dim=-1)


def _side_holding_target(
    prefix: torch.Tensor,
    centre: torch.Tensor,
    axis: int,
    sign: int,
    reached: torch.Tensor,
    target_mass: float,
):
    """Solve for the side in ``[reached - 1, reached]`` that holds the target mass.

    Which voxels the box reads is fixed across that interval, so the prefix sums
    are gathered once and only the fractions at the faces move.
    """
    lower, upper = _face_box(centre, axis, sign, reached - 0.5)
    base = torch.ceil(lower + 0.5).to(torch.long)
    top = torch.floor(upper - 0.5).to(torch.long)
    position = torch.stack([base, base + 1, top + 2, top + 3], dim=-1)
    block = prefix[
        position[:, 0, :, None, None],
        position[:, 1, None, :, None],
        position[:, 2, None, None, :],
    ]
    low, high = reached - 1.0, reached
    for _ in range(60):
        middle = 0.5 * (low + high)
        under, over = _face_box(centre, axis, sign, middle)
        start = (base - 0.5 - under).clamp(0.0, 1.0)
        end = (over - top - 0.5).clamp(0.0, 1.0)
        weight = torch.stack([-start, start - 1.0, 1.0 - end, end], dim=-1)
        held = torch.einsum(
            "ni,nj,nk,nijk->n", weight[:, 0], weight[:, 1], weight[:, 2], block
        )
        short = held < target_mass
        low = torch.where(short, middle, low)
        high = torch.where(short, high, middle)
    return 0.5 * (low + high)


def face_cubes(
    mass: torch.Tensor,
    tissue: torch.Tensor,
    target_mass: float,
    enclosing: Cubes,
) -> Cubes:
    """Build the cubes the standard's second step gives the voxels left over.

    A voxel that no cube of :func:`centred_cubes` encloses sits on a face of six
    cubes instead, one per direction, each grown outward and sideways at once
    until it holds the target mass, with no condition on the background it takes
    in. The smallest of the six, and any within a twentieth of its volume, are
    the ones the standard averages over.

    Parameters
    ----------
    mass
        Each voxel's mass in kilograms, shape ``(n1, n2, n3)``, zero where there
        is no tissue.
    tissue
        Where the body is, the same shape.
    target_mass
        The mass to average over, in kilograms.
    enclosing
        The cubes of the first step, whose reach decides which voxels are left.

    Returns
    -------
    Cubes
        The cubes of every voxel the second step reaches, several per voxel.
    """
    shape = tuple(mass.shape)
    left = tissue & ~_enclosed(enclosing, shape)
    if not bool(left.any()):
        empty = torch.zeros((0, 3), dtype=torch.float64, device=mass.device)
        return Cubes(lower=empty, upper=empty, target_mass=float(target_mass))

    centre = left.nonzero().to(torch.float64)
    prefix = _padded_prefix(mass)
    heaviest = float(mass[tissue].max())
    start = max(0, math.floor((_cube_root(target_mass / heaviest) - 1) / 2))

    sides = []
    for axis, sign in itertools.product(range(3), (1, -1)):
        side = torch.full((centre.shape[0],), math.nan, dtype=torch.float64)
        settled = torch.zeros(centre.shape[0], dtype=torch.bool, device=mass.device)
        reach = float(start + 1)
        while not bool(settled.all()) and reach <= max(shape):
            lower, upper = _face_box(centre, axis, sign, torch.full_like(side, reach))
            outside = ~_fits(lower, upper, shape)
            settled |= outside
            held = _weighted(prefix, Cubes(lower, upper, target_mass))
            found = ((held - target_mass) / target_mass >= -MASS_TOLERANCE) & ~settled
            if bool(found.any()):
                side[found] = _side_holding_target(
                    prefix,
                    centre[found],
                    axis,
                    sign,
                    torch.full((int(found.sum()),), reach, dtype=torch.float64),
                    target_mass,
                )
                settled |= found
            reach += 1.0
        sides.append(side)

    stacked = torch.stack(sides)  # (6, n_left)
    smallest = stacked.nan_to_num(nan=float("inf")).min(dim=0).values
    keep = stacked <= smallest * VOLUME_SPREAD ** (1.0 / 3.0)
    lower, upper = [], []
    for index, (axis, sign) in enumerate(itertools.product(range(3), (1, -1))):
        here = keep[index] & ~stacked[index].isnan()
        if not bool(here.any()):
            continue
        under, over = _face_box(centre[here], axis, sign, stacked[index][here])
        lower.append(under)
        upper.append(over)
    if not lower:
        empty = torch.zeros((0, 3), dtype=torch.float64, device=mass.device)
        return Cubes(lower=empty, upper=empty, target_mass=float(target_mass))
    return Cubes(
        lower=torch.cat(lower), upper=torch.cat(upper), target_mass=float(target_mass)
    )


def cube_pool(mass: torch.Tensor, tissue: torch.Tensor, target_mass: float) -> Cubes:
    """Build every averaging volume the standard puts over a body.

    The peak spatial-average SAR of a drive is the largest ``v^H Q v`` over this
    pool: the standard hands each voxel the largest average of the cubes it
    falls in, so the largest over voxels is the largest over cubes.

    Parameters
    ----------
    mass
        Each voxel's mass in kilograms, shape ``(n1, n2, n3)``, zero where there
        is no tissue.
    tissue
        Where the body is, the same shape.
    target_mass
        The mass to average over, in kilograms.

    Returns
    -------
    Cubes
        Both steps' cubes together.
    """
    first = centred_cubes(mass, tissue, target_mass)
    return first.join(face_cubes(mass, tissue, target_mass, first))
