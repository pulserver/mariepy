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

Ported from the Apache-2.0 reference implementation of the standard,
<https://github.com/umbertozanovello/IEC-IEEE-62704-1-spatial-average-SAR>:
``core/avgSARStep1.c`` and the mass and usage helpers of ``core/auxFuncs.c``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch

__all__ = ["Cubes", "averaged", "averaged_matrices", "centred_cubes"]

MASS_TOLERANCE = 1e-6
"""How far a cube's mass may sit from the target before its shell is cut back."""

BACKGROUND_SHARE = 0.1
"""How much of a valid cube's volume may be background."""


@dataclass(frozen=True)
class Cubes:
    """Averaging cubes over one body, each holding the target mass.

    Attributes
    ----------
    centre
        Shape ``(n_cubes, 3)``, the voxel each cube is built about.
    half
        Shape ``(n_cubes,)``, the cube's half-width in voxels: it spans
        ``centre - half`` to ``centre + half`` inclusive.
    fill
        Shape ``(n_cubes,)``, how far into the outermost shell the cube's faces
        reach, in ``(0, 1]``.
    target_mass
        The mass each cube holds, in kilograms.
    """

    centre: torch.Tensor
    half: torch.Tensor
    fill: torch.Tensor
    target_mass: float

    def __len__(self) -> int:
        """Count the cubes."""
        return int(self.centre.shape[0])


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
    start = max(0, math.floor((math.cbrt(target_mass / heaviest) - 1) / 2))

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
    centre = kept.nonzero()
    return Cubes(
        centre=centre,
        half=half_of[kept],
        fill=fill_of[kept],
        target_mass=float(target_mass),
    )


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


def averaged(cubes: Cubes, field: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Average a voxel field over every cube, weighting each voxel by its mass.

    Parameters
    ----------
    cubes
        From :func:`centred_cubes`.
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
    shape = tuple(mass.shape)
    weighted = (field * mass).reshape(-1, *shape)
    out = torch.zeros(
        (weighted.shape[0], len(cubes)), dtype=weighted.dtype, device=weighted.device
    )
    for half in sorted({int(value) for value in cubes.half}):
        here = cubes.half == half
        centre = cubes.centre[here]
        index = tuple(centre[:, axis] - max(half, 0) for axis in range(3))
        fill = cubes.fill[here].to(weighted.dtype)
        for row, plane in enumerate(weighted):
            corner, edge, face, interior = _cube_parts(plane, half)
            total = interior[index] + fill * face[index]
            out[row, here] = total + fill**2 * edge[index] + fill**3 * corner[index]
    return (out / cubes.target_mass).reshape(*field.shape[:-3], len(cubes))


def _cube_parts(plane: torch.Tensor, half: int):
    """Give a plane's corner, edge, face and interior sums at one half-width."""
    prefix = (
        _prefix(plane.real) + 1j * _prefix(plane.imag)
        if plane.is_complex()
        else _prefix(plane)
    )
    if half == 0:
        whole = _windowed(
            prefix, (0,) * 3, (1,) * 3, _centred(0, [n - 1 for n in prefix.shape])
        )
        zero = torch.zeros_like(whole)
        return zero, zero, zero, whole
    _, interior, corner, edge, face = _shell_sums(prefix, half)
    return corner, edge, face, interior


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
