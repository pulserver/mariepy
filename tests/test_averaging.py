"""The averaging cubes of IEC/IEEE 62704-1 over a body."""

import numpy as np
import pytest
import torch

from mariepy import averaging

RESOLUTION = 0.002
TARGET = 1e-3


def _body(size=13, seed=0, device="cpu", hollow=True):
    """A ball of tissue of uneven density, with a notch cut out of it."""
    generator = np.random.default_rng(seed)
    axis = np.arange(size) - (size - 1) / 2
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    tissue = np.sqrt(x**2 + y**2 + z**2) <= size / 2 - 1.0
    if hollow:
        tissue &= ~((np.abs(x) < 1.5) & (np.abs(y) < 1.5) & (z > 0))
    density = 900.0 + 300.0 * generator.random((size,) * 3)
    mass = np.where(tissue, density * RESOLUTION**3, 0.0)
    field = generator.random((size,) * 3) * tissue
    return (
        torch.from_numpy(mass).to(device),
        torch.from_numpy(tissue).to(device),
        torch.from_numpy(field).to(device),
    )


def _reference(mass, tissue, target):
    """The standard's first step, one voxel at a time, as it is written.

    A cube is grown about each voxel a shell at a time; the shell it stops in is
    entered by a fraction that solves a cubic in the shell's corner, edge and
    face masses.
    """
    mass, tissue = mass.cpu().numpy(), tissue.cpu().numpy()
    shape = mass.shape
    start = max(0, int(np.floor((np.cbrt(target / mass[tissue].max()) - 1) / 2)))
    cubes = {}
    for centre in np.argwhere(tissue):
        half, held, previous = start, 0.0, 0.0
        while True:
            if any(
                c - half < 0 or c + half >= n
                for c, n in zip(centre, shape, strict=True)
            ):
                break
            cut = tuple(slice(c - half, c + half + 1) for c in centre)
            if half > start or start == 0:
                block = tissue[cut]
                faces = [
                    block.take(index, axis=axis).any()
                    for axis in range(3)
                    for index in (0, -1)
                ]
                if not all(faces):
                    break
            previous, held = held, mass[cut].sum()
            if (held - target) / target >= -1e-6:
                break
            half += 1
        else:  # pragma: no cover - the loop always breaks
            continue
        if half - start < 0 or (held - target) / target < -1e-6:
            continue
        if any(
            c - half < 0 or c + half >= n for c, n in zip(centre, shape, strict=True)
        ):
            continue
        if half > start or start == 0:
            cut = tuple(slice(c - half, c + half + 1) for c in centre)
            block = tissue[cut]
            if not all(
                block.take(index, axis=axis).any()
                for axis in range(3)
                for index in (0, -1)
            ):
                continue
        fill = 1.0
        if (held - target) / target > 1e-6:
            weight = _shell_weights(shape, centre, half)
            corner = float((mass * (weight == 3)).sum())
            edge = float((mass * (weight == 2)).sum())
            face = float((mass * (weight == 1)).sum())
            roots = np.roots([corner, edge, face, previous - target])
            real = [r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real <= 1]
            fill = max(real)
        weight = _weights(shape, centre, half, fill)
        volume = float(weight.sum())
        if float((weight * ~tissue).sum()) / volume > 0.1:
            continue
        cubes[tuple(centre)] = (half, fill)
    return cubes


def _shell_weights(shape, centre, half):
    """Say, for every voxel, how many of its coordinates sit on the cube's shell."""
    counts = np.zeros(shape, dtype=int)
    cut = tuple(slice(c - half, c + half + 1) for c in centre)
    for axis, c in enumerate(centre):
        edge = np.zeros(shape, dtype=bool)
        for index in (c - half, c + half):
            take = [slice(None)] * 3
            take[axis] = index
            edge[tuple(take)] = True
        counts += edge
    inside = np.zeros(shape, dtype=bool)
    inside[cut] = True
    return counts * inside


def _weights(shape, centre, half, fill):
    """Give each voxel's share of the cube."""
    counts = _shell_weights(shape, centre, half)
    cut = tuple(slice(c - half, c + half + 1) for c in centre)
    inside = np.zeros(shape, dtype=float)
    inside[cut] = 1.0
    return inside * fill**counts


def test_every_cube_holds_the_mass_it_was_asked_for(device):
    """The fraction the outermost shell is entered by is what fixes the mass."""
    mass, tissue, _ = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    held = averaging.averaged(cubes, torch.ones_like(mass), mass) * TARGET
    torch.testing.assert_close(held, torch.full_like(held, TARGET))


def test_the_cubes_are_the_ones_the_standard_grows_voxel_by_voxel(device):
    mass, tissue, _ = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    reference = _reference(mass, tissue, TARGET)
    assert len(cubes) == len(reference)
    centres = 0.5 * (cubes.lower + cubes.upper)
    extents = 0.5 * cubes.side
    for centre, extent in zip(centres, extents, strict=True):
        want_half, want_fill = reference[tuple(round(c) for c in centre.tolist())]
        half = round(float(extent[0]))
        assert half == want_half
        assert float(extent[0]) - half + 0.5 == pytest.approx(want_fill, abs=1e-9)
        torch.testing.assert_close(extent, extent[0].expand(3))


def test_a_cube_one_shell_smaller_would_not_hold_the_mass(device):
    """A cube grows no further than it must, so its fill is what it needs."""
    mass, tissue, _ = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    centre = 0.5 * (cubes.lower + cubes.upper)
    shrunk = torch.round(0.5 * cubes.side) - 0.5
    smaller = averaging.Cubes(
        lower=centre - shrunk, upper=centre + shrunk, target_mass=cubes.target_mass
    )
    held = averaging.averaged(smaller, torch.ones_like(mass), mass) * TARGET
    assert bool((held <= TARGET * (1 + 1e-6)).all())


def test_a_uniform_field_averages_to_itself(device):
    mass, tissue, _ = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    field = torch.full_like(mass, 7.0)
    got = averaging.averaged(cubes, field, mass)
    torch.testing.assert_close(got, torch.full_like(got, 7.0))


def test_an_average_lies_between_the_values_its_cube_covers(device):
    """Every cube is a convex combination of the tissue it holds."""
    mass, tissue, field = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    got = averaging.averaged(cubes, field, mass)
    inside = field[tissue]
    assert float(got.min()) >= float(inside.min()) - 1e-12
    assert float(got.max()) <= float(inside.max()) + 1e-12


def test_averaging_is_linear_in_the_field_it_carries(device):
    mass, tissue, field = _body(device=device)
    other = field.flip(0)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    together = averaging.averaged(cubes, 2.0 * field - 3.0 * other, mass)
    apart = 2.0 * averaging.averaged(cubes, field, mass) - 3.0 * averaging.averaged(
        cubes, other, mass
    )
    torch.testing.assert_close(together, apart)


def test_a_batch_of_fields_averages_as_the_fields_do_one_by_one(device):
    mass, tissue, field = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    batch = torch.stack([field, field.flip(1), field.flip(2)])
    together = averaging.averaged(cubes, batch, mass)
    assert together.shape == (3, len(cubes))
    for index, one in enumerate(batch):
        torch.testing.assert_close(
            together[index], averaging.averaged(cubes, one, mass)
        )


def test_no_cube_reaches_outside_the_grid(device):
    mass, tissue, _ = _body(device=device)
    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    for axis, size in enumerate(mass.shape):
        assert bool((cubes.lower[:, axis] >= -0.5).all())
        assert bool((cubes.upper[:, axis] <= size - 0.5).all())


def test_a_body_too_thin_to_fill_a_cube_with_tissue_has_none(device):
    """A slab two voxels thick cannot hold a cube that is a tenth background."""
    mass = torch.zeros((13, 13, 13), dtype=torch.float64, device=device)
    tissue = torch.zeros((13, 13, 13), dtype=torch.bool, device=device)
    tissue[:, :, 6:8] = True
    mass[tissue] = 1000.0 * RESOLUTION**3
    with pytest.raises(ValueError, match="no voxel reaches"):
        averaging.centred_cubes(mass, tissue, TARGET)


def test_a_body_with_no_tissue_is_refused(device):
    grid = torch.zeros((5, 5, 5), dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match="no tissue"):
        averaging.centred_cubes(grid, grid.to(torch.bool), TARGET)


def test_a_cube_s_matrix_is_the_sar_of_every_drive_averaged_over_it(device):
    """Averaging the matrices and averaging the SAR are the same operation."""
    from mariepy import sar

    mass, tissue, _ = _body(device=device)
    generator = torch.Generator().manual_seed(3)
    channels = 3
    electric = torch.randn(
        channels, 3, *mass.shape, dtype=torch.complex128, generator=generator
    ).to(device)
    conductivity = (0.5 * tissue).to(torch.float64)
    density = mass / RESOLUTION**3
    matrices = sar.local_matrices(
        electric, conductivity, density.clamp(min=1.0), tissue
    )

    cubes = averaging.centred_cubes(mass, tissue, TARGET)
    over_cubes = averaging.averaged_matrices(cubes, matrices, mass, tissue)
    assert over_cubes.shape == (len(cubes), channels, channels)
    torch.testing.assert_close(over_cubes, over_cubes.conj().transpose(-2, -1))

    drive = torch.randn(channels, dtype=torch.complex128, generator=generator).to(
        device
    )
    local = torch.zeros(mass.shape, dtype=torch.float64, device=device)
    local[tissue] = sar.sar(matrices, drive)
    torch.testing.assert_close(
        sar.sar(over_cubes, drive), averaging.averaged(cubes, local, mass)
    )


def _overlap(lower, upper, index):
    """Give how much of voxel ``index`` lies between two bounds."""
    return max(0.0, min(upper, index + 0.5) - max(lower, index - 0.5))


def _shares(shape, lower, upper):
    """Give every voxel's share of one box, straight from the definition."""
    weights = np.ones(shape)
    for axis in range(3):
        along = np.array(
            [
                _overlap(float(lower[axis]), float(upper[axis]), j)
                for j in range(shape[axis])
            ]
        )
        weights = weights * along.reshape(
            [-1 if other == axis else 1 for other in range(3)]
        )
    return weights


def _face_box(centre, axis, sign, side):
    """Give the box of a cube with ``centre`` on one of its faces."""
    lower = [float(c) - side / 2 for c in centre]
    upper = [float(c) + side / 2 for c in centre]
    near = float(centre[axis]) - 0.5 * sign
    lower[axis], upper[axis] = (
        min(near, near + sign * side),
        max(near, near + sign * side),
    )
    return lower, upper


def _brute_face_side(mass, centre, axis, sign, target, start):
    """Grow a face-centred cube a voxel at a time, then bisect the last step."""
    shape = mass.shape

    def held(side):
        lower, upper = _face_box(centre, axis, sign, side)
        if any(lower[d] < -0.5 or upper[d] > shape[d] - 0.5 for d in range(3)):
            return None
        return float((_shares(shape, lower, upper) * mass).sum())

    reach = start + 1
    while reach <= max(shape):
        total = held(reach)
        if total is None:
            return None
        if (total - target) / target >= -1e-6:
            low, high = reach - 1.0, float(reach)
            for _ in range(60):
                middle = 0.5 * (low + high)
                if held(middle) < target:
                    low = middle
                else:
                    high = middle
            return 0.5 * (low + high)
        reach += 1
    return None


def test_every_cube_in_the_pool_holds_the_target_mass_by_its_own_shares(device):
    """The vectorised read of a cube agrees with the overlaps that define it."""
    mass, tissue, _ = _body(device=device)
    pool = averaging.cube_pool(mass, tissue, TARGET)
    numpy_mass = mass.cpu().numpy()
    for index in range(0, len(pool), max(1, len(pool) // 40)):
        weights = _shares(numpy_mass.shape, pool.lower[index], pool.upper[index])
        assert float((weights * numpy_mass).sum()) == pytest.approx(TARGET, rel=1e-12)


def test_the_pool_reads_every_cube_as_its_overlaps_do(device):
    mass, tissue, field = _body(device=device)
    pool = averaging.cube_pool(mass, tissue, TARGET)
    got = averaging.averaged(pool, field, mass)
    numpy_mass, numpy_field = mass.cpu().numpy(), field.cpu().numpy()
    for index in range(0, len(pool), max(1, len(pool) // 20)):
        weights = _shares(numpy_mass.shape, pool.lower[index], pool.upper[index])
        want = float((weights * numpy_mass * numpy_field).sum()) / TARGET
        assert float(got[index]) == pytest.approx(want, rel=1e-10)


def test_the_second_step_takes_the_voxels_the_first_one_does_not_enclose(device):
    mass, tissue, _ = _body(device=device)
    first = averaging.centred_cubes(mass, tissue, TARGET)
    left = tissue & ~averaging._enclosed(first, tuple(mass.shape))
    assert bool(left.any())
    for centre in left.nonzero()[:: max(1, int(left.sum()) // 10)]:
        share = 0.0
        for index in range(len(first)):
            weights = _shares(mass.shape, first.lower[index], first.upper[index])
            share = max(share, float(weights[tuple(centre.tolist())]))
        assert share <= averaging.WHOLE_VOXEL


def test_a_face_cube_grows_from_the_voxel_it_serves(device):
    """The voxel sits on one face of its cube and at the centre of the other two."""
    mass, tissue, _ = _body(device=device)
    first = averaging.centred_cubes(mass, tissue, TARGET)
    second = averaging.face_cubes(mass, tissue, TARGET, first)
    assert len(second) > 0
    side = second.side
    torch.testing.assert_close(side, side[:, :1].expand(-1, 3))
    on_face = ((second.lower + 0.5 - torch.round(second.lower + 0.5)).abs() < 1e-9) | (
        (second.upper - 0.5 - torch.round(second.upper - 0.5)).abs() < 1e-9
    )
    assert bool(on_face.any(dim=1).all())


def test_the_sides_of_a_face_cube_are_the_ones_grown_voxel_by_voxel(device):
    import itertools

    mass, tissue, _ = _body(size=11, device=device)
    first = averaging.centred_cubes(mass, tissue, TARGET)
    left = tissue & ~averaging._enclosed(first, tuple(mass.shape))
    second = averaging.face_cubes(mass, tissue, TARGET, first)
    numpy_mass = mass.cpu().numpy()
    start = max(
        0,
        int(
            np.floor((np.cbrt(TARGET / numpy_mass[tissue.cpu().numpy()].max()) - 1) / 2)
        ),
    )
    seen = {
        (
            tuple(round(float(x), 6) for x in low),
            tuple(round(float(x), 6) for x in high),
        )
        for low, high in zip(second.lower, second.upper, strict=True)
    }
    for centre in left.nonzero().tolist():
        sides = {}
        for axis, sign in itertools.product(range(3), (1, -1)):
            side = _brute_face_side(numpy_mass, centre, axis, sign, TARGET, start)
            if side is not None:
                sides[(axis, sign)] = side
        if not sides:
            continue
        smallest = min(sides.values())
        for (axis, sign), side in sides.items():
            if side**3 > smallest**3 * averaging.VOLUME_SPREAD:
                continue
            low, high = _face_box(centre, axis, sign, side)
            key = (
                tuple(round(x, 6) for x in low),
                tuple(round(x, 6) for x in high),
            )
            assert key in seen


def test_the_pool_holds_both_families_and_nothing_else(device):
    mass, tissue, _ = _body(device=device)
    first = averaging.centred_cubes(mass, tissue, TARGET)
    second = averaging.face_cubes(mass, tissue, TARGET, first)
    pool = averaging.cube_pool(mass, tissue, TARGET)
    assert len(pool) == len(first) + len(second)
    torch.testing.assert_close(pool.lower[: len(first)], first.lower)
    torch.testing.assert_close(pool.lower[len(first) :], second.lower)


def test_two_families_averaged_over_different_masses_do_not_pool(device):
    mass, tissue, _ = _body(device=device)
    first = averaging.centred_cubes(mass, tissue, TARGET)
    other = averaging.centred_cubes(mass, tissue, 2 * TARGET)
    with pytest.raises(ValueError, match="averages over"):
        first.join(other)
