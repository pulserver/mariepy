"""The near block: face adjacency, the surface-surface reduction, assembly."""

import itertools

import numpy as np
import pytest
import torch

from mariepy import _accelerators, vie

directfn = _accelerators.require(module="mariepy._directfn")

RESOLUTION = 0.01
WAVENUMBER = 30.0
CELL = 1.0

# MARIE's points_mapping.m writes its corners as p1..p8 of a cell of side dx
# centred on the origin. These are the same eight points, for the spot checks.
_P = [
    np.array(v, dtype=float) * CELL / 2.0
    for v in (
        (-1, -1, -1),
        (+1, -1, -1),
        (+1, +1, -1),
        (-1, +1, -1),
        (-1, -1, +1),
        (+1, -1, +1),
        (+1, +1, +1),
        (-1, +1, +1),
    )
]


def _p(index, offset=(0, 0, 0)):
    return _P[index - 1] + CELL * np.array(offset, dtype=float)


def _cells(radius=2):
    return itertools.product(range(radius), repeat=3)


def test_every_face_of_a_cell_has_four_distinct_corners_of_that_cell():
    for face in range(6):
        corners = vie.face_cycle(face, (0.0, 0.0, 0.0), CELL)
        assert len(corners) == 4
        assert len({tuple(round(v, 12) for v in c) for c in corners}) == 4
        axis, sign = face // 2, (-1.0, 1.0)[face % 2]
        for corner in corners:
            assert corner[axis] == pytest.approx(sign * CELL / 2.0)


def test_consecutive_corners_of_a_face_are_an_edge_apart():
    """A cycle round a square steps along an edge each time, never a diagonal."""
    for face in range(6):
        corners = vie.face_cycle(face, (0.0, 0.0, 0.0), CELL)
        for index in range(4):
            step = np.array(corners[(index + 1) % 4]) - np.array(corners[index])
            assert np.linalg.norm(step) == pytest.approx(CELL)


@pytest.mark.parametrize("offset", list(_cells(3)))
def test_the_adjacency_matches_how_far_apart_the_cells_are(offset):
    """Cells sharing a face, an edge or a corner, and cells sharing nothing."""
    kinds = set()
    for face_observer in range(6):
        for face_source in range(6):
            kind, _ = vie.face_adjacency(offset, face_observer, face_source, CELL)
            kinds.add(kind)
    touching = max(abs(value) for value in offset) <= 1
    assert (kinds != {None}) == touching


@pytest.mark.parametrize("offset", list(_cells(2)))
def test_every_derived_point_is_a_corner_of_the_cell_it_belongs_to(offset):
    source = {tuple(round(v, 12) for v in _p(i)) for i in range(1, 9)}
    observer = {tuple(round(v, 12) for v in _p(i, offset)) for i in range(1, 9)}
    for face_observer in range(6):
        for face_source in range(6):
            kind, points = vie.face_adjacency(offset, face_observer, face_source, CELL)
            if kind is None:
                continue
            for point in points:
                rounded = tuple(round(v, 12) for v in point)
                assert rounded in source or rounded in observer


@pytest.mark.parametrize("offset", list(_cells(2)))
def test_the_shared_feature_sits_where_directfn_reads_it(offset):
    """A misplaced shared vertex changes the answer, so its slot is the contract."""
    for face_observer in range(6):
        for face_source in range(6):
            kind, points = vie.face_adjacency(offset, face_observer, face_source, CELL)
            if kind is None:
                continue
            source = vie.face_cycle(face_source, (0.0, 0.0, 0.0), CELL)
            observer = vie.face_cycle(
                face_observer, CELL * np.array(offset, dtype=float), CELL
            )
            shared = {
                tuple(round(v, 12) for v in a)
                for a in source
                if any(np.allclose(a, b) for b in observer)
            }
            slot = {"self": set(range(4)), "edge": {2, 3}, "vertex": {2}}[kind]
            assert len(shared) == {"self": 4, "edge": 2, "vertex": 1}[kind]
            assert {tuple(round(v, 12) for v in points[i]) for i in slot} == shared


@pytest.mark.parametrize("offset", list(_cells(2)))
def test_the_first_four_points_are_the_source_face(offset):
    for face_observer in range(6):
        for face_source in range(6):
            kind, points = vie.face_adjacency(offset, face_observer, face_source, CELL)
            if kind is None:
                continue
            face = vie.face_cycle(face_source, (0.0, 0.0, 0.0), CELL)
            assert {tuple(round(v, 12) for v in p) for p in points[:4]} == {
                tuple(round(v, 12) for v in p) for p in face
            }


@pytest.mark.parametrize("offset", list(_cells(2)))
def test_the_adjacency_is_the_same_seen_from_either_cell(offset):
    """Swapping the cells swaps the faces and negates the offset."""
    opposite = tuple(-value for value in offset)
    for face_observer in range(6):
        for face_source in range(6):
            here, _ = vie.face_adjacency(offset, face_observer, face_source, CELL)
            there, _ = vie.face_adjacency(opposite, face_source, face_observer, CELL)
            assert here == there


def _singular_value(kind, points, face_observer, face_source, order=5):
    vertices = np.array(points, dtype=float)
    routine, centre = vie._singular_call(kind, vertices, directfn)
    return routine(
        vertices,
        (vertices[0] + vertices[2]) / 2.0,
        centre,
        np.array(vie._FACE_NORMALS[face_source]),
        np.array(vie._FACE_NORMALS[face_observer]),
        0.2 * np.pi,
        CELL,
        order,
        1,
        0,
        0,
    )


# Entries read out of MARIE's points_mapping.m, as (offset, observation face,
# source face, the points that file lists). Faces are zero-based here.
MARIE_ENTRIES = [
    ((0, 0, 0), 0, 0, [_p(1), _p(4), _p(8), _p(5)]),
    ((0, 0, 0), 0, 2, [_p(2), _p(6), _p(5), _p(1), _p(4), _p(8)]),
    ((0, 0, 0), 2, 0, [_p(4), _p(8), _p(5), _p(1), _p(2), _p(6)]),
    ((0, 0, 0), 1, 4, [_p(1), _p(4), _p(3), _p(2), _p(6), _p(7)]),
    ((0, 0, 0), 5, 3, [_p(4), _p(3), _p(7), _p(8), _p(5), _p(6)]),
    (
        (1, 1, 1),
        0,
        1,
        [
            _p(2),
            _p(3),
            _p(7),
            _p(6),
            _p(4, (1, 1, 1)),
            _p(8, (1, 1, 1)),
            _p(5, (1, 1, 1)),
        ],
    ),
    (
        (1, 1, 1),
        0,
        3,
        [
            _p(4),
            _p(8),
            _p(7),
            _p(3),
            _p(5, (1, 1, 1)),
            _p(8, (1, 1, 1)),
            _p(4, (1, 1, 1)),
        ],
    ),
    (
        (1, 1, 1),
        2,
        1,
        [
            _p(2),
            _p(6),
            _p(7),
            _p(3),
            _p(5, (1, 1, 1)),
            _p(6, (1, 1, 1)),
            _p(2, (1, 1, 1)),
        ],
    ),
]


@pytest.mark.parametrize(
    ("offset", "face_observer", "face_source", "tabulated"), MARIE_ENTRIES
)
def test_the_derived_points_integrate_to_what_maries_table_does(
    offset, face_observer, face_source, tabulated
):
    """The derivation replaces 144 hand-written arrays, so it answers to them."""
    kind, derived = vie.face_adjacency(offset, face_observer, face_source, CELL)
    assert kind is not None
    mine = _singular_value(kind, derived, face_observer, face_source)
    theirs = _singular_value(kind, tabulated, face_observer, face_source)
    assert abs(mine - theirs) <= 1e-12 * abs(theirs)


@pytest.mark.parametrize("offset", [(2, 0, 0), (2, 1, 0), (3, 2, 1)])
def test_the_surface_reduction_agrees_with_the_volume_rule_where_both_hold(offset):
    """Two independent forms of the same operator, away from the singularity."""
    reduced = vie.surface_surface_n(offset, RESOLUTION, WAVENUMBER, order=12)
    direct = vie.volume_volume_n(
        RESOLUTION * torch.tensor([offset], dtype=torch.float64),
        RESOLUTION,
        WAVENUMBER,
        order=10,
    )[0]
    error = torch.linalg.vector_norm(reduced - direct)
    assert error <= 1e-10 * torch.linalg.vector_norm(direct)


def test_the_self_term_has_no_off_diagonal_component():
    """A cube is symmetric under reflection, so its self term is diagonal."""
    value = vie.surface_surface_n((0, 0, 0), RESOLUTION, WAVENUMBER, order=6)
    assert abs(value[1]) <= 1e-20
    assert abs(value[2]) <= 1e-20
    assert abs(value[4]) <= 1e-20
    assert abs(value[0]) > 0


def test_the_self_term_is_the_same_along_every_axis():
    value = vie.surface_surface_n((0, 0, 0), RESOLUTION, WAVENUMBER, order=6)
    assert value[0].item() == pytest.approx(value[3].item(), rel=1e-12)
    assert value[0].item() == pytest.approx(value[5].item(), rel=1e-12)


def test_the_assembled_kernel_is_finite_at_every_offset():
    kernel = vie.kernel_n(
        (4, 4, 4), RESOLUTION, WAVENUMBER, far_order=3, medium_order=4, near_order=5
    )
    assert kernel.shape == (4, 4, 4, 6)
    assert torch.all(torch.isfinite(kernel))


def test_the_assembled_kernel_takes_its_near_block_from_the_surface_reduction():
    kernel = vie.kernel_n(
        (3, 3, 3), RESOLUTION, WAVENUMBER, far_order=3, medium_order=3, near_order=5
    )
    for cell in _cells(2):
        expected = vie.surface_surface_n(cell, RESOLUTION, WAVENUMBER, order=5)
        assert torch.allclose(kernel[cell], expected)


def test_the_assembled_kernel_falls_away_with_distance():
    kernel = vie.kernel_n(
        (5, 1, 1), RESOLUTION, WAVENUMBER, far_order=4, medium_order=4, near_order=5
    )
    magnitudes = [abs(kernel[offset, 0, 0, 0].item()) for offset in range(5)]
    assert all(later < earlier for earlier, later in itertools.pairwise(magnitudes))
