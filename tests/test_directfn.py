"""The DIRECTFN bindings pass their arguments through and converge.

These kernels are C++ running on CPU buffers, so there is no CUDA leg: what
moves to a device is the assembled operator, not the singular integral.

The invariants here are the ones that catch a binding fault. A vertex array
passed with the wrong stride or the wrong order stops being invariant under a
rigid motion of the whole configuration; a quadrature rule whose weights and
nodes are swapped stops converging as the order rises.
"""

import numpy as np
import pytest

from mariepy import _accelerators, quadrature

directfn = _accelerators.require(module="mariepy._directfn")

WAVENUMBER = 3.0
VOXEL_WAVENUMBER = 0.2 * np.pi
VOXEL_SIZE = 1.0
SHIFT = np.array([1.3, -0.7, 2.1])

# One triangle, then a pair sharing an edge, then a pair sharing a vertex.
TRIANGLE = np.array([[0.0, 0.0, 0.0], [0.11, 0.02, 0.0], [0.01, 0.09, 0.03]])
TRIANGLE_PAIR_EDGE = np.array(
    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.12, 0.11, 0.02]]
)
TRIANGLE_PAIR_VERTEX = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.12, 0.11, 0.02],
        [0.2, 0.15, 0.01],
    ]
)

# Faces of cubic cells in the x = 0 plane: one on its own, then a pair sharing
# an edge, then a pair sharing only a vertex.
_D = VOXEL_SIZE
VOXEL_FACE = np.array([[0, 0, 0], [0, _D, 0], [0, _D, _D], [0, 0, _D]], dtype=float)
VOXEL_PAIR_EDGE = np.array(
    [[0, 0, 0], [0, _D, 0], [0, _D, _D], [0, 0, _D], [0, 0, 2 * _D], [0, _D, 2 * _D]],
    dtype=float,
)
VOXEL_PAIR_VERTEX = np.array(
    [
        [0, 0, 0],
        [0, _D, 0],
        [0, _D, _D],
        [0, 0, _D],
        [0, _D, 2 * _D],
        [0, 2 * _D, 2 * _D],
        [0, 2 * _D, _D],
    ],
    dtype=float,
)
FACE_NORMAL = np.array([1.0, 0.0, 0.0])


def _rule(order):
    """Return the Gauss-Legendre rule as the bindings want it."""
    weights, nodes = quadrature.gauss_legendre_1d(order)
    return weights.numpy(), nodes.numpy()


def _rotation(angle):
    """Return a rotation about z followed by one about x, by the same angle."""
    c, s = np.cos(angle), np.sin(angle)
    about_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return about_z @ about_x


def _triangle_self(vertices, order):
    return directfn.triangle_self(vertices, WAVENUMBER, *_rule(order))


def _triangle_edge(vertices, order):
    weights, nodes = _rule(order)
    return directfn.triangle_edge(vertices, WAVENUMBER, weights, nodes, weights, nodes)


def _triangle_vertex(vertices, order):
    weights, nodes = _rule(order)
    return directfn.triangle_vertex(
        vertices, WAVENUMBER, weights, nodes, weights, nodes, weights, nodes
    )


TRIANGLE_KERNELS = [
    pytest.param(_triangle_self, TRIANGLE, 1e-13, id="self"),
    pytest.param(_triangle_edge, TRIANGLE_PAIR_EDGE, 1e-10, id="edge"),
    pytest.param(_triangle_vertex, TRIANGLE_PAIR_VERTEX, 1e-8, id="vertex"),
]


def _voxel_self(vertices, order):
    return directfn.voxel_self(
        vertices,
        (vertices[0] + vertices[2]) / 2,
        (vertices[0] + vertices[2]) / 2,
        FACE_NORMAL,
        FACE_NORMAL,
        VOXEL_WAVENUMBER,
        VOXEL_SIZE,
        order,
        1,
        1,
        1,
    )


def _voxel_edge(vertices, order):
    return directfn.voxel_edge(
        vertices,
        (vertices[0] + vertices[2]) / 2,
        (vertices[3] + vertices[5]) / 2,
        FACE_NORMAL,
        FACE_NORMAL,
        VOXEL_WAVENUMBER,
        VOXEL_SIZE,
        order,
        1,
        1,
        1,
    )


def _voxel_vertex(vertices, order):
    return directfn.voxel_vertex(
        vertices,
        (vertices[0] + vertices[2]) / 2,
        (vertices[2] + vertices[5]) / 2,
        FACE_NORMAL,
        FACE_NORMAL,
        VOXEL_WAVENUMBER,
        VOXEL_SIZE,
        order,
        1,
        1,
        1,
    )


VOXEL_KERNELS = [
    pytest.param(_voxel_self, VOXEL_FACE, id="self"),
    pytest.param(_voxel_edge, VOXEL_PAIR_EDGE, id="edge"),
    pytest.param(_voxel_vertex, VOXEL_PAIR_VERTEX, id="vertex"),
]


@pytest.mark.parametrize(("kernel", "vertices", "tol"), TRIANGLE_KERNELS)
def test_triangle_kernels_return_nine_edge_pair_terms(kernel, vertices, tol):
    got = kernel(vertices, 8)
    assert got.shape == (9,)
    assert np.all(np.isfinite(got))
    assert np.linalg.norm(got) > 0.0


@pytest.mark.parametrize(("kernel", "vertices", "tol"), TRIANGLE_KERNELS)
def test_triangle_kernels_are_invariant_under_translation(kernel, vertices, tol):
    here = kernel(vertices, 8)
    there = kernel(vertices + SHIFT, 8)
    assert np.linalg.norm(there - here) <= tol * np.linalg.norm(here)


@pytest.mark.parametrize(("kernel", "vertices", "tol"), TRIANGLE_KERNELS)
def test_triangle_kernels_are_invariant_under_rotation(kernel, vertices, tol):
    here = kernel(vertices, 8)
    turned = kernel(vertices @ _rotation(0.7).T, 8)
    assert np.linalg.norm(turned - here) <= tol * np.linalg.norm(here)


@pytest.mark.parametrize(("kernel", "vertices", "tol"), TRIANGLE_KERNELS)
def test_triangle_kernels_converge_as_the_quadrature_order_rises(kernel, vertices, tol):
    """The edge and vertex kernels converge slowly and not monotonically."""
    reference = kernel(vertices, 20)
    scale = np.linalg.norm(reference)
    coarse = np.linalg.norm(kernel(vertices, 3) - reference) / scale
    fine = np.linalg.norm(kernel(vertices, 16) - reference) / scale
    assert fine < coarse / 20.0


def test_triangle_self_reaches_machine_precision_by_order_twelve():
    reference = _triangle_self(TRIANGLE, 20)
    got = _triangle_self(TRIANGLE, 12)
    assert np.linalg.norm(got - reference) <= 1e-12 * np.linalg.norm(reference)


@pytest.mark.parametrize(("kernel", "vertices"), VOXEL_KERNELS)
def test_voxel_kernels_are_invariant_under_translation(kernel, vertices):
    here = kernel(vertices, 8)
    there = kernel(vertices + SHIFT, 8)
    assert abs(there - here) <= 1e-12 * abs(here)


@pytest.mark.parametrize(("kernel", "vertices"), VOXEL_KERNELS)
def test_voxel_kernels_converge_as_the_quadrature_order_rises(kernel, vertices):
    reference = kernel(vertices, 14)
    assert abs(kernel(vertices, 10) - reference) <= 1e-10 * abs(reference)


def test_voxel_kernels_depend_on_the_orientation_of_the_cell():
    """These kernels are written for axis-aligned cells, so the grid stays so.

    The reduced kernel index and the two scalar basis terms are defined against
    the cell's own axes. Turning the geometry while holding those indices turns
    the operator into a different one, which is why the body grid is axis
    aligned and stays that way.
    """
    turn = _rotation(0.7)
    upright = _voxel_self(VOXEL_FACE, 8)
    turned = directfn.voxel_self(
        VOXEL_FACE @ turn.T,
        ((VOXEL_FACE[0] + VOXEL_FACE[2]) / 2) @ turn.T,
        ((VOXEL_FACE[0] + VOXEL_FACE[2]) / 2) @ turn.T,
        FACE_NORMAL @ turn.T,
        FACE_NORMAL @ turn.T,
        VOXEL_WAVENUMBER,
        VOXEL_SIZE,
        8,
        1,
        1,
        1,
    )
    assert abs(turned - upright) > 0.01 * abs(upright)


def test_triangle_kernels_reject_a_vertex_array_of_the_wrong_shape():
    weights, nodes = _rule(6)
    with pytest.raises(ValueError, match=r"vertices must have shape \(3, 3\)"):
        directfn.triangle_self(TRIANGLE_PAIR_EDGE, WAVENUMBER, weights, nodes)


def test_voxel_kernels_reject_a_vertex_array_of_the_wrong_shape():
    with pytest.raises(ValueError, match=r"vertices must have shape \(4, 3\)"):
        _voxel_self(VOXEL_PAIR_EDGE, 8)


def test_triangle_kernels_reject_a_rule_whose_weights_and_nodes_differ_in_length():
    weights, _ = _rule(6)
    _, nodes = _rule(8)
    with pytest.raises(ValueError, match="same length"):
        directfn.triangle_self(TRIANGLE, WAVENUMBER, weights, nodes)
