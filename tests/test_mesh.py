"""The surface mesh, its element geometry and the GMSH 2.2 reader."""

import math

import pytest
import torch

from mariepy.mesh import _FIRST_NODE, SurfaceMesh

RADIUS = 0.05
WIDTH = 0.01

TWO_TRIANGLES = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
7 0.0 0.0 0.0
3 1.0 0.0 0.0
11 1.0 1.0 0.0
5 0.0 1.0 0.0
$EndNodes
$Elements
4
1 15 2 9 9 7
2 1 2 4 4 7 3
3 2 2 1 1 7 3 11
4 2 3 1 1 0 7 11 5
$EndElements
"""


def _loop(device, n_around=12, n_across=2):
    return SurfaceMesh.loop(
        radius=RADIUS, width=WIDTH, n_around=n_around, n_across=n_across, device=device
    )


def _write(tmp_path, text):
    path = tmp_path / "mesh.msh"
    path.write_text(text)
    return path


def test_the_gmsh_reader_renumbers_node_numbers_to_their_position(tmp_path):
    mesh = SurfaceMesh.read_gmsh22(_write(tmp_path, TWO_TRIANGLES))
    assert mesh.n_nodes == 4
    torch.testing.assert_close(
        mesh.nodes[1], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    assert mesh.triangles[0].tolist() == [0, 1, 2]
    assert mesh.triangles[1].tolist() == [0, 2, 3]


def test_the_gmsh_reader_keeps_lines_triangles_and_their_tags_apart(tmp_path):
    mesh = SurfaceMesh.read_gmsh22(_write(tmp_path, TWO_TRIANGLES))
    assert mesh.n_triangles == 2
    assert mesh.lines.tolist() == [[0, 1]]
    assert mesh.line_tags.tolist() == [4]
    assert mesh.triangle_tags.tolist() == [1, 1]


def test_the_gmsh_reader_reads_elements_with_any_number_of_tags(tmp_path):
    mesh = SurfaceMesh.read_gmsh22(_write(tmp_path, TWO_TRIANGLES))
    assert mesh.triangles[1].tolist() == [0, 2, 3]


def test_the_gmsh_reader_rejects_a_file_with_no_element_section(tmp_path):
    truncated = TWO_TRIANGLES.split("$Elements")[0]
    with pytest.raises(ValueError, match=r"no \$Nodes or no \$Elements"):
        SurfaceMesh.read_gmsh22(_write(tmp_path, truncated))


def test_the_gmsh_reader_rejects_an_element_on_an_undefined_node(tmp_path):
    broken = TWO_TRIANGLES.replace("3 2 2 1 1 7 3 11", "3 2 2 1 1 7 3 99")
    with pytest.raises(ValueError, match="undefined node number 99"):
        SurfaceMesh.read_gmsh22(_write(tmp_path, broken))


def test_the_three_edge_vectors_of_a_triangle_sum_to_zero(device):
    mesh = _loop(device)
    total = mesh.edge_vectors().sum(dim=1)
    torch.testing.assert_close(total, torch.zeros_like(total))


def test_the_height_over_each_edge_reproduces_the_triangle_area(device):
    mesh = _loop(device)
    vertices = mesh.vertices()
    edge = mesh.edge_vectors()
    lengths = mesh.edge_lengths()
    opposite = vertices - vertices[:, _FIRST_NODE, :]
    height = (
        torch.linalg.vector_norm(torch.linalg.cross(opposite, edge, dim=-1), dim=-1)
        / lengths
    )
    torch.testing.assert_close(
        0.5 * lengths * height, mesh.areas()[:, None].expand(-1, 3)
    )


def test_the_vectors_from_the_vertices_to_the_barycentre_sum_to_zero(device):
    mesh = _loop(device)
    total = mesh.rho().sum(dim=1)
    torch.testing.assert_close(total, torch.zeros_like(total))


def test_the_areas_of_the_loop_sum_to_the_inscribed_annulus(device):
    n_around = 60
    mesh = _loop(device, n_around=n_around)
    wedge = math.sin(2.0 * math.pi / n_around) / 2.0
    expected = (
        n_around * wedge * ((RADIUS + WIDTH / 2) ** 2 - (RADIUS - WIDTH / 2) ** 2)
    )
    assert float(mesh.areas().sum()) == pytest.approx(expected, rel=1e-12)


def test_the_loop_is_wound_so_that_every_normal_points_along_z(device):
    mesh = _loop(device)
    normals = mesh.normals()
    torch.testing.assert_close(normals[:, 2], torch.ones_like(normals[:, 2]))


def test_aligning_to_lines_puts_each_line_in_cyclic_order_in_its_first_triangle(device):
    mesh = _loop(device)
    for line in mesh.lines:
        first, second = int(line[0]), int(line[1])
        holds = ((mesh.triangles == first).any(dim=1)) & (
            (mesh.triangles == second).any(dim=1)
        )
        earlier = int(torch.nonzero(holds, as_tuple=False).flatten()[0])
        corner = mesh.triangles[earlier]
        at_first = int(torch.nonzero(corner == first, as_tuple=False)[0])
        at_second = int(torch.nonzero(corner == second, as_tuple=False)[0])
        assert (at_second - at_first) % 3 == 1


def test_aligning_to_lines_rejects_a_line_that_is_not_an_interior_edge(device):
    mesh = _loop(device)
    rim = SurfaceMesh(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        triangle_tags=mesh.triangle_tags,
        lines=torch.tensor([[0, 1]], device=mesh.device),
        line_tags=torch.ones(1, dtype=torch.int64, device=mesh.device),
    )
    with pytest.raises(ValueError, match="shared by"):
        rim.align_to_lines()


def test_a_mesh_rejects_a_triangle_naming_a_node_that_does_not_exist(device):
    mesh = _loop(device)
    with pytest.raises(ValueError, match="outside"):
        SurfaceMesh(
            nodes=mesh.nodes,
            triangles=mesh.triangles + mesh.n_nodes,
            triangle_tags=mesh.triangle_tags,
            lines=mesh.lines,
            line_tags=mesh.line_tags,
        )


def test_a_loop_too_coarse_to_close_is_refused(device):
    with pytest.raises(ValueError, match="n_around"):
        SurfaceMesh.loop(
            radius=RADIUS, width=WIDTH, n_around=2, n_across=1, device=device
        )


def test_a_loop_wider_than_its_diameter_is_refused(device):
    with pytest.raises(ValueError, match="width"):
        SurfaceMesh.loop(
            radius=RADIUS, width=4 * RADIUS, n_around=8, n_across=1, device=device
        )
