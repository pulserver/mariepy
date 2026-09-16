"""The RWG basis on a coil: edges, signs, degrees of freedom, ports and loads."""

import json
import math

import pytest
import torch

from mariepy.coil import Port, SurfaceCoil, pair_tags, read_lumped_elements
from mariepy.mesh import SurfaceMesh

RADIUS = 0.05
WIDTH = 0.01
N_AROUND = 12
N_ACROSS = 2

ELEMENTS = {
    "coil_configuration": {
        "elements": [
            {
                "number": 1,
                "type": "port",
                "load": "capacitorParallel_inductorSeries",
                "value": [1e-12, 2e-9],
                "Q": [200.0, 100.0],
                "optim": {"boolean": 0},
                "cross_talk": None,
                "excitation": {"entity": 1, "TxRx": "Tx"},
            },
            {
                "number": 2,
                "type": "element",
                "load": "mutual_inductor",
                "value": 3.3e-9,
                "Q": 250.0,
                "optim": {"boolean": 0},
                "cross_talk": {"coupled_port": 1, "coupled_value": 5e-9},
                "excitation": {"entity": 2, "TxRx": "Tx"},
            },
        ]
    }
}


def _mesh(device, n_around=N_AROUND, n_across=N_ACROSS):
    return SurfaceMesh.loop(
        radius=RADIUS, width=WIDTH, n_around=n_around, n_across=n_across, device=device
    )


def _port(tag=1, kind="port"):
    return Port(
        tag=tag, kind=kind, load="capacitor", value=1e-12, quality=200.0, voltage=1.0
    )


def _coil(device, **mesh_arguments):
    return SurfaceCoil.build(_mesh(device, **mesh_arguments), (_port(),))


def test_every_interior_edge_carries_exactly_one_basis_function(device):
    coil = _coil(device)
    occurrences = torch.bincount(
        coil.edge_of_triangle.reshape(-1), minlength=coil.n_edges
    )
    assert int((occurrences == 2).sum()) == coil.n_dof
    assert sorted(coil.dof_of_edge[occurrences == 2].tolist()) == list(
        range(coil.n_dof)
    )


def test_an_edge_on_the_rim_carries_no_basis_function(device):
    coil = _coil(device)
    occurrences = torch.bincount(
        coil.edge_of_triangle.reshape(-1), minlength=coil.n_edges
    )
    assert bool((coil.dof_of_edge[occurrences == 1] == -1).all())
    assert int((occurrences == 1).sum()) == 2 * N_AROUND


def test_the_two_triangles_of_a_basis_function_carry_opposite_signs(device):
    coil = _coil(device)
    dof = coil.dof_of_triangle()
    carried = dof >= 0
    total = torch.zeros(coil.n_dof, dtype=torch.int64, device=dof.device)
    total.scatter_add_(0, dof[carried], coil.signs[carried])
    torch.testing.assert_close(total, torch.zeros_like(total))


def test_the_normal_current_of_a_basis_function_is_continuous_across_its_edge(device):
    coil = _coil(device)
    corners = coil.rwg_vertices()
    free_positive, free_negative, first, second = corners.unbind(dim=1)
    along = second - first
    length = torch.linalg.vector_norm(along, dim=-1)
    midpoint = 0.5 * (first + second)

    flux = []
    for free in (free_positive, free_negative):
        arm = midpoint - free
        outward = (
            arm - (arm * along).sum(-1, keepdim=True) / length[:, None] ** 2 * along
        )
        outward = outward / torch.linalg.vector_norm(outward, dim=-1, keepdim=True)
        area = 0.5 * torch.linalg.vector_norm(
            torch.linalg.cross(along, arm, dim=-1), dim=-1
        )
        flux.append(length / (2.0 * area) * (arm * outward).sum(-1))

    torch.testing.assert_close(flux[0], torch.ones_like(flux[0]))
    torch.testing.assert_close(flux[1], torch.ones_like(flux[1]))


def test_the_positive_triangle_of_a_basis_function_is_the_one_whose_edge_runs_forwards(
    device,
):
    coil = _coil(device)
    corners = coil.rwg_vertices()
    dof = coil.dof_of_triangle()
    nodes = coil.mesh.nodes
    triangle, local = torch.nonzero((dof >= 0) & (coil.signs > 0), as_tuple=True)
    free = nodes[coil.mesh.triangles[triangle, local]]
    torch.testing.assert_close(corners[dof[triangle, local], 0], free)


def test_port_degrees_of_freedom_are_numbered_before_every_other_edge(device):
    coil = _coil(device)
    (port,) = coil.ports
    assert port.dofs.tolist() == list(range(N_ACROSS))
    assert coil.n_driven == 1


def test_a_port_spans_one_degree_of_freedom_for_each_edge_across_the_strip(device):
    coil = _coil(device, n_across=3)
    (port,) = coil.ports
    assert port.dofs.numel() == 3


def test_lumped_elements_are_numbered_after_the_driven_ports(device):
    mesh = _mesh(device)
    lines = torch.cat([mesh.lines, mesh.triangles[:1, 1:]])
    tags = torch.tensor([1, 1, 2], dtype=torch.int64, device=mesh.device)
    tagged = SurfaceMesh(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        triangle_tags=mesh.triangle_tags,
        lines=lines,
        line_tags=tags,
    )
    coil = SurfaceCoil.build(tagged, (_port(tag=2, kind="element"), _port(tag=1)))
    driven, load = coil.ports
    assert driven.tag == 1
    assert driven.dofs.tolist() == [0, 1]
    assert load.tag == 2
    assert load.dofs.tolist() == [2]


def test_the_adjacency_classes_partition_every_pair_of_triangles(device):
    coil = _coil(device)
    n_triangles = coil.mesh.n_triangles
    shared = coil.shared_vertices(torch.arange(n_triangles, device=coil.mesh.device))
    below = torch.tril(torch.ones_like(shared, dtype=torch.bool), diagonal=-1)
    assert coil.adjacency.edge.shape[0] == int(((shared == 2) & below).sum())
    assert coil.adjacency.vertex.shape[0] == int(((shared == 1) & below).sum())
    assert int((torch.diagonal(shared) == 3).sum()) == n_triangles


def test_an_edge_adjacent_pair_is_the_pair_of_one_interior_edge(device):
    coil = _coil(device)
    occurrences = torch.bincount(
        coil.edge_of_triangle.reshape(-1), minlength=coil.n_edges
    )
    assert coil.adjacency.edge.shape[0] == int((occurrences == 2).sum())
    assert bool((coil.adjacency.edge[:, 0] < coil.adjacency.edge[:, 1]).all())


def test_the_edge_of_a_degree_of_freedom_gives_back_its_own_length(device):
    coil = _coil(device)
    pairs = coil.edges[coil.edge_of_dof]
    nodes = coil.mesh.nodes
    expected = torch.linalg.vector_norm(nodes[pairs[:, 1]] - nodes[pairs[:, 0]], dim=-1)
    torch.testing.assert_close(coil.dof_lengths(), expected)
    corners = coil.rwg_vertices()
    torch.testing.assert_close(
        torch.linalg.vector_norm(corners[:, 3] - corners[:, 2], dim=-1), expected
    )


def _tags(*values):
    return torch.tensor(values, dtype=torch.int64)


def test_elements_numbered_as_the_tags_keep_their_own_tag_whatever_the_file_order():
    paired = pair_tags((_port(tag=2), _port(tag=1)), _tags(1, 1, 2))
    assert [port.tag for port in paired] == [2, 1]


def test_elements_numbered_apart_from_the_tags_take_them_in_file_order():
    elements = (
        _port(tag=1),
        Port(
            tag=2,
            kind="element",
            load="mutual_inductor",
            value=1e-9,
            quality=100.0,
            voltage=0.0,
            coupled_tag=1,
            coupled_value=2e-10,
        ),
    )
    first, second = pair_tags(elements, _tags(1002, 1001, 1002, 0))
    assert (first.tag, second.tag) == (1001, 1002)
    assert second.coupled_tag == 1001


@pytest.mark.parametrize(
    ("numbers", "tags"),
    [((2, 1), (1001, 1002)), ((1, 2), (1001,)), ((1,), (1001, 1002))],
    ids=["out of order", "more elements than tags", "fewer elements than tags"],
)
def test_elements_that_neither_are_the_tags_nor_list_one_per_tag_in_order_are_refused(
    numbers, tags
):
    with pytest.raises(ValueError, match="cannot pair"):
        pair_tags(tuple(_port(tag=number) for number in numbers), _tags(*tags))


def test_a_mutual_inductor_whose_partner_the_file_does_not_define_is_refused():
    lonely = Port(
        tag=1,
        kind="element",
        load="mutual_inductor",
        value=1e-9,
        quality=100.0,
        voltage=0.0,
        coupled_tag=9,
        coupled_value=2e-10,
    )
    with pytest.raises(ValueError, match="partner 9"):
        pair_tags((lonely,), _tags(1))


def test_building_a_coil_rejects_an_element_whose_tag_sits_on_the_rim(device):
    mesh = _mesh(device)
    rim = SurfaceCoil.build(mesh)
    occurrences = torch.bincount(rim.edge_of_triangle.reshape(-1))
    edge = rim.edges[int(torch.nonzero(occurrences == 1)[0])]
    on_rim = SurfaceMesh(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        triangle_tags=mesh.triangle_tags,
        lines=edge[None],
        line_tags=torch.tensor([1], dtype=torch.int64, device=mesh.device),
    )
    with pytest.raises(ValueError, match="names no interior edge"):
        SurfaceCoil.build(on_rim, (_port(tag=1),))


def test_building_a_coil_rejects_a_surface_with_a_non_manifold_edge(device):
    mesh = _mesh(device)
    doubled = SurfaceMesh(
        nodes=mesh.nodes,
        triangles=torch.cat([mesh.triangles, mesh.triangles[:1]]),
        triangle_tags=torch.cat([mesh.triangle_tags, mesh.triangle_tags[:1]]),
        lines=mesh.lines,
        line_tags=mesh.line_tags,
    )
    with pytest.raises(ValueError, match="more than two triangles"):
        SurfaceCoil.build(doubled)


def test_a_lumped_element_file_becomes_ports_carrying_its_tags_and_values(tmp_path):
    path = tmp_path / "elements.json"
    path.write_text(json.dumps(ELEMENTS))
    driven, load = read_lumped_elements(path)
    assert (driven.tag, driven.kind, driven.voltage) == (1, "port", 1.0)
    assert driven.value == pytest.approx(1e-12)
    assert (load.tag, load.kind, load.voltage) == (2, "element", 0.0)
    assert (load.coupled_tag, load.coupled_value) == (1, 5e-9)


@pytest.mark.parametrize(
    "cross_talk",
    [{"coupled_port": 1, "coupled_value": 5e-9}, [1, 5e-9]],
    ids=["object", "list"],
)
def test_a_mutual_inductor_names_its_partner_in_either_form_marie_writes(
    tmp_path, cross_talk
):
    written = json.loads(json.dumps(ELEMENTS))
    written["coil_configuration"]["elements"][1]["cross_talk"] = cross_talk
    path = tmp_path / "elements.json"
    path.write_text(json.dumps(written))
    _, load = read_lumped_elements(path)
    assert (load.coupled_tag, load.coupled_value) == (1, 5e-9)


def test_a_mutual_inductor_that_names_no_partner_is_refused(tmp_path):
    written = json.loads(json.dumps(ELEMENTS))
    written["coil_configuration"]["elements"][1]["cross_talk"] = []
    path = tmp_path / "elements.json"
    path.write_text(json.dumps(written))
    with pytest.raises(ValueError, match="names no partner"):
        read_lumped_elements(path)


def test_tuning_turns_every_tunable_element_into_a_driven_port(tmp_path):
    written = json.loads(json.dumps(ELEMENTS))
    written["coil_configuration"]["elements"][1]["optim"] = {"boolean": 1}
    path = tmp_path / "elements.json"
    path.write_text(json.dumps(written))
    assert [port.kind for port in read_lumped_elements(path)] == ["port", "element"]
    tuned = read_lumped_elements(path, tmd=True)
    assert [(port.kind, port.voltage) for port in tuned] == [
        ("port", 1.0),
        ("port", 1.0),
    ]


def test_a_lumped_element_file_with_an_unknown_type_is_refused(tmp_path):
    broken = json.loads(json.dumps(ELEMENTS))
    broken["coil_configuration"]["elements"][0]["type"] = "antenna"
    path = tmp_path / "elements.json"
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="unknown type"):
        read_lumped_elements(path)


@pytest.mark.parametrize(
    ("load", "value", "quality"),
    [("resistor", 2.0, 1.0), ("inductor", 5e-9, 120.0), ("capacitor", 3e-12, 200.0)],
)
def test_a_lumped_load_carries_its_reactance_and_the_loss_its_quality_factor_implies(
    load, value, quality
):
    omega = 2.0 * math.pi * 297.2e6
    port = Port(
        tag=1, kind="element", load=load, value=value, quality=quality, voltage=0.0
    )
    impedance, loss = port.impedance(omega)
    assert impedance.real == pytest.approx(loss)
    if load == "resistor":
        assert impedance.imag == pytest.approx(0.0)
    else:
        assert abs(impedance.imag) == pytest.approx(quality * loss)


def test_an_unknown_lumped_load_is_refused():
    port = Port(
        tag=1, kind="element", load="memristor", value=1.0, quality=1.0, voltage=0.0
    )
    with pytest.raises(ValueError, match="unknown lumped load"):
        port.impedance(1.0)
