"""A MARIE simulation file, read into the body, coil and frequency it names."""

import json

import numpy as np
import pytest

from mariepy import cosim
from mariepy.constants import Medium
from mariepy.inputs import read_case
from mariepy.mesh import SurfaceMesh
from mariepy.solver import solve
from mariepy.wire import WireCoil

from .marie_files import write_gmsh22, write_marie_body, write_wire_gmsh22

ELEMENTS = {
    "coil_configuration": {
        "elements": [
            {
                "number": 1,
                "type": "port",
                "load": "capacitorParallel",
                "value": [1e-12],
                "Q": [200.0],
                "optim": {"boolean": 0},
                "cross_talk": [],
                "excitation": {"entity": 1, "TxRx": "Tx"},
            },
        ]
    }
}

SETTINGS = {
    "B0": 7.0,
    "Nucleus": "1H",
    "Basis_Functions_VIE": 0,
    "BodyFile": "Scatterers/1cm/tiny.mat",
    "CoilFile": "Loop/loop.msh",
    "WireFile": "",
    "ShieldFile": "",
    "SurfaceBasisSupportFile": "",
    "TMD": 0,
    "BasisFile": "",
}


def _data(tmp_path, **changes):
    """Lay out a MARIE data directory holding one case, and return its input file."""
    data = tmp_path / "data"
    (data / "inputs").mkdir(parents=True)
    (data / "bodies" / "Scatterers" / "1cm").mkdir(parents=True)
    (data / "coils" / "coil_files" / "Loop").mkdir(parents=True)

    tissue = np.zeros((3, 3, 3), dtype=bool)
    tissue[1, 1, 1] = True
    write_marie_body(
        data / "bodies" / "Scatterers" / "1cm" / "tiny.mat",
        np.where(tissue, 52.0, 1.0),
        np.where(tissue, 0.55, 0.0),
        pitch=0.01,
        origin=(-0.01, -0.01, -0.01),
        tissue=tissue,
    )

    mesh = SurfaceMesh.loop(radius=0.05, width=0.01, n_around=8, n_across=1)
    write_gmsh22(data / "coils" / "coil_files" / "Loop" / "loop.msh", mesh)
    (data / "coils" / "coil_files" / "Loop" / "loop.json").write_text(
        json.dumps(ELEMENTS)
    )

    path = data / "inputs" / "case.json"
    path.write_text(json.dumps({**SETTINGS, **changes}))
    return path


def test_a_marie_simulation_file_gives_the_body_the_coil_and_the_frequency_it_names(
    tmp_path,
):
    case = read_case(_data(tmp_path))
    assert case.medium.frequency == pytest.approx(Medium(7.0, "1H").frequency)
    assert case.body.shape == (3, 3, 3)
    assert case.body.n_voxels == 1
    assert case.body.resolution == pytest.approx(0.01)
    assert case.coil.n_driven == 1
    assert case.coil.ports[0].dofs.numel() > 0
    assert [t.number for t in case.network.terminals] == [1]
    assert case.network.roles == {"Tx"}
    assert not case.network.tmd


def test_a_case_read_from_marie_files_solves(tmp_path):
    """The reader hands the solver what it takes; low orders keep this cheap."""
    case = read_case(_data(tmp_path))
    result = solve(
        case.body, case.coil, case.medium, far_order=2, medium_order=2, near_order=4
    )
    assert all(residual <= 1e-5 for residual in result.ports.residual)
    assert float(result.impedance[0, 0].real) > 0.0

    closed = cosim.co_simulate(
        case.network, result.admittance, case.medium.angular_frequency
    )
    assert closed.values == ((1e-12,),)
    voltage = closed.transmit[:, 0]
    taken = 0.5 * float((voltage.conj() @ result.admittance.cpu() @ voltage).real)
    accepted = 0.5 * (1 - abs(complex(closed.scattering[0, 0])) ** 2)
    assert taken <= accepted * (1 + 1e-9)


def test_the_data_directory_can_be_named_apart_from_the_input_file(tmp_path):
    path = _data(tmp_path)
    moved = tmp_path / "elsewhere.json"
    moved.write_text(path.read_text())
    assert read_case(moved, data=tmp_path / "data").body.n_voxels == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"WireFile": "Loop/wire.msh"}, "wire coil and a surface coil"),
        ({"CoilFile": "", "BasisFile": "basis.mat"}, "milestone 4"),
    ],
    ids=["wire and surface coil", "field basis only"],
)
def test_a_simulation_file_this_port_does_not_cover_is_refused_by_name(
    tmp_path, changes, reason
):
    with pytest.raises(NotImplementedError, match=reason):
        read_case(_data(tmp_path, **changes))


def test_a_simulation_file_that_names_no_coil_is_refused(tmp_path):
    with pytest.raises(ValueError, match="names no coil"):
        read_case(_data(tmp_path, CoilFile=""))


def test_a_wire_coil_the_simulation_file_names_is_read_with_its_ports(tmp_path):
    path = _data(tmp_path, CoilFile="", WireFile="Loop/wire.msh")
    wires = path.parent.parent / "coils" / "wire_files" / "Loop"
    wires.mkdir(parents=True)
    write_wire_gmsh22(wires / "wire.msh", n_segments=12, port_nodes=[0])
    (wires / "wire.json").write_text(json.dumps(ELEMENTS))
    case = read_case(path)
    assert isinstance(case.coil, WireCoil)
    assert case.coil.n_dof == 12
    assert case.coil.ports[0].dofs.tolist() == [0, 1]
    assert case.network.roles == {"Tx"}


@pytest.mark.parametrize(("basis", "linear"), [(0, False), (1, True)])
def test_the_simulation_file_names_the_body_basis(tmp_path, basis, linear):
    assert read_case(_data(tmp_path, Basis_Functions_VIE=basis)).linear is linear


def test_a_body_basis_marie_does_not_know_is_refused(tmp_path):
    with pytest.raises(ValueError, match="body basis 2"):
        read_case(_data(tmp_path, Basis_Functions_VIE=2))


def test_a_shield_the_simulation_file_names_is_read_with_its_basis(tmp_path):
    path = _data(tmp_path, ShieldFile="Sphere/shield.msh")
    folder = tmp_path / "data" / "coils" / "shield_files" / "Sphere"
    folder.mkdir(parents=True)
    write_gmsh22(folder / "shield.msh", SurfaceMesh.sphere(radius=0.08, subdivisions=1))
    case = read_case(path)
    assert case.shield is not None
    assert case.shield.n_dof == 120
    assert case.shield.n_driven == 0


def test_a_simulation_file_without_a_shield_reads_none(tmp_path):
    assert read_case(_data(tmp_path)).shield is None
