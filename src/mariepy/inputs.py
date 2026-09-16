"""A MARIE simulation file, read into the body, coil and frequency it names.

Ported from MARIE 3.0's ``src_utils/src_loaders/load_inputs.m`` and the file
paths ``src_geometry/geo_assembly.m`` resolves. A simulation file is a JSON
object in ``<data>/inputs/``; it names a body under ``<data>/bodies/``, a
surface coil under ``<data>/coils/coil_files/``, whose lumped elements sit
beside it with the same name and a ``.json`` suffix, and optionally an RF shield
under ``<data>/coils/shield_files/``, whose lumped elements, if it has any, sit
beside it in the same way.

Only the cases :func:`mariepy.solver.solve` covers are read: a surface coil
around a body in either basis. A file that asks for anything else raises, naming
the milestone that covers it, rather than being solved as something it is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil, read_lumped_elements
from mariepy.constants import Medium
from mariepy.cosim import Network, read_network
from mariepy.mesh import SurfaceMesh

__all__ = ["Case", "read_case"]


@dataclass(frozen=True)
class Case:
    """Everything a MARIE simulation file names, ready for :func:`~mariepy.solver.solve`.

    Attributes
    ----------
    medium
        The field strength and nucleus, hence the frequency.
    body
        The body model.
    coil
        The surface coil, its ports and its lumped elements.
    linear
        Whether the file asks for the piecewise-linear body basis; pass it to
        :func:`~mariepy.solver.solve`.
    shield
        The RF shield the file names, or None; pass it to
        :func:`~mariepy.solver.solve`.
    network
        The coil's ports and lumped values as co-simulation reads them, with
        the file's ``TMD`` flag; pass it to :func:`~mariepy.cosim.co_simulate`.
    """

    medium: Medium
    body: VoxelBody
    coil: SurfaceCoil
    linear: bool = False
    shield: SurfaceCoil | None = None
    network: Network | None = None


def read_case(
    path: str | Path,
    *,
    data: str | Path | None = None,
    device: torch.device | str | None = None,
) -> Case:
    """Read a MARIE simulation file and the body and coil it names.

    Parameters
    ----------
    path
        The JSON simulation file.
    data
        MARIE's data directory, which holds ``bodies/`` and ``coils/``. By
        default, the parent of the directory holding ``path``, as MARIE lays it
        out.
    device
        Device the body and coil are built on.

    Returns
    -------
    Case
        The medium, body and coil.

    Raises
    ------
    NotImplementedError
        If the file asks for a wire coil or a precomputed field basis.
    ValueError
        If the file names no surface coil, or a body basis MARIE does not know.
    """
    path = Path(path)
    data = path.parent.parent if data is None else Path(data)
    settings = json.loads(path.read_text())

    basis = int(settings.get("Basis_Functions_VIE", 0))
    if basis not in (0, 1):
        raise ValueError(f"{path.name} names body basis {basis}; MARIE knows 0 and 1")
    if settings.get("WireFile"):
        raise NotImplementedError(
            f"{path.name} names a wire coil, which is milestone 3"
        )
    if settings.get("BasisFile") and not settings.get("CoilFile"):
        raise NotImplementedError(
            f"{path.name} asks for a precomputed field basis, which is milestone 4"
        )
    coil_name = settings.get("CoilFile")
    if not coil_name:
        raise ValueError(f"{path.name} names no surface coil")

    coil_file = data / "coils" / "coil_files" / coil_name
    mesh = SurfaceMesh.read_gmsh22(coil_file, device=device or "cpu")
    tmd = bool(settings.get("TMD", 0))
    elements = read_lumped_elements(coil_file.with_suffix(".json"), tmd=tmd)

    shield = None
    if settings.get("ShieldFile"):
        shield_file = data / "coils" / "shield_files" / settings["ShieldFile"]
        shield_mesh = SurfaceMesh.read_gmsh22(shield_file, device=device or "cpu")
        shield_elements = ()
        if shield_file.with_suffix(".json").is_file():
            shield_elements = read_lumped_elements(
                shield_file.with_suffix(".json"), tmd=tmd
            )
        shield = SurfaceCoil.build(shield_mesh, shield_elements)

    return Case(
        medium=Medium(float(settings["B0"]), settings.get("Nucleus", "1H")),
        body=VoxelBody.read_marie(
            data / "bodies" / settings["BodyFile"], device=device
        ),
        coil=SurfaceCoil.build(mesh, elements),
        linear=basis == 1,
        shield=shield,
        network=read_network(coil_file.with_suffix(".json"), tmd=tmd),
    )
