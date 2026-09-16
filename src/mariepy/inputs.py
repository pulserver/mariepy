"""A MARIE simulation file, read into the body, coil and frequency it names.

Ported from MARIE 3.0's ``src_utils/src_loaders/load_inputs.m`` and the file
paths ``src_geometry/geo_assembly.m`` resolves. A simulation file is a JSON
object in ``<data>/inputs/``; it names a body under ``<data>/bodies/`` and a
surface coil under ``<data>/coils/coil_files/``, whose lumped elements sit
beside it with the same name and a ``.json`` suffix.

Only the case :func:`mariepy.solver.solve` covers is read: a surface coil around
a piecewise-constant body. A file that asks for anything else raises, naming the
milestone that covers it, rather than being solved as something it is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil, read_lumped_elements
from mariepy.constants import Medium
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
    """

    medium: Medium
    body: VoxelBody
    coil: SurfaceCoil


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
        If the file asks for the piecewise-linear basis, a wire coil, an RF
        shield, or a precomputed field basis.
    ValueError
        If the file names no surface coil.
    """
    path = Path(path)
    data = path.parent.parent if data is None else Path(data)
    settings = json.loads(path.read_text())

    if int(settings.get("Basis_Functions_VIE", 0)) != 0:
        raise NotImplementedError(
            f"{path.name} asks for the piecewise-linear body basis, which is "
            "milestone 2"
        )
    if settings.get("ShieldFile"):
        raise NotImplementedError(
            f"{path.name} names an RF shield, which is milestone 2"
        )
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
    elements = read_lumped_elements(
        coil_file.with_suffix(".json"), tmd=bool(settings.get("TMD", 0))
    )

    return Case(
        medium=Medium(float(settings["B0"]), settings.get("Nucleus", "1H")),
        body=VoxelBody.read_marie(
            data / "bodies" / settings["BodyFile"], device=device
        ),
        coil=SurfaceCoil.build(mesh, elements),
    )
