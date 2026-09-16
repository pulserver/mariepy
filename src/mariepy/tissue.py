"""Tissue properties, and the body a labelled volume makes.

A head model arrives as a volume of tissue labels. What the solver and the SAR
matrices want instead is three grids -- relative permittivity, conductivity and
mass density -- so each label carries a record of the properties that make them.

Permittivity and conductivity come from the four-Cole-Cole dispersion of Gabriel,
Lau and Gabriel 1996 (*Phys. Med. Biol.* **41** 2271), evaluated at the coil's
frequency:

    eps*(w) = eps_inf + sum_n dele_n / (1 + (j w tau_n) ** (1 - alpha_n))
              + sigma_i / (j w eps_0),

whose real part is the relative permittivity and whose imaginary part carries the
conductivity, ``sigma = -w eps_0 Im eps*``. The parameters of that model are
measured numbers, so they are read from a file rather than written here: a table
is the head-model pipeline's output alongside the labels it names, and
:func:`read_table` states the columns it must carry.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from mariepy.body import VoxelBody
from mariepy.constants import Medium

__all__ = [
    "GABRIEL_COLUMNS",
    "LabelledBody",
    "Tissue",
    "build",
    "dielectric",
    "read_table",
]

GABRIEL_COLUMNS = (
    "ef",
    "del1",
    "tau1",
    "alf1",
    "del2",
    "tau2",
    "alf2",
    "del3",
    "tau3",
    "alf3",
    "del4",
    "tau4",
    "alf4",
    "sig",
)
"""The dispersion parameters of one tissue, in the order the 1996 paper prints them."""

_TIME_SCALE = (1e-12, 1e-9, 1e-6, 1e-3)
"""The paper gives the four relaxation times in ps, ns, us and ms."""


@dataclass(frozen=True)
class Tissue:
    """What one label is made of.

    Attributes
    ----------
    name
        The tissue's name, as the segmentation calls it.
    dispersion
        The fourteen parameters of :data:`GABRIEL_COLUMNS`, with every
        relaxation time in seconds, shape ``(14,)``.
    density
        Mass density in kg/m^3.
    """

    name: str
    dispersion: torch.Tensor
    density: float


@dataclass(frozen=True)
class LabelledBody:
    """A body built from a label volume, and the masses of its voxels.

    Attributes
    ----------
    body
        The grid, its permittivity and its conductivity at one frequency.
    density
        Mass density in kg/m^3 over the same grid, zero outside the body.
    tissues
        The label each voxel carries, same shape.
    """

    body: VoxelBody
    density: torch.Tensor
    tissues: torch.Tensor


def dielectric(
    dispersion: torch.Tensor, frequency: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the four-Cole-Cole model at one frequency.

    Parameters
    ----------
    dispersion
        Shape ``(..., 14)``, the parameters of :data:`GABRIEL_COLUMNS` with
        every relaxation time in seconds.
    frequency
        Frequency in Hz.

    Returns
    -------
    permittivity : torch.Tensor
        Relative permittivity, shape ``(...)``.
    conductivity : torch.Tensor
        Conductivity in S/m, the same shape.

    Raises
    ------
    ValueError
        ``dispersion`` does not end in fourteen parameters, or the frequency is
        not positive.
    """
    if dispersion.shape[-1] != len(GABRIEL_COLUMNS):
        raise ValueError(
            f"a tissue carries {len(GABRIEL_COLUMNS)} parameters, got "
            f"{dispersion.shape[-1]}"
        )
    if frequency <= 0.0:
        raise ValueError(
            f"the model is written for a positive frequency, got {frequency}"
        )
    omega = 2.0 * torch.pi * frequency
    value = dispersion[..., 0].to(torch.complex128)
    for term in range(4):
        strength = dispersion[..., 1 + 3 * term]
        relaxation = dispersion[..., 2 + 3 * term]
        broadening = dispersion[..., 3 + 3 * term]
        pole = (1j * omega * relaxation.to(torch.complex128)) ** (1.0 - broadening)
        value = value + strength / (1.0 + pole)
    vacuum = Medium(1.0).permittivity
    value = value - 1j * dispersion[..., 13] / (omega * vacuum)
    return value.real, -omega * vacuum * value.imag


def read_table(path) -> dict[int, Tissue]:
    """Read the tissue table a label volume is named against.

    The file is a comma-separated table whose header names, in any order,
    ``label``, ``name``, ``density`` and the fourteen columns of
    :data:`GABRIEL_COLUMNS`. The relaxation times are in the paper's own units:
    ``tau1`` in ps, ``tau2`` in ns, ``tau3`` in us and ``tau4`` in ms.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    dict
        One :class:`Tissue` per label.

    Raises
    ------
    ValueError
        A column is missing, or a label appears twice.
    """
    wanted = {"label", "name", "density", *GABRIEL_COLUMNS}
    table: dict[int, Tissue] = {}
    with Path(path).open(newline="") as handle:
        rows = csv.DictReader(handle)
        missing = wanted - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"the table has no {', '.join(sorted(missing))} column")
        for row in rows:
            label = int(row["label"])
            if label in table:
                raise ValueError(f"label {label} appears twice")
            values = [float(row[column]) for column in GABRIEL_COLUMNS]
            for term, scale in enumerate(_TIME_SCALE):
                values[2 + 3 * term] *= scale
            table[label] = Tissue(
                name=row["name"],
                dispersion=torch.tensor(values, dtype=torch.float64),
                density=float(row["density"]),
            )
    if not table:
        raise ValueError("the table names no tissue")
    return table


def build(
    labels: torch.Tensor,
    table: dict[int, Tissue],
    medium: Medium,
    resolution: float,
    *,
    origin: tuple[float, float, float] | None = None,
    background=(0,),
) -> LabelledBody:
    """Turn a label volume into a body at one frequency.

    Every label in the volume must be one the table names or one ``background``
    names, because a tissue left out of the table would quietly become air and
    take its share of the SAR with it.

    Parameters
    ----------
    labels
        The tissue label of every voxel, shape ``(n1, n2, n3)``, integer.
    table
        From :func:`read_table`.
    medium
        Supplies the frequency the properties are evaluated at.
    resolution
        Voxel pitch in metres.
    origin
        Coordinates of voxel ``(0, 0, 0)``; the grid is centred by default.
    background
        The labels that mean free space.

    Returns
    -------
    LabelledBody
        The body, its densities and the labels they came from.

    Raises
    ------
    ValueError
        A label is neither in the table nor background, or no voxel carries a
        label the table names.
    """
    shape = tuple(labels.shape)
    present = {int(value) for value in torch.unique(labels)}
    unnamed = sorted(present - set(table) - set(background))
    if unnamed:
        raise ValueError(
            f"the volume carries labels the table does not name: "
            f"{', '.join(str(label) for label in unnamed)}"
        )
    device = labels.device
    permittivity = torch.ones(shape, dtype=torch.float64, device=device)
    conductivity = torch.zeros(shape, dtype=torch.float64, device=device)
    density = torch.zeros(shape, dtype=torch.float64, device=device)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)

    for label, tissue in table.items():
        here = labels == label
        if not bool(here.any()):
            continue
        relative, sigma = dielectric(tissue.dispersion.to(device), medium.frequency)
        permittivity[here] = relative
        conductivity[here] = sigma
        density[here] = tissue.density
        mask |= here

    if not bool(mask.any()):
        raise ValueError("no voxel carries a label the table names")
    if origin is None:
        origin = tuple(-0.5 * (size - 1) * resolution for size in shape)
    return LabelledBody(
        body=VoxelBody(
            permittivity=permittivity,
            conductivity=conductivity,
            mask=mask,
            resolution=resolution,
            origin=origin,
        ),
        density=density,
        tissues=labels,
    )
