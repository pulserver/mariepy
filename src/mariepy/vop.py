"""Virtual observation points: a small set of matrices that bounds local SAR.

A body model gives one SAR matrix per averaging volume, hundreds of thousands of
them, and a pulse designer needs the largest ``v^H Q v`` over all of them. The
compression of Eichfelder and Gebhardt (MRM 2011, doi 10.1002/mrm.22927)
replaces the set by a few matrices, each dominating a cluster of the originals,
so that the largest ``v^H A v`` over the virtual observation points is never
below the true peak and overestimates it by a bounded amount.

The bound is set by ``margin``, a fraction of the largest eigenvalue in the set:
that is how much spectral norm a cluster's matrix is allowed to add over the
cluster's own worst member.

:func:`write` and :func:`read` carry the points to a pulse designer in the file
``PLAN.md`` calls the output contract: a NumPy archive of the points, one
head-average matrix per body model, and the metadata that says what drive the
matrices are written in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["VopFile", "compress", "dominates", "read", "write"]


def _deficiency(matrix: torch.Tensor) -> torch.Tensor:
    """Give the smallest positive semi-definite ``D`` with ``matrix + D`` still so."""
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * (-values).clamp(min=0)) @ vectors.conj().transpose(-2, -1)


def compress(
    matrices: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a stack of SAR matrices into virtual observation points.

    Parameters
    ----------
    matrices
        Shape ``(n, n_channels, n_channels)``, complex, Hermitian and positive
        semi-definite, as :func:`mariepy.sar.local_matrices` gives them.
    margin
        Overestimation allowed, as a fraction of the largest eigenvalue over the
        whole stack. A cluster grows while the spectral norm its matrix adds
        over the cluster's worst member stays within that.

    Returns
    -------
    vops : torch.Tensor
        Shape ``(n_vops, n_channels, n_channels)``, Hermitian, each dominating
        its cluster.
    cluster : torch.Tensor
        Shape ``(n,)``, which virtual observation point covers each input
        matrix.

    Raises
    ------
    ValueError
        ``matrices`` is not a stack of square matrices, or ``margin`` is
        negative.
    """
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(
            f"the matrices are (n, n_channels, n_channels), got {tuple(matrices.shape)}"
        )
    if margin < 0.0:
        raise ValueError(
            f"the margin is a fraction of the largest eigenvalue, got {margin}"
        )

    allowed = margin * float(torch.linalg.eigvalsh(matrices)[:, -1].max())
    remaining = matrices
    origin = torch.arange(matrices.shape[0], device=matrices.device)
    cluster = torch.empty(matrices.shape[0], dtype=torch.long, device=matrices.device)
    vops = []

    while remaining.shape[0]:
        worst = remaining[int(torch.linalg.eigvalsh(remaining)[:, -1].argmax())]
        gaps = torch.linalg.eigvalsh(worst[None] - remaining)[:, 0]
        order = torch.argsort(gaps, descending=True, stable=True)
        remaining, origin = remaining[order], origin[order]

        added = torch.zeros_like(worst)
        taken = 0
        while taken < remaining.shape[0]:
            trial = added + _deficiency(worst - remaining[taken])
            # The order puts a matrix that ``worst`` already dominates first, so
            # its deficiency is zero and a cluster is never empty -- except at a
            # zero margin, where rounding in that zero decides, which is what
            # ``taken`` guards.
            if taken and float(torch.linalg.eigvalsh(trial)[-1]) > allowed:
                break
            added, taken = trial, taken + 1

        cluster[origin[:taken]] = len(vops)
        vops.append(worst + added)
        remaining, origin = remaining[taken:], origin[taken:]

    return torch.stack(vops), cluster


def dominates(vops: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Give how far each matrix is from being dominated by a set of points.

    ``A`` dominates ``Q`` when ``A - Q`` is positive semi-definite, so that
    ``v^H Q v <= v^H A v`` for every drive. The value returned is the smallest
    eigenvalue of ``A - Q`` for the best ``A``: non-negative where the bound
    holds.

    Parameters
    ----------
    vops
        Shape ``(n_vops, n_channels, n_channels)``.
    matrices
        Shape ``(n, n_channels, n_channels)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n,)``, real.
    """
    gaps = torch.linalg.eigvalsh(vops[:, None] - matrices[None, :])[..., 0]
    return gaps.max(dim=0).values


METADATA_KEYS = (
    "coil",
    "frequency_hz",
    "drive_unit",
    "channels",
    "averaging",
    "bodies",
    "compression_margin",
    "mariepy_version",
    "data_licence",
)
"""The metadata the output contract requires, in the order it lists them."""


@dataclass(frozen=True)
class VopFile:
    """What one virtual-observation-point file holds.

    Attributes
    ----------
    vops
        Shape ``(n_vops, n_channels, n_channels)``, complex, Hermitian, for
        10 g local SAR.
    global_matrix
        Shape ``(n_bodies, n_channels, n_channels)``, the head-average matrix of
        each body model, in the order ``metadata["bodies"]`` names them.
    metadata
        The keys of :data:`METADATA_KEYS`.
    """

    vops: torch.Tensor
    global_matrix: torch.Tensor
    metadata: dict


def _hermitian(matrices, name):
    """Return the matrices as complex128 numpy, refusing a stack that is not Hermitian."""
    array = np.asarray(
        matrices.detach().cpu().numpy() if torch.is_tensor(matrices) else matrices,
        dtype=np.complex128,
    )
    if array.ndim != 3 or array.shape[-1] != array.shape[-2]:
        raise ValueError(f"{name} is (n, n_channels, n_channels), got {array.shape}")
    if not np.allclose(array, np.conj(np.swapaxes(array, -1, -2)), atol=0, rtol=1e-10):
        raise ValueError(f"{name} holds a matrix that is not Hermitian")
    return array


def write(
    path,
    vops,
    global_matrix,
    *,
    coil: str,
    frequency_hz: float,
    drive_unit: str,
    channels,
    averaging: str,
    bodies,
    compression_margin: float,
    data_licence: str,
) -> None:
    """Write the virtual observation points a pulse designer reads.

    Parameters
    ----------
    path
        File to write, conventionally with a ``.npz`` suffix.
    vops
        Shape ``(n_vops, n_channels, n_channels)``, Hermitian.
    global_matrix
        Shape ``(n_bodies, n_channels, n_channels)``, Hermitian, one per body
        model.
    coil
        Identity string of the coil model.
    frequency_hz
        Frequency the fields were solved at.
    drive_unit
        What one unit of a channel's drive is, in the terms the output contract
        allows: an incident root-watt for a port-driven coil, one unit of the
        pattern for a coil defined by fixed current patterns.
    channels
        Channel names, in the matrices' own order.
    averaging
        Target mass and method of the spatial average.
    bodies
        Identifiers of the body models, in ``global_matrix``'s order.
    compression_margin
        Overestimation allowed in the compression, as :func:`compress` takes it.
    data_licence
        Licence of the file, set by its body models.

    Raises
    ------
    ValueError
        A stack is not Hermitian, the two stacks disagree on the channel count,
        or the names do not match the matrices they label.
    """
    from mariepy import __version__

    points = _hermitian(vops, "vops")
    whole = _hermitian(global_matrix, "global_matrix")
    if points.shape[-1] != whole.shape[-1]:
        raise ValueError(
            f"the points carry {points.shape[-1]} channels and the head-average "
            f"matrices {whole.shape[-1]}"
        )
    names = [str(name) for name in channels]
    models = [str(name) for name in bodies]
    if len(names) != points.shape[-1]:
        raise ValueError(
            f"{len(names)} channel names label {points.shape[-1]} channels"
        )
    if len(models) != whole.shape[0]:
        raise ValueError(
            f"{len(models)} body names label {whole.shape[0]} head-average matrices"
        )

    metadata = {
        "coil": str(coil),
        "frequency_hz": float(frequency_hz),
        "drive_unit": str(drive_unit),
        "channels": names,
        "averaging": str(averaging),
        "bodies": models,
        "compression_margin": float(compression_margin),
        "mariepy_version": __version__,
        "data_licence": str(data_licence),
    }
    np.savez_compressed(
        path,
        vops=points,
        global_matrix=whole,
        metadata=np.array(json.dumps(metadata)),
    )


def read(path, *, device=None) -> VopFile:
    """Read a file :func:`write` wrote, without unpickling anything.

    Parameters
    ----------
    path
        File to read.
    device
        Device to place the matrices on; the CPU by default.

    Returns
    -------
    VopFile
        The points, the head-average matrices and the metadata.

    Raises
    ------
    ValueError
        The archive is missing an entry the contract requires.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = {"vops", "global_matrix", "metadata"} - set(archive.files)
        if missing:
            raise ValueError(f"the archive has no {', '.join(sorted(missing))}")
        metadata = json.loads(str(archive["metadata"]))
        absent = [key for key in METADATA_KEYS if key not in metadata]
        if absent:
            raise ValueError(f"the metadata has no {', '.join(absent)}")
        return VopFile(
            vops=torch.from_numpy(archive["vops"]).to(device),
            global_matrix=torch.from_numpy(archive["global_matrix"]).to(device),
            metadata=metadata,
        )
