"""Virtual observation points: a small set of matrices that bounds local SAR.

A body model gives one SAR matrix per averaging volume, hundreds of thousands of
them, and a pulse designer needs the largest ``v^H Q v`` over all of them. The
compression of Eichfelder and Gebhardt (MRM 2011, doi 10.1002/mrm.22927)
replaces the set by a few matrices, each dominating a cluster of the originals,
so that the largest ``v^H A v`` over the virtual observation points is never
below the true peak and overestimates it by a bounded amount.

The bound is set by ``margin``, a fraction of the largest eigenvalue in the set.
Each point is its cluster's strongest matrix plus that much of the identity, so
it overestimates every drive's SAR by at most the margin, and its cluster is
every matrix left that it dominates.

:func:`write` and :func:`read` carry the points to a pulse designer in the file
``PLAN.md`` calls the output contract: a NumPy archive of the points, one
head-average matrix per body model, and the metadata that says what drive the
matrices are written in and what safety factor every local SAR read from them
carries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["VopFile", "compress", "dominates", "read", "write"]


_log = logging.getLogger(__name__)

# Matrices taken in one call. A stack of a population's cubes is hundreds of
# thousands, and a batched Hermitian decomposition on the accelerator asks for
# workspace per matrix that such a stack cannot be given: the eigenvalues are
# taken on the processor, and the accelerator only factors.
CHUNK = 1 << 17


def _largest(matrices: torch.Tensor) -> torch.Tensor:
    """Largest eigenvalue of every matrix of a stack, on the processor."""
    return torch.cat(
        [
            torch.linalg.eigvalsh(matrices[start : start + CHUNK].cpu())[:, -1]
            for start in range(0, matrices.shape[0], CHUNK)
        ]
    )


def _dominated(point: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Tell which matrices of a stack ``point`` dominates, in double precision.

    ``point - Q`` is positive definite exactly when its Cholesky factorisation
    exists, which costs a fraction of an eigenvalue sweep and runs batched on
    either device.
    """
    return torch.cat(
        [
            torch.linalg.cholesky_ex(
                point - matrices[start : start + CHUNK].to(point.dtype)
            ).info
            == 0
            for start in range(0, matrices.shape[0], CHUNK)
        ]
    )


def compress(
    matrices: torch.Tensor, margin: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress a stack of SAR matrices into virtual observation points.

    Parameters
    ----------
    matrices
        Shape ``(n, n_channels, n_channels)``, complex, Hermitian and positive
        semi-definite, as :func:`mariepy.sar.local_matrices` gives them. The
        points are found on the device the stack is on.
    margin
        Overestimation allowed, as a fraction of the largest eigenvalue over the
        whole stack. Each point is the strongest matrix left plus this much of
        the identity.

    Returns
    -------
    vops : torch.Tensor
        Shape ``(n_vops, n_channels, n_channels)``, complex128, Hermitian, each
        dominating its cluster.
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

    device = matrices.device
    # Every matrix keeps the largest eigenvalue it started with: the loop only
    # takes matrices out of the stack, so it is taken once rather than once per
    # point.
    largest = _largest(matrices).to(device)
    allowed = margin * float(largest.max())
    identity = allowed * torch.eye(
        matrices.shape[-1], dtype=torch.complex128, device=device
    )
    remaining = matrices
    origin = torch.arange(matrices.shape[0], device=device)
    cluster = torch.empty(matrices.shape[0], dtype=torch.long, device=device)
    vops = []

    while remaining.shape[0]:
        strongest = int(largest.argmax())
        point = remaining[strongest].to(torch.complex128) + identity
        taken = _dominated(point, remaining)
        # At a zero margin the strongest matrix minus itself is singular, and
        # the factorisation refuses it.
        taken[strongest] = True

        cluster[origin[taken]] = len(vops)
        vops.append(point)
        remaining, origin, largest = remaining[~taken], origin[~taken], largest[~taken]
        _log.info(
            "point %d covers %d matrices, %d left",
            len(vops),
            int(taken.sum()),
            remaining.shape[0],
        )

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
    "safety_factor",
    "safety_basis",
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
        The keys of :data:`METADATA_KEYS`, and ``transmit`` where the file was
        written for one transmit configuration.
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
    safety_factor: float,
    safety_basis: str,
    data_licence: str,
    transmit: str | None = None,
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
        pattern for a coil defined by fixed current patterns. A file a scanner
        reads states the drive as the scanner programs it, channel by channel.
    channels
        Channel names, in the matrices' own order.
    averaging
        Target mass and method of the spatial average.
    bodies
        Identifiers of the body models, in ``global_matrix``'s order.
    compression_margin
        Overestimation allowed in the compression, as :func:`compress` takes it.
    safety_factor
        Factor, at least 1, every local SAR read from the file is multiplied
        by, for what the model does not carry: the coil model's error against
        measurement, anatomy outside the population and positioning. A factor
        on the matrices themselves would cancel in a ratio of two drives'
        SAR, so it travels as metadata.
    safety_basis
        One line stating what ``safety_factor`` covers and where it comes from.
    data_licence
        Licence of the file, set by its body models.
    transmit
        Identity of the transmit configuration the file is valid for, as the
        scanner reports it; a scanner refuses a file written for another.

    Raises
    ------
    ValueError
        A stack is not Hermitian, the two stacks disagree on the channel count,
        the safety factor is below 1 or has no basis, or the names do not match
        the matrices they label.
    """
    from mariepy import __version__

    points = _hermitian(vops, "vops")
    whole = _hermitian(global_matrix, "global_matrix")
    if points.shape[-1] != whole.shape[-1]:
        raise ValueError(
            f"the points carry {points.shape[-1]} channels and the head-average "
            f"matrices {whole.shape[-1]}"
        )
    if not (np.isfinite(safety_factor) and safety_factor >= 1.0):
        raise ValueError(f"the safety factor is at least 1, got {safety_factor}")
    if not str(safety_basis).strip():
        raise ValueError("the safety factor needs a basis stating what it covers")
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
        "safety_factor": float(safety_factor),
        "safety_basis": str(safety_basis),
        "mariepy_version": __version__,
        "data_licence": str(data_licence),
    }
    if transmit is not None:
        metadata["transmit"] = str(transmit)
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
