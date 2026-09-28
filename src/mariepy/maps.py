"""The circular components of a coil's magnetic field over a body, as a file.

A map file is a NumPy ``.npz`` archive, loadable without pickle, that holds,
for every channel of one coil, the two circular components
:func:`mariepy.fields.circular_components` gives at the voxel centres,

    plus = mu0 (Hx + j Hy),    minus = mu0 (Hx - j Hy),

with the time dependence ``exp(+j omega t)``. ``plus`` is therefore twice the
complex amplitude of the part of the field rotating counterclockwise about +z,
and ``minus`` twice that of the part rotating clockwise. Which of them excites
and which receives depends on the nucleus and on the sense of the static field
along z, which the file leaves to its reader.

| Entry | Shape and type | Meaning |
|---|---|---|
| ``plus``, ``minus`` | (Nc, n1, n2, n3), complex64 | Tesla per unit drive, zero outside the body |
| ``mask`` | (n1, n2, n3), bool | The voxels that carry tissue |
| ``metadata`` | JSON string | :data:`METADATA_KEYS` |

Voxel ``(i, j, k)`` is centred at ``origin + resolution * (i, j, k)``, in the
frame ``metadata["frame"]`` names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["METADATA_KEYS", "MapFile", "read", "write"]

METADATA_KEYS = (
    "coil",
    "channels",
    "frequency_hz",
    "drive_unit",
    "origin",
    "resolution",
    "frame",
    "bodies",
    "mariepy_version",
    "data_licence",
)
"""The metadata a map file carries."""


@dataclass(frozen=True)
class MapFile:
    """What one map file holds.

    Attributes
    ----------
    plus, minus
        Shape ``(n_channels, n1, n2, n3)``, complex, in tesla per unit drive.
    mask
        Shape ``(n1, n2, n3)``, boolean.
    metadata
        The keys of :data:`METADATA_KEYS`.
    """

    plus: torch.Tensor
    minus: torch.Tensor
    mask: torch.Tensor
    metadata: dict


def _array(values) -> np.ndarray:
    return values.detach().cpu().numpy() if torch.is_tensor(values) else values


def write(
    path,
    plus,
    minus,
    mask,
    *,
    coil: str,
    channels,
    frequency_hz: float,
    drive_unit: str,
    origin,
    resolution: float,
    frame: str,
    bodies,
    data_licence: str,
) -> None:
    """Write the circular components of every channel of a coil.

    Parameters
    ----------
    path
        File to write, conventionally with a ``.npz`` suffix.
    plus, minus
        Shape ``(n_channels, n1, n2, n3)``, in tesla per unit drive; stored in
        single precision.
    mask
        Shape ``(n1, n2, n3)``, the voxels that carry tissue.
    coil
        Identity string of the coil model.
    channels
        Channel names, in the maps' own order.
    frequency_hz
        Frequency the fields were solved at.
    drive_unit
        What one unit of a channel's drive is.
    origin
        Coordinates of voxel ``(0, 0, 0)``'s centre, in metres.
    resolution
        Voxel pitch in metres.
    frame
        The frame the coordinates are in.
    bodies
        Identifiers of the body models the fields were solved in.
    data_licence
        Licence of the file, set by its body models.

    Raises
    ------
    ValueError
        The two stacks and the mask disagree in shape, or the channel names do
        not match the maps they label.
    """
    from mariepy import __version__

    counter = np.asarray(_array(plus), dtype=np.complex64)
    clockwise = np.asarray(_array(minus), dtype=np.complex64)
    inside = np.asarray(_array(mask), dtype=bool)
    if counter.shape != clockwise.shape or counter.ndim != 4:
        raise ValueError(
            f"plus and minus are (n_channels, n1, n2, n3) alike, got "
            f"{counter.shape} and {clockwise.shape}"
        )
    if counter.shape[1:] != inside.shape:
        raise ValueError(
            f"the maps are {counter.shape[1:]} and the mask {inside.shape}"
        )
    names = [str(name) for name in channels]
    if len(names) != counter.shape[0]:
        raise ValueError(f"{len(names)} channel names label {counter.shape[0]} maps")

    metadata = {
        "coil": str(coil),
        "channels": names,
        "frequency_hz": float(frequency_hz),
        "drive_unit": str(drive_unit),
        "origin": [float(value) for value in origin],
        "resolution": float(resolution),
        "frame": str(frame),
        "bodies": [str(name) for name in bodies],
        "mariepy_version": __version__,
        "data_licence": str(data_licence),
    }
    np.savez_compressed(
        path,
        plus=counter,
        minus=clockwise,
        mask=inside,
        metadata=np.array(json.dumps(metadata)),
    )


def read(path, *, device=None) -> MapFile:
    """Read a file :func:`write` wrote, without unpickling anything.

    Parameters
    ----------
    path
        File to read.
    device
        Device to place the maps on; the CPU by default.

    Returns
    -------
    MapFile
        The maps, the mask and the metadata.

    Raises
    ------
    ValueError
        The archive is missing an entry or a metadata key.
    """
    with np.load(path, allow_pickle=False) as archive:
        missing = {"plus", "minus", "mask", "metadata"} - set(archive.files)
        if missing:
            raise ValueError(f"the archive has no {', '.join(sorted(missing))}")
        metadata = json.loads(str(archive["metadata"]))
        absent = [key for key in METADATA_KEYS if key not in metadata]
        if absent:
            raise ValueError(f"the metadata has no {', '.join(absent)}")
        return MapFile(
            plus=torch.from_numpy(archive["plus"]).to(device),
            minus=torch.from_numpy(archive["minus"]).to(device),
            mask=torch.from_numpy(archive["mask"]).to(device),
            metadata=metadata,
        )
