"""A body model on a uniform Cartesian grid, and its dielectric contrast.

Ported from MARIE 3.0's ``src_geometry/body_geometry/geo_body_domain.m``,
``grid3d.m``, ``src_physics/src_electromagnetism/em_assembly.m`` and
``src_utils/src_loaders/update_RHBM.m``.

The volume integral equation is solved for the polarisation current in the
voxels the mask selects. A field over the grid has shape
``(..., 3, n1, n2, n3)``, with the three spatial axes last so that an FFT over
them batches whatever leads; the solution vector that GMRES sees is that field
restricted to the mask and flattened, of length ``3 * n_voxels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mariepy.constants import Medium

__all__ = ["Contrast", "VoxelBody"]


@dataclass(frozen=True)
class Contrast:
    """The dielectric contrast of a body at one frequency.

    Attributes
    ----------
    relative
        ``Mr``, the complex relative permittivity ``eps_r + sigma / (j omega
        eps_0)``, over the whole grid.
    scattering
        ``Mc``, that less one: the contrast that drives the scattered field.
    reduced
        ``Mcr``, ``Mc / Mr``, which the Galerkin form of the operator carries.
    """

    relative: torch.Tensor
    scattering: torch.Tensor
    reduced: torch.Tensor


@dataclass(frozen=True)
class VoxelBody:
    """A piecewise-constant body on a uniform grid.

    Attributes
    ----------
    permittivity
        Relative permittivity, shape ``(n1, n2, n3)``, real.
    conductivity
        Conductivity in S/m, same shape.
    mask
        Which voxels carry tissue, same shape, boolean.
    resolution
        Grid pitch in metres.
    origin
        Coordinates of voxel ``(0, 0, 0)`` in metres.
    """

    permittivity: torch.Tensor
    conductivity: torch.Tensor
    mask: torch.Tensor
    resolution: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Check that the three grids agree and the pitch is positive."""
        shapes = {
            tuple(self.permittivity.shape),
            tuple(self.conductivity.shape),
            tuple(self.mask.shape),
        }
        if len(shapes) != 1:
            raise ValueError(
                f"permittivity, conductivity and mask must share a shape; got {shapes}"
            )
        if self.permittivity.ndim != 3:
            raise ValueError(
                f"a body lives on a three-dimensional grid, got "
                f"{self.permittivity.ndim} axes"
            )
        if self.resolution <= 0.0:
            raise ValueError(f"the grid pitch must be positive, got {self.resolution}")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the grid shape."""
        n1, n2, n3 = self.permittivity.shape
        return n1, n2, n3

    @property
    def n_voxels(self) -> int:
        """Return how many voxels the mask selects."""
        return int(self.mask.sum().item())

    @property
    def n_dof(self) -> int:
        """Return the length of the solution vector, three per masked voxel."""
        return 3 * self.n_voxels

    @property
    def device(self) -> torch.device:
        """Return the device the grids live on."""
        return self.permittivity.device

    def coordinates(self) -> torch.Tensor:
        """Return the voxel centres, shape ``(3, n1, n2, n3)``.

        Returns
        -------
        torch.Tensor
            Cartesian coordinates in metres, the component axis first.
        """
        axes = [
            self.origin[axis]
            + self.resolution
            * torch.arange(n, device=self.device, dtype=self.permittivity.dtype)
            for axis, n in enumerate(self.shape)
        ]
        grids = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grids, dim=0)

    def contrast(self, medium: Medium) -> Contrast:
        """Return the dielectric contrast at the medium's frequency.

        Parameters
        ----------
        medium
            Supplies the angular frequency and the permittivity of free space.

        Returns
        -------
        Contrast
            ``Mr``, ``Mc`` and ``Mcr`` over the whole grid, complex. Outside the
            mask the permittivity is 1 and the conductivity 0, so ``Mc`` is zero
            there and ``Mcr`` with it.
        """
        scaling = torch.tensor(
            medium.electric_scaling, device=self.device, dtype=torch.complex128
        )
        relative = self.permittivity.to(torch.complex128) + (
            self.conductivity.to(torch.complex128) / scaling
        )
        scattering = relative - 1.0
        reduced = scattering / relative
        return Contrast(relative=relative, scattering=scattering, reduced=reduced)

    def to_dof(self, field: torch.Tensor) -> torch.Tensor:
        """Restrict a field over the grid to the solution vector.

        Parameters
        ----------
        field
            Shape ``(..., 3, n1, n2, n3)`` for the constant basis, or
            ``(..., 12, n1, n2, n3)`` for the linear basis.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_components * n_voxels)``, the components in order.
        """
        self._check_field(field)
        leading = field.shape[:-4]
        n_components = field.shape[-4]
        flat = field.reshape(*leading, n_components, -1)
        return flat[..., self.mask.reshape(-1)].reshape(
            *leading, n_components * self.n_voxels
        )

    def from_dof(self, vector: torch.Tensor) -> torch.Tensor:
        """Spread a solution vector back over the grid, zero outside the mask.

        Parameters
        ----------
        vector
            Shape ``(..., 3 * n_voxels)`` for the constant basis, or
            ``(..., 12 * n_voxels)`` for the linear basis.

        Returns
        -------
        torch.Tensor
            Shape ``(..., 3, n1, n2, n3)`` or ``(..., 12, n1, n2, n3)``.
        """
        n_components, remainder = divmod(vector.shape[-1], self.n_voxels)
        if remainder or n_components not in (3, 12):
            raise ValueError(
                f"the solution vector has {vector.shape[-1]} entries, expected "
                f"3 or 12 per voxel of {self.n_voxels}"
            )
        leading = vector.shape[:-1]
        field = torch.zeros(
            (*leading, n_components, int(torch.tensor(self.shape).prod())),
            device=vector.device,
            dtype=vector.dtype,
        )
        field[..., self.mask.reshape(-1)] = vector.reshape(
            *leading, n_components, self.n_voxels
        )
        return field.reshape(*leading, n_components, *self.shape)

    def _check_field(self, field: torch.Tensor) -> None:
        """Raise unless the field's trailing axes match the grid."""
        if (
            field.ndim < 4
            or field.shape[-4] not in (3, 12)
            or tuple(field.shape[-3:]) != self.shape
        ):
            raise ValueError(
                "a field must end in (3 or 12, "
                f"{', '.join(str(n) for n in self.shape)}); got {tuple(field.shape)}"
            )

    @classmethod
    def sphere(
        cls,
        radius: float,
        resolution: float,
        permittivity: float,
        conductivity: float,
        *,
        padding: int = 0,
        device: torch.device | str | None = None,
    ) -> VoxelBody:
        """Return a homogeneous sphere centred on the grid.

        A voxel belongs to the sphere when its centre does, which is the
        staircase approximation the piecewise-constant basis implies.

        Parameters
        ----------
        radius
            Sphere radius in metres.
        resolution
            Grid pitch in metres.
        permittivity
            Relative permittivity inside the sphere.
        conductivity
            Conductivity inside the sphere, in S/m.
        padding
            Extra voxels of free space on each side.
        device
            Device the grids live on.

        Returns
        -------
        VoxelBody
            A body whose mask is the sphere.
        """
        half = int(torch.ceil(torch.tensor(radius / resolution)).item()) + padding
        n = 2 * half + 1
        offsets = torch.arange(n, device=device, dtype=torch.float64) - half
        axis = resolution * offsets
        x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
        inside = (x**2 + y**2 + z**2) <= radius**2

        return cls(
            permittivity=torch.where(
                inside,
                torch.tensor(permittivity, device=device, dtype=torch.float64),
                torch.tensor(1.0, device=device, dtype=torch.float64),
            ),
            conductivity=torch.where(
                inside,
                torch.tensor(conductivity, device=device, dtype=torch.float64),
                torch.tensor(0.0, device=device, dtype=torch.float64),
            ),
            mask=inside,
            resolution=resolution,
            origin=(float(axis[0]), float(axis[0]), float(axis[0])),
        )

    @classmethod
    def read_marie(
        cls, path: str | Path, *, device: torch.device | str | None = None
    ) -> VoxelBody:
        """Read a body model saved as MARIE's ``RHBM`` structure.

        Ported from ``geo_body_domain.m``. The file is a MATLAB v5 ``.mat``
        holding a struct ``RHBM`` with the voxel centres ``r``, shape
        ``(n1, n2, n3, 3)`` with x varying along the first axis, and the
        per-voxel ``epsilon_r`` and ``sigma_e``. The grid pitch is the step
        between the first two centres along x, as MARIE takes it.

        The body is the set of voxels ``idxS`` lists, one-based and in MATLAB's
        column-major order. Some of MARIE's own files carry no ``idxS``; for
        those the body is every voxel with ``sigma_e > 0``, the rule
        ``update_RHBM.m`` applies, which reproduces ``idxS`` exactly on the files
        that do carry it. A lossless scatterer therefore needs ``idxS``.

        The struct's density field, ``rho`` or ``rhos``, is proton density and is
        not read.

        Parameters
        ----------
        path
            File to read.
        device
            Device the grids are built on.

        Returns
        -------
        VoxelBody
            The body the file describes.

        Raises
        ------
        ImportError
            If scipy, which reads the file, is not installed.
        ValueError
            If the file holds no ``RHBM``, or its grid is not uniform.
        """
        try:
            from scipy.io import loadmat
        except ImportError as error:  # pragma: no cover - scipy is a dev dependency
            raise ImportError("reading a MARIE body model needs scipy") from error

        held = loadmat(Path(path), squeeze_me=True, struct_as_record=False)
        if "RHBM" not in held:
            raise ValueError(f"{path} holds no RHBM structure")
        rhbm = held["RHBM"]

        centres = np.asarray(rhbm.r, dtype=np.float64)
        permittivity = np.asarray(rhbm.epsilon_r, dtype=np.float64)
        conductivity = np.asarray(rhbm.sigma_e, dtype=np.float64)
        shape = permittivity.shape
        if centres.shape != (*shape, 3):
            raise ValueError(f"{path}: r is {centres.shape}, and epsilon_r is {shape}")

        resolution = abs(float(centres[1, 0, 0, 0] - centres[0, 0, 0, 0]))
        expected = centres[0, 0, 0] + resolution * np.stack(
            np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1
        )
        drift = float(np.abs(centres - expected).max())
        if drift > 1e-9 * max(resolution, 1.0):
            raise ValueError(
                f"{path}: the voxel centres are not a uniform grid of pitch "
                f"{resolution}; they depart from one by {drift:.3g} m"
            )

        if "idxS" in rhbm._fieldnames:
            indices = np.asarray(rhbm.idxS, dtype=np.int64).ravel() - 1
            mask = np.zeros(permittivity.size, dtype=bool)
            mask[indices] = True
            mask = mask.reshape(shape, order="F")
        else:
            mask = conductivity > 0

        def grid(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(values, device=device)

        return cls(
            permittivity=grid(permittivity),
            conductivity=grid(conductivity),
            mask=grid(mask),
            resolution=resolution,
            origin=tuple(float(value) for value in centres[0, 0, 0]),
        )
