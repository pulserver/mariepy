"""A body model on a uniform Cartesian grid, and its dielectric contrast.

Ported from MARIE 3.0's ``src_geometry/body_geometry/geo_body_domain.m``,
``grid3d.m`` and ``src_physics/src_electromagnetism/em_assembly.m``.

The volume integral equation is solved for the polarisation current in the
voxels the mask selects. A field over the grid has shape
``(..., 3, n1, n2, n3)``, with the three spatial axes last so that an FFT over
them batches whatever leads; the solution vector that GMRES sees is that field
restricted to the mask and flattened, of length ``3 * n_voxels``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            Shape ``(..., 3, n1, n2, n3)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., 3 * n_voxels)``, the components in order.
        """
        self._check_field(field)
        leading = field.shape[:-4]
        flat = field.reshape(*leading, 3, -1)
        return flat[..., self.mask.reshape(-1)].reshape(*leading, self.n_dof)

    def from_dof(self, vector: torch.Tensor) -> torch.Tensor:
        """Spread a solution vector back over the grid, zero outside the mask.

        Parameters
        ----------
        vector
            Shape ``(..., 3 * n_voxels)``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., 3, n1, n2, n3)``.
        """
        if vector.shape[-1] != self.n_dof:
            raise ValueError(
                f"the solution vector has {vector.shape[-1]} entries, "
                f"expected {self.n_dof}"
            )
        leading = vector.shape[:-1]
        field = torch.zeros(
            (*leading, 3, int(torch.tensor(self.shape).prod())),
            device=vector.device,
            dtype=vector.dtype,
        )
        field[..., self.mask.reshape(-1)] = vector.reshape(*leading, 3, self.n_voxels)
        return field.reshape(*leading, 3, *self.shape)

    def _check_field(self, field: torch.Tensor) -> None:
        """Raise unless the field's trailing axes match the grid."""
        if field.ndim < 4 or tuple(field.shape[-4:]) != (3, *self.shape):
            raise ValueError(
                f"a field must end in (3, {', '.join(str(n) for n in self.shape)}); "
                f"got {tuple(field.shape)}"
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
