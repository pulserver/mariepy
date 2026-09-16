"""The coupled coil-and-body operator, and the preconditioner that tames it.

Ported from MARIE 3.0's ``src_solver/src_mvp/mvp_svie/mvp_svie_pfft.m``,
``src_solver/src_preconditioners/prec_wsvie.m`` and ``prec_LU.m``.

The unknown is one vector: the coil's surface-current coefficients followed by
the body's polarisation current. The coil's rows say that the tangential
electric field vanishes on the conductor; the body's rows say that the
polarisation current is what the total field induces. Everything except the two
near corrections and the coil's own matrix passes through the extended grid, so
one convolution carries the coil to the body, the body to the coil and the body
to itself at the same time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mariepy import vie
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.pfft import Coupling
from mariepy.preconditioner import body_diagonal
from mariepy.sie import CoilSystem
from mariepy.wire import WireCoil

if TYPE_CHECKING:
    from mariepy.shield import Shield

__all__ = ["CoupledOperator", "ShieldedOperator"]


@dataclass(frozen=True)
class CoupledOperator:
    """One coil and one body, solved together.

    Attributes
    ----------
    body
        The body and its grid.
    coil
        The coil and its basis.
    medium
        The frequency everything was built at.
    system
        The coil's own matrix and its port drive.
    coupling
        The projection, the scatter and the near corrections.
    """

    body: VoxelBody
    coil: SurfaceCoil | WireCoil
    medium: Medium
    system: CoilSystem
    coupling: Coupling

    @property
    def n_coil(self) -> int:
        """Number of coil unknowns."""
        return self.coil.n_dof

    @property
    def n_body(self) -> int:
        """Number of body unknowns: 3 or 12 per voxel, by the body's basis."""
        return self.coupling.n_components * self.body.n_voxels

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the coupled operator to one solution vector.

        Parameters
        ----------
        vector
            Shape ``(n_coil + n_body,)``, complex.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        coil_current = vector[: self.n_coil]
        body_current = vector[self.n_coil :]
        scaling = self.medium.electric_scaling

        on_body = _apply(self.coupling.scatter, body_current)
        on_grid = _apply(self.coupling.project, coil_current) + on_body
        n_components = self.coupling.n_components
        field = on_grid.reshape(n_components, *self.coupling.grid.shape)
        applied = (
            vie.apply_n(self.coupling.symbols_n, field)
            - vie.apply_g(field, self.coupling.grid.resolution)
        ).reshape(-1) / scaling

        induced = (
            vie.apply_g(
                self._contrast_inverse()
                * on_body.reshape(n_components, *self.coupling.grid.shape),
                self.coupling.grid.resolution,
            ).reshape(-1)
            / scaling
        )

        coil_out = (
            _apply(self.coupling.project.transpose(0, 1), applied)
            + _apply(self.coupling.coil, coil_current)
            + _apply(self.coupling.electric.transpose(0, 1), body_current)
        )
        body_out = (
            -_apply(self.coupling.scatter.transpose(0, 1), applied)
            + _apply(self.coupling.scatter.transpose(0, 1), induced)
            - _apply(self.coupling.electric, coil_current)
        )
        return torch.cat([coil_out, body_out])

    def split(self, vector: torch.Tensor):
        """Separate a solution vector into its coil, body and shield parts.

        Parameters
        ----------
        vector
            Shape ``(..., n_coil + n_body)``.

        Returns
        -------
        tuple
            The coil part, the body part, and ``None`` for the shield.
        """
        return vector[..., : self.n_coil], vector[..., self.n_coil :], None

    def right_hand_side(self) -> torch.Tensor:
        """Give the drive of each port over the whole unknown vector.

        Returns
        -------
        torch.Tensor
            Shape ``(n_driven, n_coil + n_body)``, complex: only the coil's own
            rows are driven.
        """
        drive = self.system.excitation
        rest = torch.zeros(
            (drive.shape[0], self.n_body), dtype=drive.dtype, device=drive.device
        )
        return torch.cat([drive, rest], dim=1)

    def preconditioner(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the split left preconditioner.

        The coil block is inverted exactly, by an LU factorisation of the coil's
        own matrix, and the body block by its Galerkin mass term, which is
        diagonal in either basis.

        Returns
        -------
        callable
            Applies the preconditioner to a vector.
        """
        factors = torch.linalg.lu_factor(self.system.impedance)
        diagonal = body_diagonal(self.body, self.medium, linear=self.coupling.linear)

        def apply(vector: torch.Tensor) -> torch.Tensor:
            coil = torch.linalg.lu_solve(*factors, vector[: self.n_coil, None])[:, 0]
            return torch.cat([coil, diagonal * vector[self.n_coil :]])

        return apply

    def _contrast_inverse(self) -> torch.Tensor:
        """Give ``1 / Mc`` over the extended grid, zero where there is no tissue."""
        grid = self.coupling.grid
        inverse = torch.zeros(grid.shape, dtype=torch.complex128, device=grid.device)
        start = grid.body_origin
        scattering = self.body.contrast(self.medium).scattering
        block = torch.zeros_like(scattering)
        block[self.body.mask] = 1.0 / scattering[self.body.mask]
        inverse[
            start[0] : start[0] + self.body.shape[0],
            start[1] : start[1] + self.body.shape[1],
            start[2] : start[2] + self.body.shape[2],
        ] = block
        return inverse


def _apply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Multiply a sparse matrix by a vector."""
    return torch.sparse.mm(matrix, vector[:, None])[:, 0]


@dataclass(frozen=True)
class ShieldedOperator:
    """A coil, a body and a shield around them, solved together.

    Ported from MARIE's ``mvp_svie_pfft_tt.m`` and ``ie_solver_svie_pfft_tt.m``.
    The unknown is the shield's current, then the coil's, then the body's. The
    coil and the body interact as in :class:`CoupledOperator`; the shield
    reaches the coil through its dense block and the body through its tensor
    trains, with the signs the coil's own coupling to the body carries.

    Attributes
    ----------
    coupled
        The coil and the body.
    shield
        The shield and its couplings.
    """

    coupled: CoupledOperator
    shield: Shield

    @property
    def body(self) -> VoxelBody:
        """The body and its grid."""
        return self.coupled.body

    @property
    def coil(self) -> SurfaceCoil:
        """The coil and its basis."""
        return self.coupled.coil

    @property
    def medium(self) -> Medium:
        """The frequency everything was built at."""
        return self.coupled.medium

    @property
    def system(self) -> CoilSystem:
        """The coil's own matrix and its port drive."""
        return self.coupled.system

    @property
    def coupling(self) -> Coupling:
        """The coil-and-body pieces."""
        return self.coupled.coupling

    @property
    def n_shield(self) -> int:
        """Number of shield unknowns."""
        return self.shield.n_dof

    @property
    def n_coil(self) -> int:
        """Number of coil unknowns."""
        return self.coupled.n_coil

    @property
    def n_body(self) -> int:
        """Number of body unknowns."""
        return self.coupled.n_body

    def split(self, vector: torch.Tensor):
        """Separate a solution vector into its coil, body and shield parts.

        Parameters
        ----------
        vector
            Shape ``(..., n_shield + n_coil + n_body)``.

        Returns
        -------
        tuple of torch.Tensor
            The coil part, the body part and the shield part.
        """
        shield = vector[..., : self.n_shield]
        rest = vector[..., self.n_shield :]
        coil, body, _ = self.coupled.split(rest)
        return coil, body, shield

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the shielded operator to one solution vector.

        Parameters
        ----------
        vector
            Shape ``(n_shield + n_coil + n_body,)``, complex.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        from mariepy import shield as shield_module

        coil_current, body_current, shield_current = self.split(vector)
        inner = self.coupled(vector[self.n_shield :])
        coil_out = (
            inner[: self.n_coil]
            + self.shield.coil_coupling.transpose(0, 1) @ shield_current
        )
        body_out = inner[self.n_coil :] - shield_module.apply(
            self.shield.electric, shield_current, self.body
        )
        shield_out = (
            self.shield.system.impedance @ shield_current
            + self.shield.coil_coupling @ coil_current
            + shield_module.apply_transpose(
                self.shield.electric, body_current, self.body
            )
        )
        return torch.cat([shield_out, coil_out, body_out])

    def right_hand_side(self) -> torch.Tensor:
        """Give the drive of each port over the whole unknown vector.

        Returns
        -------
        torch.Tensor
            Shape ``(n_driven, n_shield + n_coil + n_body)``: the shield takes
            no drive of its own.
        """
        inner = self.coupled.right_hand_side()
        none = torch.zeros(
            (inner.shape[0], self.n_shield), dtype=inner.dtype, device=inner.device
        )
        return torch.cat([none, inner], dim=1)

    def preconditioner(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the split left preconditioner.

        Ported from ``prec_wsvie.m``: the shield and the coil together are
        inverted exactly, by an LU factorisation of their joint matrix, and the
        body by its Galerkin mass term.

        Returns
        -------
        callable
            Applies the preconditioner to a vector.
        """
        joint = torch.cat(
            [
                torch.cat(
                    [self.shield.system.impedance, self.shield.coil_coupling], dim=1
                ),
                torch.cat(
                    [
                        self.shield.coil_coupling.transpose(0, 1),
                        self.system.impedance,
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
        factors = torch.linalg.lu_factor(joint)
        diagonal = body_diagonal(self.body, self.medium, linear=self.coupling.linear)
        surfaces = self.n_shield + self.n_coil

        def apply(vector: torch.Tensor) -> torch.Tensor:
            head = torch.linalg.lu_solve(*factors, vector[:surfaces, None])[:, 0]
            return torch.cat([head, diagonal * vector[surfaces:]])

        return apply
