"""An RF shield between the coil and the body.

Ported from MARIE 3.0's shield path: ``src_geometry/scoil_geometry/geo_shield.m``
for the surface, ``src_integral_equations/src_sie/shield_sie_assembly.m`` for
its own matrix, ``Assembly_SIE_block_par.m`` for its coupling to the coil, and
``src_integral_equations/src_wsvie/src_tt/tt_shield_coupling_assembly.m`` for
its coupling to the body.

A shield is a conducting surface with no port of its own, carrying the RWG
basis a coil does. It reaches the coil through a dense block, and the body
through a tensor train over (body voxel, shield basis function): the shield
surrounds the body, so the precorrected FFT's expansion grid would have to span
both, and the train holds the same coupling in far less.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mariepy import coupling as coupling_kernels
from mariepy import tt
from mariepy import wire as wire_module
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.sie import CoilSystem, coupling_matrix
from mariepy.sie import assemble as assemble_system

__all__ = ["Shield", "apply", "apply_transpose", "assemble", "body_coupling"]


@dataclass(frozen=True)
class Shield:
    """A shield and everything that couples it to the coil and the body.

    Attributes
    ----------
    surface
        The shield's surface and its basis.
    system
        Its own matrix, MARIE's ``SIE.Z_shield``.
    coil_coupling
        Its interaction with the coil, MARIE's ``Zsc``, shape
        ``(n_shield, n_coil)``: shield rows, coil columns.
    electric
        Its N coupling to the body grid, MARIE's ``tt_Zbs_N``: one tensor train
        over ``(n1, n2, n3, n_shield)`` per body unknown of a voxel, 3 or 12,
        ordered as the body's.
    magnetic
        The same for the K coupling, MARIE's ``tt_Zbs_K``.
    """

    surface: SurfaceCoil
    system: CoilSystem
    coil_coupling: torch.Tensor
    electric: tuple[tt.TensorTrain, ...]
    magnetic: tuple[tt.TensorTrain, ...]

    @property
    def n_dof(self) -> int:
        """Number of shield unknowns."""
        return self.surface.n_dof


def assemble(
    surface: SurfaceCoil,
    coil: SurfaceCoil | wire_module.WireCoil | wire_module.CombinedCoil,
    body: VoxelBody,
    medium: Medium,
    *,
    tol: float = 1e-3,
    linear: bool = False,
    triangle_order: int = 4,
    cell_order: int = 2,
    near_order: int = 4,
) -> Shield:
    """Build a shield's matrix and its couplings to the coil and the body.

    Parameters
    ----------
    surface
        The shield's surface, with its lumped elements and any driven ports.
    coil
        The coil it surrounds: a surface coil, a wire coil or both.
    body
        The body it surrounds.
    medium
        Free-space constants at the working frequency.
    tol
        Relative accuracy of the tensor trains, MARIE's ``tol_TT``.
    linear
        Couple to the body's piecewise-linear basis.
    triangle_order, cell_order
        Quadrature orders of the shield-to-body kernels.
    near_order
        Quadrature order of the coil-to-shield block on each triangle.

    Returns
    -------
    Shield
        Ready for :class:`mariepy.system.ShieldedOperator`.
    """
    return Shield(
        surface=surface,
        system=assemble_system(surface, medium),
        coil_coupling=_coil_coupling(surface, coil, medium, near_order),
        electric=body_coupling(
            body,
            surface,
            medium,
            magnetic=False,
            tol=tol,
            linear=linear,
            triangle_order=triangle_order,
            cell_order=cell_order,
        ),
        magnetic=body_coupling(
            body,
            surface,
            medium,
            magnetic=True,
            tol=tol,
            linear=linear,
            triangle_order=triangle_order,
            cell_order=cell_order,
        ),
    )


def _coil_coupling(surface, coil, medium, order):
    """Assemble the shield's interaction with each kind of coil, shield rows first.

    A wire's block is MARIE's ``Zsw``, the transpose of the wire-to-surface
    interaction, as ``wsvie_coupling_assembly.m`` builds it.
    """
    if isinstance(coil, wire_module.CombinedCoil):
        return torch.cat(
            [
                _coil_coupling(surface, coil.wire, medium, order),
                _coil_coupling(surface, coil.surface, medium, order),
            ],
            dim=1,
        )
    if isinstance(coil, wire_module.WireCoil):
        return wire_module.surface_coupling(
            coil, surface, medium, triangle_order=order
        ).transpose(0, 1)
    return coupling_matrix(surface, coil, medium, order=order)


def body_coupling(
    body: VoxelBody,
    surface: SurfaceCoil,
    medium: Medium,
    *,
    magnetic: bool,
    tol: float = 1e-3,
    linear: bool = False,
    triangle_order: int = 4,
    cell_order: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[tt.TensorTrain, ...]:
    """Build the shield-to-body coupling as tensor trains, one per body unknown.

    Ported from ``tt_shield_coupling_assembly.m``. Each entry is the coupling
    kernel of one shield basis function averaged over one voxel, times the
    voxel volume, as ``assemble_coupling_matrices_voxels_*.m`` compute it from
    MARIE's 24 coupling sources, and as :func:`mariepy.pfft.direct_coupling`
    does for the coil. The trains span the whole body grid, voxels outside the
    body included.

    Parameters
    ----------
    body
        Supplies the grid.
    surface
        The shield and its basis.
    medium
        Free-space constants.
    magnetic
        Build the K coupling rather than the N one.
    tol
        Relative accuracy of each train.
    linear
        Build one train per unknown of the piecewise-linear basis.
    triangle_order, cell_order
        Quadrature orders of the kernel.
    generator
        Source of the cross approximation's random start.

    Returns
    -------
    tuple of TensorTrain
        One train per body unknown of a voxel, 3 or 12, over
        ``(n1, n2, n3, n_shield)``.
    """
    kernel = coupling_kernels.coupling_k if magnetic else coupling_kernels.coupling_n
    corners = surface.rwg_vertices().cpu()
    centres = body.coordinates().cpu()
    volume = body.resolution**3
    sizes = (*body.shape, surface.n_dof)
    terms = range(4) if linear else range(1)

    trains = []
    for component in range(3):
        for term in terms:

            def entries(subscripts, component=component, term=term):
                points = centres[
                    :, subscripts[:, 0], subscripts[:, 1], subscripts[:, 2]
                ]
                values = kernel(
                    corners[subscripts[:, 3]],
                    points.transpose(0, 1).contiguous(),
                    medium,
                    triangle_order=triangle_order,
                    cell_size=body.resolution,
                    cell_order=cell_order,
                    basis_term=term,
                )
                return volume * values[:, component]

            trains.append(
                tt.cross(sizes, entries, tol, generator=generator).to(body.device)
            )
    return tuple(trains)


def apply(
    trains: tuple[tt.TensorTrain, ...], current: torch.Tensor, body: VoxelBody
) -> torch.Tensor:
    """Give the moments a shield current puts on the body's unknowns.

    Ported from ``mvp_coupling_cross_TT_pwc.m`` and its linear counterpart,
    restricted to the body as MARIE's callers restrict it.

    Parameters
    ----------
    trains
        :attr:`Shield.electric` or :attr:`Shield.magnetic`.
    current
        Shield currents, shape ``(n_shield,)`` or ``(n_ports, n_shield)``.
    body
        Supplies the mask.

    Returns
    -------
    torch.Tensor
        Shape ``(c * n_voxels,)``, or ``(n_ports, c * n_voxels)``.
    """
    batch = current.ndim == 2
    rows = current if batch else current[None]
    out = []
    for row in rows:
        field = torch.stack([train.apply(row) for train in trains])
        out.append(body.to_dof(field))
    stacked = torch.stack(out)
    return stacked if batch else stacked[0]


def apply_transpose(
    trains: tuple[tt.TensorTrain, ...], current: torch.Tensor, body: VoxelBody
) -> torch.Tensor:
    """Give the moments a body current puts on the shield's unknowns.

    Ported from ``mvp_coupling_cross_TT_transpose_pwc.m`` and its linear
    counterpart.

    Parameters
    ----------
    trains
        :attr:`Shield.electric`.
    current
        Body current, shape ``(c * n_voxels,)``.
    body
        Supplies the mask.

    Returns
    -------
    torch.Tensor
        Shape ``(n_shield,)``.
    """
    field = body.from_dof(current)
    return sum(
        train.apply_transpose(field[component])
        for component, train in enumerate(trains)
    )
