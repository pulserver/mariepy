"""The electric and magnetic field in the body, and the power it absorbs.

Ported from MARIE 3.0's ``src_physics/src_electromagnetism/em_ehfield_wsvie.m``
and its two halves, ``em_efield/em_efield_svie/em_efield_svie_pfft.m`` and
``em_hfield/em_hfield_svie/em_hfield_svie_pfft.m``.

The field is taken the way the operator is applied: the coil's currents and the
body's own go through the extended grid, the near corrections put back what the
projection gets wrong, and the result is a cell integral, so dividing by the
cell volume gives the field itself. The coil's part and the body's are kept
apart because the power balance needs both: what the coil delivers to the body
is what the body absorbs plus what it scatters back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mariepy import vie
from mariepy.system import CoupledOperator

__all__ = [
    "Fields",
    "absorbed_power",
    "circular_components",
    "compute",
    "delivered_power",
]


@dataclass(frozen=True)
class Fields:
    """The field each port drives in the body.

    Attributes
    ----------
    electric
        Total electric field, shape ``(n_ports, 3, n1, n2, n3)``, in V/m, zero
        outside the body.
    magnetic
        Total magnetic field, same shape, in A/m.
    incident
        The part of the electric field the coil's own current puts there, the
        quantity MARIE calls incident.
    scattered
        The part the body's own polarisation current puts there.
    """

    electric: torch.Tensor
    magnetic: torch.Tensor
    incident: torch.Tensor
    scattered: torch.Tensor


def compute(
    operator: CoupledOperator, coil: torch.Tensor, body: torch.Tensor
) -> Fields:
    """Evaluate the field of every port's solution over the body.

    Parameters
    ----------
    operator
        The coupled operator the currents were solved with.
    coil
        Coil currents, shape ``(n_ports, n_dof)``.
    body
        Body currents, shape ``(n_ports, 3 * n_voxels)``.

    Returns
    -------
    Fields
        The total field and the two parts it is made of.
    """
    coupling = operator.coupling
    grid = coupling.grid
    volume = grid.resolution**3
    scaling = operator.medium.electric_scaling

    from_coil = _spread(coupling.project, coil, grid.shape)
    from_body = _spread(coupling.scatter, body, grid.shape)

    incident = _test(
        coupling.scatter, _electric(coupling, from_coil, grid.resolution) / scaling
    ) + _apply(coupling.electric, coil)
    scattered = _test(
        coupling.scatter, _electric(coupling, from_body, grid.resolution) / scaling
    )
    magnetic = (
        _test(coupling.scatter, vie.apply_k(coupling.symbols_k, from_coil))
        + _apply(coupling.magnetic, coil)
        + _test(coupling.scatter, vie.apply_k(coupling.symbols_k, from_body))
    )

    incident = operator.body.from_dof(incident / volume)
    scattered = operator.body.from_dof(scattered / volume)
    return Fields(
        electric=incident + scattered,
        magnetic=operator.body.from_dof(magnetic / volume),
        incident=incident,
        scattered=scattered,
    )


def absorbed_power(operator: CoupledOperator, fields: Fields) -> torch.Tensor:
    """Integrate the ohmic loss of the electric field over the body.

    Parameters
    ----------
    operator
        Supplies the body's conductivity and its cell volume.
    fields
        The fields the ports drive.

    Returns
    -------
    torch.Tensor
        Shape ``(n_ports,)``, real, in watts.
    """
    body = operator.body
    density = 0.5 * body.conductivity * (fields.electric.abs() ** 2).sum(dim=-4)
    return (density * body.mask).sum(dim=(-3, -2, -1)) * body.resolution**3


def delivered_power(operator: CoupledOperator, coil: torch.Tensor) -> torch.Tensor:
    """Give the power each port delivers to the coil.

    Parameters
    ----------
    operator
        Supplies the port drive.
    coil
        Coil currents, shape ``(n_ports, n_dof)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n_ports,)``, real, in watts. It is spent on the conductor, on
        the body, and on radiation, and the three cannot be separated from the
        port alone: the coil's field and the body's interfere, so what the
        pair radiates is not what the coil would radiate by itself.
    """
    drive = operator.system.excitation
    return -0.5 * torch.real((drive.conj() * coil).sum(dim=-1))


def power_balance(
    operator: CoupledOperator, fields: Fields, body: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Take the body's extinction, absorption and scattering from the fields.

    The body's own rows say the total field is the coil's plus the body's own,
    so testing that against the polarisation current splits the power the body
    takes from the coil into what it turns to heat and what it puts back.

    Parameters
    ----------
    operator
        Supplies the cell volume.
    fields
        The fields the ports drive.
    body
        Body currents, shape ``(n_ports, 3 * n_voxels)``.

    Returns
    -------
    taken : torch.Tensor
        Power the body takes out of the coil's field, shape ``(n_ports,)``.
    absorbed : torch.Tensor
        Power it turns to heat.
    scattered : torch.Tensor
        Power its own current puts back.
    """
    volume = operator.body.resolution**3
    incident = operator.body.to_dof(fields.incident)
    scattered = operator.body.to_dof(fields.scattered)
    taken = 0.5 * volume * torch.real((body.conj() * incident).sum(dim=-1))
    given = -0.5 * volume * torch.real((body.conj() * scattered).sum(dim=-1))
    return taken, taken - given, given


def circular_components(
    operator: CoupledOperator, fields: Fields
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give the transmit and receive circular components of the magnetic field.

    Parameters
    ----------
    operator
        Supplies the permeability of free space.
    fields
        The fields the ports drive.

    Returns
    -------
    plus : torch.Tensor
        ``B1+``, shape ``(n_ports, n1, n2, n3)``, in tesla.
    minus : torch.Tensor
        ``B1-``, same shape.
    """
    permeability = operator.medium.permeability
    transverse = fields.magnetic[..., 0, :, :, :], fields.magnetic[..., 1, :, :, :]
    return (
        permeability * (transverse[0] + 1j * transverse[1]),
        permeability * (transverse[0] - 1j * transverse[1]),
    )


def _spread(matrix: torch.Tensor, current: torch.Tensor, shape) -> torch.Tensor:
    """Put a batch of currents on the extended grid."""
    spread = torch.sparse.mm(matrix, current.transpose(0, 1))
    return spread.transpose(0, 1).reshape(current.shape[0], 3, *shape)


def _test(matrix: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Restrict a batch of grid fields to the body's degrees of freedom."""
    flat = field.reshape(field.shape[0], -1).transpose(0, 1)
    return torch.sparse.mm(matrix.transpose(0, 1), flat).transpose(0, 1)


def _apply(matrix: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Multiply a sparse matrix by a batch of vectors."""
    return torch.sparse.mm(matrix, current.transpose(0, 1)).transpose(0, 1)


def _electric(coupling, field: torch.Tensor, resolution: float) -> torch.Tensor:
    """Apply the body's electric kernel, less its own delta term."""
    return vie.apply_n(coupling.symbols_n, field) - vie.apply_g(field, resolution)
