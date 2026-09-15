"""Admittance, impedance and scattering parameters of the coil's ports.

Ported from MARIE 3.0's ``src_physics/src_electronics/src_network_parameters/
np_compute.m`` and its transforms ``np_y2z.m``, ``np_z2y.m`` and ``np_z2s.m``.

The admittance of port pair ``(c, d)`` is the current the drive of port ``d``
pushes through port ``c``, which the delta-gap convention delivers as the
contraction of one port's drive vector with the other port's solution. That
contraction is reciprocal by physics but not by construction, which is why
:func:`port_admittance` returns it unsymmetrised and :func:`symmetrise` is a
separate step.
"""

from __future__ import annotations

import torch

__all__ = ["port_admittance", "symmetrise", "y_to_z", "z_to_s", "z_to_y"]


def port_admittance(excitation: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Contract each port's drive with each port's solution.

    Parameters
    ----------
    excitation
        Port drives, shape ``(n_ports, n_dof)``, as :func:`mariepy.sie.port_excitation`
        returns them.
    current
        Solution for each drive, same shape.

    Returns
    -------
    torch.Tensor
        Shape ``(n_ports, n_ports)``, complex, MARIE's ``Ip``: the admittance
        before it is symmetrised.
    """
    return -(excitation @ current.transpose(-2, -1))


def symmetrise(matrix: torch.Tensor) -> torch.Tensor:
    """Average a port matrix with its transpose.

    Parameters
    ----------
    matrix
        Shape ``(n_ports, n_ports)``.

    Returns
    -------
    torch.Tensor
        The reciprocal part of it.
    """
    return 0.5 * (matrix + matrix.transpose(-2, -1))


def y_to_z(admittance: torch.Tensor) -> torch.Tensor:
    """Invert an admittance matrix into an impedance matrix.

    Parameters
    ----------
    admittance
        Shape ``(n_ports, n_ports)``, in siemens.

    Returns
    -------
    torch.Tensor
        The impedance matrix in ohms.
    """
    return torch.linalg.inv(admittance)


def z_to_y(impedance: torch.Tensor) -> torch.Tensor:
    """Invert an impedance matrix into an admittance matrix.

    Parameters
    ----------
    impedance
        Shape ``(n_ports, n_ports)``, in ohms.

    Returns
    -------
    torch.Tensor
        The admittance matrix in siemens.
    """
    return torch.linalg.inv(impedance)


def z_to_s(impedance: torch.Tensor, reference: float = 50.0) -> torch.Tensor:
    """Refer an impedance matrix to a transmission-line impedance.

    Parameters
    ----------
    impedance
        Shape ``(n_ports, n_ports)``, in ohms.
    reference
        Line impedance in ohms.

    Returns
    -------
    torch.Tensor
        The scattering matrix.
    """
    identity = reference * torch.eye(
        impedance.shape[-1], dtype=impedance.dtype, device=impedance.device
    )
    return torch.linalg.solve(impedance + identity, impedance - identity)
