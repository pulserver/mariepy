"""Coil performance maps: SNR, transmit efficiency and the SENSE g-factor.

Ported from MARIE 3.0's ``src_physics/src_performance_metrics/em_SNR.m``,
``em_TXE.m``, ``src_electromagnetism/em_g_factor.m``, and the noise matrices the
field routines form (``em_efield_svie_pfft.m``, ``em_efield_svie_mrgf.m``).

Every map rests on the noise covariance of the channels, ``Psi``: what the
body, the conductor and the lumped elements dissipate, as a quadratic form in
the channel drives. Its body part is ``integral sigma E_i^* E_j``, twice the
power the pair dissipates. MARIE adds the conductor's and the elements' parts
at half that scale, and only on the diagonal; here they enter at the body's
scale, with their channel-to-channel terms.
"""

from __future__ import annotations

import math

import torch

from mariepy import vie
from mariepy.constants import Medium

__all__ = [
    "BODY_TEMPERATURE",
    "BOLTZMANN",
    "SPINS",
    "equilibrium_magnetisation",
    "g_factor",
    "noise_covariance",
    "snr",
    "transmit_efficiency",
]

BOLTZMANN = 1.3806503e-23
"""MARIE's Boltzmann constant, in J/K."""

PLANCK = 6.626e-34
"""MARIE's Planck constant, in J s."""

BODY_TEMPERATURE = 310.0
"""MARIE's body temperature, in K."""

WATER_PROTONS = 6.691e28

SPINS: dict[str, tuple[float, float]] = {
    "1H": (0.5, WATER_PROTONS),
    "2H": (1.0, WATER_PROTONS),
    "3HE": (0.5, 2e25),
    "13C": (0.5, 0.011 * WATER_PROTONS),
    "7LI": (1.5, 0.925 * WATER_PROTONS),
    "15N": (0.5, 4e25),
}
"""Spin and number density per cubic metre of each nucleus MARIE tabulates.

MARIE sets nitrogen-15's density twice; the second value is the one it uses.
"""


def equilibrium_magnetisation(
    medium: Medium, temperature: float = BODY_TEMPERATURE
) -> float:
    """Return the thermal-equilibrium magnetisation per unit volume, MARIE's ``M0``.

    ``M0 = N (gamma hbar)^2 I (I + 1) B0 / (3 k T)``.

    Parameters
    ----------
    medium
        Supplies the field strength and the nucleus.
    temperature
        Sample temperature in kelvin.

    Returns
    -------
    float
        In amperes per metre.

    Raises
    ------
    KeyError
        If MARIE tabulates no spin and density for the nucleus.
    """
    key = medium.nucleus.upper()
    if key not in SPINS:
        raise KeyError(f"no spin and density for {medium.nucleus!r}")
    spin, density = SPINS[key]
    gamma = abs(medium.gyromagnetic_ratio) * 2 * math.pi
    hbar = PLANCK / (2 * math.pi)
    return (
        density
        * (gamma * hbar) ** 2
        * spin
        * (spin + 1)
        * medium.field_strength
        / (3 * BOLTZMANN * temperature)
    )


def noise_covariance(
    electric: torch.Tensor,
    conductivity: torch.Tensor,
    mask: torch.Tensor,
    resolution: float,
    *,
    coil: torch.Tensor | None = None,
    loss: torch.Tensor | None = None,
) -> torch.Tensor:
    """Form the channels' noise covariance from their fields and coil currents.

    Parameters
    ----------
    electric
        Each channel's electric field, shape ``(n, c, n1, n2, n3)`` with ``c``
        3 or 12, as :attr:`mariepy.fields.Fields.electric` holds it.
    conductivity
        Conductivity in S/m, shape ``(n1, n2, n3)``.
    mask
        Where the body is, same shape.
    resolution
        Voxel pitch in metres.
    coil
        Each channel's coil currents, shape ``(n, n_dof)``, or None to leave
        the conductor out.
    loss
        The conductor's and lumped elements' resistance, shape
        ``(n_dof, n_dof)``: :attr:`mariepy.sie.CoilSystem.copper_loss` plus
        :attr:`mariepy.sie.CoilSystem.lumped_loss`.

    Returns
    -------
    torch.Tensor
        ``Psi``, shape ``(n, n)``, Hermitian, in ohms per unit drive squared
        times the drive's units: ``Psi_ij = integral sigma E_i E_j^*`` plus
        ``J_i^T R J_j^*``, which MARIE writes as the conjugate of
        ``E^H sigma E``.
    """
    mass = vie.mass(electric.shape[-4], resolution).to(electric.device)
    weight = (conductivity * mask).to(torch.float64)
    covariance = torch.einsum(
        "pcxyz,qcxyz,c,xyz->pq",
        electric,
        electric.conj(),
        mass.to(electric.dtype),
        weight.to(electric.dtype),
    )
    if coil is not None and loss is not None:
        resistance = loss.real.to(coil.dtype)
        covariance = covariance + coil @ resistance @ coil.conj().transpose(0, 1)
    return covariance


def snr(
    b1_minus: torch.Tensor,
    covariance: torch.Tensor,
    medium: Medium,
    resolution: float,
    mask: torch.Tensor | None = None,
    *,
    temperature: float = BODY_TEMPERATURE,
) -> torch.Tensor:
    """Map the optimally combined SNR, as ``em_SNR.m`` does.

    ``SNR = V omega M0 sqrt(S^H Psi^-1 S) / sqrt(4 k T)``, with ``S`` the
    channels' ``B1-`` at a voxel and ``V`` its volume.

    Parameters
    ----------
    b1_minus
        Shape ``(n, n1, n2, n3)``, as
        :func:`mariepy.fields.circular_components` returns it.
    covariance
        ``Psi``, from :func:`noise_covariance`.
    medium
        Supplies the frequency and the nucleus.
    resolution
        Voxel pitch in metres.
    mask
        Where to map; zero elsewhere. By default everywhere.
    temperature
        Sample temperature in kelvin.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3)``, real.
    """
    scale = (
        resolution**3
        * medium.angular_frequency
        * equilibrium_magnetisation(medium, temperature)
        / math.sqrt(4 * BOLTZMANN * temperature)
    )
    inverse = torch.linalg.inv(covariance.to(b1_minus.dtype))
    sensitivity = b1_minus.reshape(b1_minus.shape[0], -1)
    quadratic = torch.einsum("pv,pq,qv->v", sensitivity.conj(), inverse, sensitivity)
    values = scale * torch.sqrt(quadratic.real.clamp(min=0))
    values = values.reshape(b1_minus.shape[1:])
    return values if mask is None else values * mask


def transmit_efficiency(
    b1_plus: torch.Tensor,
    covariance: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map the largest ``|B1+|^2`` per unit dissipated power, as ``em_TXE.m`` does.

    The largest eigenvalue of ``conj(Psi^-1) b^H b`` for the channels' row of
    ``B1+``, which has rank one, is ``b conj(Psi^-1) b^H``.

    Parameters
    ----------
    b1_plus
        Shape ``(n, n1, n2, n3)``.
    covariance
        ``Psi``, from :func:`noise_covariance`.
    mask
        Where to map; zero elsewhere. By default everywhere.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3)``, real.
    """
    inverse = torch.linalg.inv(covariance.to(b1_plus.dtype)).conj()
    field = b1_plus.reshape(b1_plus.shape[0], -1)
    values = torch.einsum("pv,pq,qv->v", field, inverse, field.conj()).real
    values = values.reshape(b1_plus.shape[1:])
    return values if mask is None else values * mask


def g_factor(
    covariance: torch.Tensor,
    sensitivity: torch.Tensor,
    mask: torch.Tensor,
    phase_reduction: int,
    frequency_reduction: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the SENSE g-factor of one slice, as ``em_g_factor.m`` does per plane.

    The sensitivities are whitened by the Cholesky factor of ``Psi``, and each
    set of pixels that fold onto one another, a field of view over the
    reduction apart, gives ``g = sqrt(diag((S^H S)^+) diag(S^H S))``.

    Parameters
    ----------
    covariance
        ``Psi``.
    sensitivity
        The channels' ``B1-`` over the slice, shape ``(n, n_phase, n_freq)``.
    mask
        Where the slice holds tissue, shape ``(n_phase, n_freq)``.
    phase_reduction, frequency_reduction
        Acceleration along each direction, MARIE's ``Rp`` and ``Rf``.

    Returns
    -------
    g : torch.Tensor
        Shape ``(n_phase, n_freq)``, zero outside the mask.
    inverse : torch.Tensor
        ``1 / g`` inside the mask, zero outside.
    """
    psi = covariance.clone()
    psi.diagonal().copy_(psi.diagonal().real)
    lower = torch.linalg.cholesky(psi.to(sensitivity.dtype))
    n, n_phase, n_freq = sensitivity.shape
    whitened = torch.linalg.solve_triangular(
        lower, sensitivity.reshape(n, -1), upper=False
    ).reshape(n, n_phase, n_freq)

    step_phase = n_phase // phase_reduction
    step_freq = n_freq // frequency_reduction
    g = torch.zeros((n_phase, n_freq), dtype=torch.float64, device=sensitivity.device)
    for x in range(n_freq // frequency_reduction):
        for y in range(n_phase // phase_reduction):
            rows = torch.arange(y, n_phase, step_phase, device=sensitivity.device)
            columns = torch.arange(x, n_freq, step_freq, device=sensitivity.device)
            folded = whitened[:, rows][:, :, columns].reshape(n, -1)
            gram = folded.conj().transpose(0, 1) @ folded
            unfolding = torch.linalg.pinv(gram)
            values = torch.sqrt(
                (torch.diagonal(unfolding) * torch.diagonal(gram)).abs()
            )
            g[rows[:, None], columns[None, :]] = values.reshape(
                rows.numel(), columns.numel()
            ).to(torch.float64)
    inside = mask.to(torch.bool)
    g = g * inside
    inverse = torch.where(inside & (g > 0), 1.0 / g.clamp(min=1e-300), 0.0)
    return g, inverse
