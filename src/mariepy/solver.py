"""Driving the body solver.

Ported from MARIE 3.0's ``src_solver/src_ie_solver/ie_solver_vie/
ie_solver_vie.m`` and ``src_solver/src_mvp/mvp_vie/mvp_vie.m``.

With no coil present the body is driven by an incident field, and the volume
integral equation is solved for the polarisation current it induces. That is the
configuration the analytic Mie series covers, and the one the solver is checked
against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mariepy import fields, network, pfft, sie, vie, wire
from mariepy import shield as shield_module
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.fields import Fields
from mariepy.gmres import Solution, gmres
from mariepy.preconditioner import body_diagonal
from mariepy.system import CoupledOperator, ShieldedOperator
from mariepy.tucker import circulant_tucker
from mariepy.wire import CombinedCoil, WireCoil

__all__ = [
    "BodyOperator",
    "PortSolution",
    "Result",
    "assemble_coil",
    "solve",
    "solve_body",
    "solve_ports",
]


@dataclass(frozen=True)
class BodyOperator:
    """The volume integral operator of one body at one frequency.

    Attributes
    ----------
    body
        The grid and its contrast.
    medium
        The frequency the kernel was built at.
    symbols
        The compressed N kernel.
    linear
        Whether the current is written in the linear basis.
    """

    body: VoxelBody
    medium: Medium
    symbols: tuple
    linear: bool = False

    @classmethod
    def build(
        cls,
        body: VoxelBody,
        medium: Medium,
        *,
        tol: float = 1e-7,
        far_order: int = 4,
        medium_order: int = 8,
        near_order: int = 15,
        linear: bool = False,
    ) -> BodyOperator:
        """Assemble and compress the kernel for one body.

        Parameters
        ----------
        body
            The grid to build the kernel on.
        medium
            Supplies the wavenumber.
        tol
            Relative tolerance of the Tucker compression.
        far_order, medium_order, near_order
            Quadrature orders for the three regimes.
        linear
            Write the current in the piecewise-linear basis, twelve functions
            per voxel, as MARIE does when ``Basis_Functions_VIE`` is 1.

        Returns
        -------
        BodyOperator
            Ready to apply.
        """
        kernel = vie.kernel_n(
            body.shape,
            body.resolution,
            medium.wavenumber,
            far_order=far_order,
            medium_order=medium_order,
            near_order=near_order,
            linear=linear,
        )
        return cls(
            body=body,
            medium=medium,
            symbols=circulant_tucker(kernel.to(body.device), tol),
            linear=linear,
        )

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Apply the operator to a solution vector.

        The equation is ``J - Mc/Mr * G^-1 N J = ce * Mc/Mr * E_inc``, which is
        MARIE's ``mvp_vie.m`` with its contrast written out.

        Parameters
        ----------
        current
            Shape ``(3 * n_voxels,)``, or ``(12 * n_voxels,)`` in the linear
            basis.

        Returns
        -------
        torch.Tensor
            Same shape.
        """
        field = self.body.from_dof(current)
        applied = vie.apply_n(self.symbols, field)
        contrast = self.body.contrast(self.medium)
        reduced = contrast.reduced.unsqueeze(0)
        scattered = reduced * vie.apply_inverse_g(applied, self.body.resolution)
        return current - self.body.to_dof(scattered)

    def right_hand_side(self, incident: torch.Tensor) -> torch.Tensor:
        """Turn an incident electric field into the solver's right-hand side.

        Parameters
        ----------
        incident
            The incident electric field in the operator's basis: shape
            ``(3, n1, n2, n3)``, or ``(12, n1, n2, n3)`` in the linear basis, as
            :func:`mariepy.incident.plane_wave` returns it.

        Returns
        -------
        torch.Tensor
            Shape ``(3 * n_voxels,)`` or ``(12 * n_voxels,)``.
        """
        contrast = self.body.contrast(self.medium)
        scaling = torch.tensor(
            self.medium.electric_scaling,
            device=incident.device,
            dtype=torch.complex128,
        )
        driven = scaling * contrast.reduced.unsqueeze(0) * incident
        return self.body.to_dof(driven)

    def total_field(
        self, current: torch.Tensor, incident: torch.Tensor
    ) -> torch.Tensor:
        """Return the total electric field the solved current implies.

        The scattered field is ``G^-1 (N J - G J) / (j omega eps_0)``, as
        MARIE's ``em_efield_vie_excitation.m`` forms it.

        Parameters
        ----------
        current
            The solved polarisation current.
        incident
            The field that drove it, in the operator's basis.

        Returns
        -------
        torch.Tensor
            The total field in the operator's basis, shape ``(3, n1, n2, n3)``
            or ``(12, n1, n2, n3)``, zero outside the mask.
        """
        scaling = torch.tensor(
            self.medium.electric_scaling, device=current.device, dtype=torch.complex128
        )
        polarisation = self.body.from_dof(current)
        applied = vie.apply_n(self.symbols, polarisation)
        scattered = vie.apply_inverse_g(
            (applied - vie.apply_g(polarisation, self.body.resolution)) / scaling,
            self.body.resolution,
        )
        return self.body.mask.unsqueeze(0) * (incident + scattered)


def solve_body(
    operator: BodyOperator,
    incident: torch.Tensor,
    *,
    tol: float = 1e-5,
    restart: int = 50,
    maxit: int = 200,
) -> Solution:
    """Solve for the polarisation current an incident field induces.

    Parameters
    ----------
    operator
        The body operator.
    incident
        The incident electric field in the operator's basis.
    tol
        Target for the preconditioned relative residual.
    restart, maxit
        Passed to :func:`mariepy.gmres.gmres`.

    Returns
    -------
    mariepy.gmres.Solution
        The solved current and the residual history.
    """
    diagonal = body_diagonal(operator.body, operator.medium, linear=operator.linear)
    return gmres(
        operator,
        operator.right_hand_side(incident),
        preconditioner=lambda vector: diagonal * vector,
        tol=tol,
        restart=restart,
        maxit=maxit,
    )


@dataclass(frozen=True)
class PortSolution:
    """The currents one coil drives, and how the solves went.

    Attributes
    ----------
    coil
        Surface-current coefficients, shape ``(n_ports, n_dof)``.
    body
        Polarisation current, shape ``(n_ports, c * n_voxels)`` with ``c`` 3 or
        12.
    residual
        Final relative residual of each port's solve.
    iterations
        Iterations each port's solve took.
    shield
        Shield currents, shape ``(n_ports, n_shield)``, or None without a
        shield.
    """

    coil: torch.Tensor
    body: torch.Tensor
    residual: tuple[float, ...]
    iterations: tuple[int, ...]
    shield: torch.Tensor | None = None


def solve_ports(
    operator: CoupledOperator | ShieldedOperator,
    *,
    tol: float = 1e-5,
    restart: int = 50,
    maxit: int = 200,
) -> PortSolution:
    """Drive each port in turn and solve the coupled system.

    Ported from MARIE 3.0's ``src_solver/src_ie_solver/ie_solver_svie/
    ie_solver_svie_pfft.m``.

    Parameters
    ----------
    operator
        The coupled operator.
    tol
        Target for the preconditioned relative residual.
    restart
        Iterations per restart cycle.
    maxit
        Maximum restart cycles.

    Returns
    -------
    PortSolution
        The coil and body currents of every port.
    """
    drives = operator.right_hand_side()
    precondition = operator.preconditioner()
    solutions, residual, iterations = [], [], []
    for row in range(drives.shape[0]):
        solution = gmres(
            operator,
            drives[row],
            preconditioner=precondition,
            restart=restart,
            tol=tol,
            maxit=maxit,
        )
        solutions.append(solution.x)
        residual.append(float(solution.residuals[-1]))
        iterations.append(len(solution.residuals) - 1)
    coil, body, shield = operator.split(torch.stack(solutions))
    return PortSolution(
        coil=coil,
        body=body,
        residual=tuple(residual),
        iterations=tuple(iterations),
        shield=shield,
    )


@dataclass(frozen=True)
class Result:
    """Everything one coil and one body produce at one frequency.

    Attributes
    ----------
    operator
        The coupled operator that was solved.
    ports
        Each port's coil and body currents.
    admittance
        Port admittance in siemens, symmetrised as ``np_compute.m`` does.
    impedance
        Port impedance in ohms.
    scattering
        Port scattering parameters, referred to the line impedance.
    fields
        The electric and magnetic field each port drives in the body.
    """

    operator: CoupledOperator
    ports: PortSolution
    admittance: torch.Tensor
    impedance: torch.Tensor
    scattering: torch.Tensor
    fields: Fields


def assemble_coil(
    coil: SurfaceCoil | WireCoil | CombinedCoil, medium: Medium
) -> sie.CoilSystem:
    """Build any coil's own system: surface, wire, or both together.

    Parameters
    ----------
    coil
        The coil.
    medium
        Free-space constants at the working frequency.

    Returns
    -------
    mariepy.sie.CoilSystem
        Its matrix, losses and port drive.
    """
    if isinstance(coil, CombinedCoil):
        return wire.assemble_combined(coil, medium)
    if isinstance(coil, WireCoil):
        return wire.assemble(coil, medium)
    return sie.assemble(coil, medium)


def solve(
    body: VoxelBody,
    coil: SurfaceCoil | WireCoil | CombinedCoil,
    medium: Medium,
    *,
    tol: float = 1e-5,
    reference: float = 50.0,
    triangle_order: int = 4,
    cell_order: int = 2,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
    linear: bool = False,
    shield: SurfaceCoil | None = None,
) -> Result:
    """Drive one coil against one body, and return the port matrices and fields.

    Ported from MARIE 3.0's ``src_solver/src_ie_solver/solver_wsvie.m`` and the
    path ``src_runners/MARIE_runner.m`` takes for a surface coil around a voxel
    body with precorrected FFT coupling.

    Parameters
    ----------
    body
        The body and its grid.
    coil
        The coil, its basis and its ports: a surface coil, a wire coil, or
        both together.
    medium
        The frequency to solve at.
    tol
        Target for the preconditioned relative residual of every port's solve,
        and the tolerance the operator compressions are derived from.
    reference
        Line impedance the scattering parameters are referred to, in ohms.
    triangle_order, cell_order
        Quadrature orders of the coupling kernels; for a wire coil,
        ``triangle_order`` is the Gauss points per segment, MARIE's
        ``Quad_order_wie_coup``.
    far_order, medium_order, near_order
        Quadrature orders of the body kernels.
    linear
        Give the body the piecewise-linear basis, as MARIE does when
        ``Basis_Functions_VIE`` is 1.
    shield
        An RF shield around the coil and the body. Its coupling to the body is
        built as tensor trains to ``tol`` times a hundred, as MARIE's
        ``load_inputs.m`` sets ``tol_TT``.

    Returns
    -------
    Result
        The port parameters and the fields.

    """
    system = assemble_coil(coil, medium)
    coupling = pfft.assemble(
        body,
        coil,
        system.impedance,
        medium,
        tol=tol * 1e-2,
        triangle_order=triangle_order,
        cell_order=cell_order,
        far_order=far_order,
        medium_order=medium_order,
        near_order=near_order,
        linear=linear,
    )
    operator = CoupledOperator(
        body=body, coil=coil, medium=medium, system=system, coupling=coupling
    )
    if shield is not None:
        operator = ShieldedOperator(
            coupled=operator,
            shield=shield_module.assemble(
                shield,
                coil,
                body,
                medium,
                tol=tol * 1e2,
                linear=linear,
                triangle_order=triangle_order,
                cell_order=cell_order,
            ),
        )
    ports = solve_ports(operator, tol=tol)
    admittance = network.symmetrise(
        network.port_admittance(
            operator.excitation, operator.conductors(ports.coil, ports.shield)
        )
    )
    impedance = network.y_to_z(admittance)
    return Result(
        operator=operator,
        ports=ports,
        admittance=admittance,
        impedance=impedance,
        scattering=network.z_to_s(impedance, reference),
        fields=fields.compute(operator, ports.coil, ports.body, ports.shield),
    )
