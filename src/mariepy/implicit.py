"""A body solved with the coil folded into it, from one precomputation per coil.

Ported from MARIE 2.0 (GPL-3.0-or-later): the coil-implicit solve of
``src_numerics/+src_solver/Solver_Coil_Implicit.m`` and
``+src_mvp/VSIE_mvp/VSIE_decoupled_*.m``, with its perturbation basis from
``+src_coupling``, following Guryev et al., "MARIE 2.0: a perturbation matrix
based patient-specific MRI field simulator", IEEE TBME 70 (2023) 1575.

Eliminating the coil currents ``Jc`` from the coupled system

    [ Acc   Zbc^T ] [Jc]   [F]
    [ -Zbc  Zbb   ] [Jb] = [0]

leaves an equation in the body current alone,

    (Zbb + Zbc Zc^-1 Zbc^T) Jb = Zbc Zc^-1 F,

whose second term is the coil's perturbation of the body. It depends on the
coil, the frequency and the grid, never on the tissue, so it is compressed once
over the region of a grid a body may occupy and serves every body within it.

The compression departs from MARIE 2.0's, which samples ``Zbc`` by cross
approximation and keeps factors the size of the grid. Here the coupling's range
is found from the coil's side: a randomized Nyström approximation of the Gram
matrix ``Zbc^H Zbc`` gives the right singular vectors of ``Zbc``, and the
perturbation is truncated in that basis. What is kept per coil is therefore a
handful of coil current patterns; each body turns them into its own factors
with one product each, and both factors are taken on the body's own voxels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from mariepy import fields, network, pfft
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.gmres import gmres, refine
from mariepy.preconditioner import body_diagonal
from mariepy.sie import CoilSystem
from mariepy.solver import PortSolution, Result, assemble_coil
from mariepy.system import CoupledOperator
from mariepy.wire import CombinedCoil, WireCoil

__all__ = ["BodyPerturbation", "CoilPerturbation", "tissue_region"]

Coil = SurfaceCoil | WireCoil | CombinedCoil

_log = logging.getLogger(__name__)


def tissue_region(grid: VoxelBody, coil: Coil, clearance: float) -> torch.Tensor:
    """Mark the voxels of a grid that keep a clearance from every conductor.

    Tissue close to a conductor sees the near field of each of its edges, which
    no low-rank perturbation holds, and real bodies keep their distance from a
    coil. A voxel is excluded when a conductor passes within ``clearance`` of
    its centre along every axis.

    Parameters
    ----------
    grid
        The grid bodies will be given on.
    coil
        The conductors.
    clearance
        Distance in metres a voxel keeps from the conductors.

    Returns
    -------
    torch.Tensor
        Shape of the grid, boolean: where a body may put tissue.
    """
    points = _conductor_points(coil, 0.5 * grid.resolution).to(grid.device)
    origin = torch.tensor(grid.origin, dtype=torch.float64, device=grid.device)
    index = torch.round((points - origin) / grid.resolution).to(torch.int64)
    limits = torch.tensor(grid.shape, device=grid.device)
    held = ((index >= 0) & (index < limits)).all(dim=1)
    touched = torch.zeros(grid.shape, dtype=torch.float32, device=grid.device)
    touched[tuple(index[held].T)] = 1.0
    reach = math.ceil(clearance / grid.resolution)
    if reach > 0:
        touched = torch.nn.functional.max_pool3d(
            touched[None, None], 2 * reach + 1, stride=1, padding=reach
        )[0, 0]
    return touched == 0


def _conductor_points(coil: Coil, spacing: float) -> torch.Tensor:
    """Sample every conductor at no more than ``spacing`` apart."""
    if isinstance(coil, CombinedCoil):
        return torch.cat(
            [
                _conductor_points(coil.wire, spacing),
                _conductor_points(coil.surface, spacing),
            ]
        )
    if isinstance(coil, WireCoil):
        starts = torch.cat([coil.first, coil.centre])
        stops = torch.cat([coil.centre, coil.last])
    else:
        nodes = coil.mesh.nodes
        starts, stops = nodes[coil.edges[:, 0]], nodes[coil.edges[:, 1]]
    longest = float(torch.linalg.vector_norm(stops - starts, dim=-1).max())
    steps = max(1, math.ceil(longest / spacing))
    fractions = torch.linspace(
        0.0, 1.0, steps + 1, dtype=starts.dtype, device=starts.device
    )
    samples = (
        starts[:, None, :] + fractions[None, :, None] * (stops - starts)[:, None, :]
    )
    return samples.reshape(-1, 3)


def _region_body(grid: VoxelBody, region: torch.Tensor) -> VoxelBody:
    """Free space on a grid, with a region of it counted as body."""
    return VoxelBody(
        permittivity=torch.ones_like(grid.permittivity),
        conductivity=torch.zeros_like(grid.conductivity),
        mask=region.to(grid.mask.device),
        resolution=grid.resolution,
        origin=grid.origin,
    )


@dataclass(frozen=True)
class CoilPerturbation:
    """One coil's effect on any body within a region of one grid, compressed.

    The perturbation is ``Zbc Zc^-1 Zbc^T ~ (Zbc L)(Zbc R)^T``, with ``L`` and
    ``R`` the columns of :attr:`left` and :attr:`right`.

    Attributes
    ----------
    operator
        The coupled operator over the region a body may occupy.
    impedance
        The coil matrix ``Zc`` the coil is eliminated with.
    factors
        Its LU factorisation.
    drive
        ``Zc^-1 F``, shape ``(n_coil, n_ports)``: the coil current each port
        drives in free space.
    left, right
        Coil current patterns, shape ``(n_coil, rank)`` each.
    singular_values
        The singular values of ``Zbc`` over the region that the basis kept.
    """

    operator: CoupledOperator
    impedance: torch.Tensor
    factors: tuple[torch.Tensor, torch.Tensor]
    drive: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    singular_values: torch.Tensor

    @property
    def rank(self) -> int:
        """Rank the perturbation was kept at."""
        return self.left.shape[1]

    @classmethod
    def build(
        cls,
        grid: VoxelBody,
        coil: Coil,
        medium: Medium,
        *,
        tol: float = 1e-3,
        region: torch.Tensor | None = None,
        block: int = 16,
        impedance: torch.Tensor | None = None,
        system: CoilSystem | None = None,
        linear: bool = False,
        coupling_tol: float = 1e-7,
        **orders,
    ) -> CoilPerturbation:
        """Compress a coil's perturbation over a grid.

        Parameters
        ----------
        grid
            The grid bodies will be given on, and by default, through its mask,
            where they may put tissue: one body, or the union of a population's
            masks. Its tissue properties are ignored.
        coil
            The coil, its basis and its ports.
        medium
            The frequency.
        tol
            Relative tolerance of both truncations: of the coupling's singular
            values against the largest, and of the perturbation's tail against
            its largest singular value.
        region
            Where on the grid bodies may put tissue, boolean, in place of
            ``grid``'s mask; :func:`tissue_region` gives an envelope that keeps a
            clearance from the conductors. The build's cost and the rank the
            perturbation needs both grow with the region, and fastest near the
            conductors.
        block
            Random coil currents drawn at a time while the coupling's range is
            sampled; each costs two products on the extended grid.
        impedance
            The coil matrix to eliminate the coil with. By default the coil's own
            method-of-moments matrix; the coupled operator's coil block makes
            the solve reproduce :func:`mariepy.solver.solve_ports` exactly.
        system
            The coil's system, when it is already assembled.
        linear
            Give the body the piecewise-linear basis.
        coupling_tol
            Tucker tolerance of the precorrected FFT coupling.
        **orders
            Quadrature orders, as :func:`mariepy.pfft.assemble` takes them.

        Returns
        -------
        CoilPerturbation
            Ready to solve any body within the region.
        """
        system = assemble_coil(coil, medium) if system is None else system
        box = _region_body(grid, grid.mask if region is None else region)
        coupling = pfft.assemble(
            box,
            coil,
            system.impedance,
            medium,
            tol=coupling_tol,
            linear=linear,
            **orders,
        )
        operator = CoupledOperator(
            body=box, coil=coil, medium=medium, system=system, coupling=coupling
        )
        impedance = system.impedance if impedance is None else impedance
        factors = torch.linalg.lu_factor(impedance)
        drive = torch.linalg.lu_solve(*factors, system.excitation.transpose(0, 1))

        basis, singular = _coupling_range(operator, tol=tol, block=block)
        # Zbc V = Q S, so the perturbation is Q (S V^H Zc^-1 conj(V) S) Q^T.
        inverse = torch.linalg.lu_solve(*factors, basis.conj())
        core = singular[:, None] * (basis.conj().transpose(0, 1) @ inverse)
        core = core * singular[None, :]
        u, s, vh = torch.linalg.svd(core)
        rank = _tail_rank(s, tol)
        scaled = basis / singular[None, :]
        left = scaled @ (u[:, :rank] * s[None, :rank])
        right = scaled @ vh[:rank].transpose(0, 1)
        return cls(
            operator=operator,
            impedance=impedance,
            factors=factors,
            drive=drive,
            left=left,
            right=right,
            singular_values=singular,
        )

    def prepare(self, body: VoxelBody) -> BodyPerturbation:
        """Turn the coil patterns into one body's factors.

        Parameters
        ----------
        body
            A body on the grid the perturbation was built on.

        Returns
        -------
        BodyPerturbation
            The body's operator, factors and port drives.

        Raises
        ------
        ValueError
            If ``body`` is not on the grid the perturbation was built on, or
            puts tissue outside its region.
        """
        grid = self.operator.body
        if (
            tuple(body.shape) != tuple(grid.shape)
            or not math.isclose(body.resolution, grid.resolution)
            or any(
                not math.isclose(a, b, abs_tol=1e-9 * grid.resolution)
                for a, b in zip(body.origin, grid.origin, strict=True)
            )
        ):
            raise ValueError(
                "the body must be on the grid the perturbation was built on: shape "
                f"{tuple(grid.shape)}, pitch {grid.resolution}, origin {grid.origin}"
            )
        box = self.operator
        operator = CoupledOperator(
            body=body,
            coil=box.coil,
            medium=box.medium,
            system=box.system,
            coupling=pfft.restrict(box.coupling, body.mask),
        )

        # In double precision whatever the solve's: the patterns are scaled by
        # the inverse of the coupling's singular values, and their products
        # cancel to that extent.
        def coupled(patterns: torch.Tensor) -> torch.Tensor:
            columns = [
                operator.couple(patterns[:, k]) for k in range(patterns.shape[1])
            ]
            return torch.stack(columns, dim=1)

        return BodyPerturbation(
            coil=self,
            operator=operator,
            left=coupled(self.left),
            right=coupled(self.right),
            right_hand_side=coupled(self.drive),
        )

    def solve(
        self,
        body: VoxelBody,
        *,
        tol: float = 1e-5,
        restart: int = 50,
        maxit: int = 200,
        precision: str = "mixed",
        reference: float = 50.0,
    ) -> Result:
        """Solve one body on the grid, driven by each port in turn.

        Parameters
        ----------
        body
            A body on the grid the perturbation was built on.
        tol
            Target for the preconditioned relative residual of each port's
            solve.
        restart, maxit
            Passed to the Krylov solver.
        precision
            ``"mixed"`` iterates with the body's products in complex64 and
            finishes in complex128; ``"double"`` stays in complex128.
        reference
            Line impedance the scattering parameters are referred to, in ohms.

        Returns
        -------
        mariepy.solver.Result
            The port matrices and fields, as :func:`mariepy.solver.solve`
            returns them.
        """
        prepared = self.prepare(body)
        return prepared.solve(
            tol=tol,
            restart=restart,
            maxit=maxit,
            precision=precision,
            reference=reference,
        )


@dataclass(frozen=True)
class BodyPerturbation:
    """A coil's perturbation taken onto one body.

    Attributes
    ----------
    coil
        The compressed perturbation it came from.
    operator
        The coupled operator of this body and the coil.
    left, right
        ``Zbc L`` and ``Zbc R`` on the body's degrees of freedom, shape
        ``(n_body, rank)`` each.
    right_hand_side
        ``Zbc Zc^-1 F``, shape ``(n_body, n_ports)``.
    """

    coil: CoilPerturbation
    operator: CoupledOperator
    left: torch.Tensor
    right: torch.Tensor
    right_hand_side: torch.Tensor

    def __call__(self, body_current: torch.Tensor) -> torch.Tensor:
        """Apply ``Zbb`` plus the perturbation to a body current.

        Parameters
        ----------
        body_current
            Shape ``(n_body,)``, complex128 or complex64; the body's own block
            is taken in its precision, the perturbation in complex128.

        Returns
        -------
        torch.Tensor
            Same shape and precision.
        """
        own = self.operator.body_block(body_current)
        double = body_current.to(torch.complex128)
        perturbation = self.left @ (self.right.transpose(0, 1) @ double)
        return own + perturbation.to(body_current.dtype)

    def solve(
        self,
        *,
        tol: float = 1e-5,
        restart: int = 50,
        maxit: int = 200,
        precision: str = "mixed",
        reference: float = 50.0,
    ) -> Result:
        """Solve the body for each port, and recover the coil and the fields.

        Parameters
        ----------
        tol, restart, maxit, precision, reference
            As :meth:`CoilPerturbation.solve` takes them.

        Returns
        -------
        mariepy.solver.Result
            The port matrices and fields.
        """
        operator = self.operator
        diagonal = body_diagonal(
            operator.body, operator.medium, linear=operator.coupling.linear
        )

        def precondition(vector: torch.Tensor) -> torch.Tensor:
            return diagonal.to(vector.dtype) * vector

        if precision == "double":
            krylov = gmres
        elif precision == "mixed":
            krylov = refine
        else:
            raise ValueError(f"precision is 'double' or 'mixed', got {precision!r}")

        excitation = operator.system.excitation
        bodies, coils, residual, iterations = [], [], [], []
        for port in range(self.right_hand_side.shape[1]):
            solution = krylov(
                self,
                self.right_hand_side[:, port].contiguous(),
                preconditioner=precondition,
                restart=restart,
                tol=tol,
                maxit=maxit,
            )
            body_current = solution.x
            back = operator.couple_transpose(body_current)
            coils.append(
                torch.linalg.lu_solve(
                    *self.coil.factors, (excitation[port] - back)[:, None]
                )[:, 0]
            )
            bodies.append(body_current)
            residual.append(float(solution.residuals[-1]))
            iterations.append(len(solution.residuals) - 1)

        ports = PortSolution(
            coil=torch.stack(coils),
            body=torch.stack(bodies),
            residual=tuple(residual),
            iterations=tuple(iterations),
        )
        admittance = network.symmetrise(network.port_admittance(excitation, ports.coil))
        impedance = network.y_to_z(admittance)
        return Result(
            operator=operator,
            ports=ports,
            admittance=admittance,
            impedance=impedance,
            scattering=network.z_to_s(impedance, reference),
            fields=fields.compute(operator, ports.coil, ports.body),
        )


def _coupling_range(
    operator: CoupledOperator, *, tol: float, block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the coupling's right singular vectors above a relative tolerance.

    A randomized Nyström approximation of ``G = Zbc^H Zbc`` (Tropp et al., SIAM
    J. Matrix Anal. Appl. 38 (2017) 1454) grows by ``block`` columns until its
    smallest kept eigenvalue falls below ``tol**2`` times its largest.

    Returns
    -------
    basis : torch.Tensor
        Shape ``(n_coil, q)``, orthonormal columns.
    singular : torch.Tensor
        Shape ``(q,)``, float64 as complex128: the singular values of ``Zbc``.
    """
    n_coil = operator.n_coil
    device = operator.system.impedance.device
    generator = torch.Generator().manual_seed(0)

    def gram(column: torch.Tensor) -> torch.Tensor:
        field = operator.couple(column)
        return operator.couple_transpose(field.conj()).conj()

    test = torch.zeros((n_coil, 0), dtype=torch.complex128, device=device)
    sketch = torch.zeros_like(test)
    while True:
        width = min(block, n_coil - test.shape[1])
        fresh = torch.complex(
            torch.randn((n_coil, width), generator=generator, dtype=torch.float64),
            torch.randn((n_coil, width), generator=generator, dtype=torch.float64),
        ).to(device)
        for _ in range(2):
            fresh = fresh - test @ (test.conj().transpose(0, 1) @ fresh)
        fresh, _ = torch.linalg.qr(fresh)
        test = torch.cat([test, fresh], dim=1)
        sketch = torch.cat(
            [sketch, torch.stack([gram(fresh[:, k]) for k in range(width)], dim=1)],
            dim=1,
        )
        vectors, values = _nystrom(test, sketch)
        kept = values > (tol**2) * values[0]
        _log.info(
            "coupling range: %d coil currents sampled, %d singular values above %g",
            test.shape[1],
            int(kept.sum()),
            tol,
        )
        if int(kept.sum()) <= test.shape[1] - block // 2 or test.shape[1] >= n_coil:
            break
    singular = torch.sqrt(values[kept]).to(torch.complex128)
    return vectors[:, kept], singular


def _nystrom(
    test: torch.Tensor, sketch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give the eigenpairs of a PSD matrix from its action on orthonormal columns."""
    shift = (
        math.sqrt(test.shape[0])
        * torch.finfo(torch.float64).eps
        * float(torch.linalg.matrix_norm(sketch, ord=2))
    )
    shifted = sketch + shift * test
    core = test.conj().transpose(0, 1) @ shifted
    core = 0.5 * (core + core.conj().transpose(0, 1))
    lower = torch.linalg.cholesky(core)
    factor = torch.linalg.solve_triangular(
        lower.conj().transpose(0, 1), shifted, upper=True, left=False
    )
    vectors, singular, _ = torch.linalg.svd(factor, full_matrices=False)
    values = torch.clamp(singular**2 - shift, min=0.0)
    return vectors, values


def _tail_rank(singular: torch.Tensor, tol: float) -> int:
    """Smallest rank whose discarded singular values have norm within ``tol``."""
    tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(singular**2, (0,)), 0), (0,)))
    beyond = torch.cat([tail[1:], torch.zeros(1, dtype=tail.dtype, device=tail.device)])
    return max(1, int((beyond > tol * singular[0]).sum()) + 1)
