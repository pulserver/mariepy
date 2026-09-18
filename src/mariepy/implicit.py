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

``Zbc`` over the region is approximated first, by MARIE 2.0's cross
approximation: rows and columns of it are integrated directly, at one kernel
evaluation per entry and no convolution, and the sampling follows the coupling
rather than the region, taking each round's columns where the coil-side basis
has the largest volume. The perturbation's core is then a small matrix between
the coupling's two factors, ``Zbc ~ B C``, and truncating

    B (C Zc^-1 C^T) B^T

leaves the factors over the whole region, so a body only selects the rows of
the voxels it occupies and no product on the extended grid is taken per body.

The coupling this holds is the one the quadrature gives, where the coupled
operator of :mod:`mariepy.system` projects its far interactions onto the
grid. The two therefore differ by that projection, as they do already in the
coil matrix the coil is eliminated with, and what the approximation leaves of
the operator's own coupling is measured on every build.
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
from mariepy.tt import _maxvol
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


def _chebyshev_cells(grid: pfft.ExtendedGrid, points: int) -> torch.Tensor:
    """Take the region's cells on a Chebyshev grid of ``points`` a side.

    Ported from ``sample_rows.m``. Chebyshev nodes over the region's own extent
    crowd towards its faces, which is where the coupling varies fastest, and
    the tensor grid of them is intersected with the region.
    """
    region = grid.mask
    held = torch.nonzero(region)
    axes = []
    for axis in range(3):
        first = int(held[:, axis].min())
        last = int(held[:, axis].max())
        nodes = torch.cos(torch.linspace(0.0, math.pi, points + 1, dtype=torch.float64))
        index = torch.round(first + 0.5 * (last - first) * (1.0 + nodes))
        axes.append(torch.unique(index.to(torch.int64)))
    block = torch.cartesian_prod(*axes).to(region.device)
    inside = region[block[:, 0], block[:, 1], block[:, 2]]
    return grid.flatten(block[inside])


def _row_numbers(
    grid: pfft.ExtendedGrid, cells: torch.Tensor, n_components: int
) -> torch.Tensor:
    """Give named cells their unknowns' numbers, as the body numbers its own.

    Ported from the row numbering ``sample_rows.m`` builds: each component runs
    over every cell of the region, one component after the next.
    """
    within = pfft.body_numbering(grid)[cells]
    n_region = int(grid.mask.sum())
    offsets = n_region * torch.arange(n_components, device=cells.device)
    return (offsets[:, None] + within[None, :]).reshape(-1)


def _region_rows(
    region: torch.Tensor, mask: torch.Tensor, n_components: int
) -> torch.Tensor:
    """Give the rows of a region's unknowns that belong to a body within it.

    A grid's unknowns run component by component over the cells its mask keeps,
    in the mask's own order, so a body's rows are the region rows of the cells
    it occupies, repeated for each component.
    """
    inside = mask.to(region.device)[region]
    cells = torch.nonzero(inside, as_tuple=True)[0]
    offsets = int(region.sum()) * torch.arange(n_components, device=cells.device)
    return (offsets[:, None] + cells[None, :]).reshape(-1)


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

    The perturbation is ``Zbc Zc^-1 Zbc^T``, kept as the product of
    :attr:`region_left` and the transpose of :attr:`region_right`.

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
    coil_side
        The coupling's coil-side factor, shape ``(rank, n_coil)``: with
        :attr:`region_left` and :attr:`region_right` it is what the cross
        approximation of ``Zbc`` gave.
    region_left, region_right
        ``Zbc L`` and ``Zbc R`` over the whole region, shape
        ``(n_region, rank)`` each, in the precision :meth:`build` stored them
        in. A body takes its own factors from their rows.
    region_drive
        ``Zbc Zc^-1 F`` over the whole region, shape ``(n_region, n_ports)``,
        complex128: it is the right-hand side each solve starts from.
    singular_values
        The singular values of ``Zbc`` over the region that the basis kept.
    """

    operator: CoupledOperator
    impedance: torch.Tensor
    factors: tuple[torch.Tensor, torch.Tensor]
    drive: torch.Tensor
    coil_side: torch.Tensor
    region_left: torch.Tensor
    region_right: torch.Tensor
    region_drive: torch.Tensor
    singular_values: torch.Tensor

    @property
    def rank(self) -> int:
        """Rank the perturbation was kept at."""
        return self.region_left.shape[1]

    @classmethod
    def build(
        cls,
        grid: VoxelBody,
        coil: Coil,
        medium: Medium,
        *,
        tol: float = 1e-3,
        region: torch.Tensor | None = None,
        block: int = 100,
        iterations: int = 30,
        stalls: int = 10,
        checks: int = 2,
        impedance: torch.Tensor | None = None,
        system: CoilSystem | None = None,
        linear: bool = False,
        coupling_tol: float = 1e-7,
        store: torch.dtype = torch.complex64,
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
            Coil unknowns the cross approximation starts from, MARIE 2.0's
            first rank guess, which also sets the first Chebyshev grid.
        iterations
            Rounds of sampling at most.
        stalls
            Rounds whose rank barely moves before the sampling gives up.
        checks
            Random coil currents the approximation is measured against once it
            is found, each costing two products on the extended grid.
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
        store
            Precision the region-sized factors are kept in. They hold
            ``rank`` vectors over the region, so complex64 halves what the
            build carries, at a rounding far below ``tol``. Their right-hand
            side stays complex128.
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

        body, coil_side, singular = _cross_coupling(
            operator,
            tol=tol,
            rank=block,
            iterations=iterations,
            stalls=stalls,
            checks=checks,
            orders={
                name: value
                for name, value in orders.items()
                if name in ("triangle_order", "cell_order")
            },
        )
        # Zbc ~ body coil_side, so the perturbation is body C body^T with
        # C = coil_side Zc^-1 coil_side^T, which is small and truncates there.
        core = coil_side @ torch.linalg.lu_solve(*factors, coil_side.transpose(0, 1))
        u, s, vh = torch.linalg.svd(core)
        rank = _tail_rank(s, tol)
        _log.info("perturbation rank %d of the coupling's %d", rank, core.shape[0])
        return cls(
            operator=operator,
            impedance=impedance,
            factors=factors,
            drive=drive,
            coil_side=coil_side,
            region_left=(body @ (u[:, :rank] * s[None, :rank])).to(store),
            region_right=(body @ vh[:rank].transpose(0, 1)).to(store),
            region_drive=body @ (coil_side @ drive),
            singular_values=singular,
        )

    def prepare(self, body: VoxelBody) -> BodyPerturbation:
        """Take one body's factors from the region's, by its own rows.

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
        rows = _region_rows(grid.mask, body.mask, box.coupling.n_components)
        return BodyPerturbation(
            coil=self,
            operator=operator,
            left=self.region_left[rows],
            right=self.region_right[rows],
            right_hand_side=self.region_drive[rows],
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
        ``(n_body, rank)`` each, in the precision the build stored them in.
    right_hand_side
        ``Zbc Zc^-1 F``, shape ``(n_body, n_ports)``, complex128.
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
            is taken in its precision, the perturbation in the factors'.

        Returns
        -------
        torch.Tensor
            Same shape and precision.
        """
        return self.operator.body_block(body_current) + self.perturb(body_current)

    def perturb(self, body_current: torch.Tensor) -> torch.Tensor:
        """Apply the compressed ``Zbc Zc^-1 Zbc^T`` to a body current.

        Parameters
        ----------
        body_current
            Shape ``(n_body,)``, complex128 or complex64.

        Returns
        -------
        torch.Tensor
            Same shape and precision; the product is taken in the factors'.
        """
        current = body_current.to(self.left.dtype)
        product = self.left @ (self.right.transpose(0, 1) @ current)
        return product.to(body_current.dtype)

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


def _cross_coupling(
    operator: CoupledOperator,
    *,
    tol: float,
    rank: int,
    iterations: int,
    stalls: int,
    checks: int,
    orders: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate the coupling over the region from rows and columns of it.

    Ported from MARIE 2.0's ``cross_cheb_2d.m`` with ``sample_rows.m``,
    ``sample_cols.m`` and ``maxvol.m``, and the Frobenius termination criterion
    of ``Sampling_approx_Base.m``. The coupling is never applied: its rows and
    columns are integrated directly, one kernel evaluation per entry and no
    convolution.

    Sampled columns span a body-side basis and sampled rows a coil-side basis,
    both orthonormalised, and the coupling's core is fitted on the sampled
    submatrix through their pseudo-inverses over those rows and columns. Each
    round adds the columns where the coil-side basis has the largest volume,
    which is what makes the sampling follow the coupling rather than the
    region, and the rows of a Chebyshev grid one step finer. It stops once the
    core's singular values settle to ``tol``.

    Where MARIE 2.0 subsamples the rows of the chosen cells at random to bound
    their number, every component of a chosen cell is kept here: the rows only
    have to span the row space, and keeping them whole leaves the submatrix in
    the order the body numbers its own unknowns.

    Returns
    -------
    body : torch.Tensor
        Shape ``(n_region, rank)``: the coupling's body-side factor.
    coil : torch.Tensor
        Shape ``(rank, n_coil)``: its coil-side factor, so that their product
        is the coupling.
    singular : torch.Tensor
        The core's singular values, which are the coupling's own.
    """
    grid = operator.coupling.grid
    region_cells = grid.body_cells()
    n_components = operator.coupling.n_components
    n_coil = operator.n_coil
    generator = torch.Generator().manual_seed(0)

    def sample(cells: torch.Tensor, dofs: torch.Tensor | None) -> torch.Tensor:
        return pfft.coupling_rows(
            grid,
            operator.coil,
            operator.medium,
            cells,
            dofs=dofs,
            linear=operator.coupling.linear,
            **orders,
        )

    points = 2 * math.ceil(rank ** (1.0 / 3.0))
    cells = _chebyshev_cells(grid, points)
    dofs = torch.sort(
        torch.randperm(n_coil, generator=generator)[:rank].to(grid.device)
    ).values
    body_basis, _, _ = torch.linalg.svd(sample(region_cells, dofs), full_matrices=False)
    coil_basis, _, _ = torch.linalg.svd(
        sample(cells, None).conj().transpose(0, 1), full_matrices=False
    )

    def fit() -> tuple:
        rows = _row_numbers(grid, cells, n_components)
        core, body_map, coil_map = _fitted_core(
            body_basis[rows], coil_basis[dofs], sample(cells, dofs)
        )
        u, values, vh = torch.linalg.svd(core, full_matrices=False)
        held = _tail_rank(values, tol)
        _log.info(
            "cross approximation: %d rows, %d columns, rank %d",
            int(rows.numel()),
            int(dofs.numel()),
            held,
        )
        return u, values, vh, body_map, coil_map, held

    kept = fit()
    before, before_rank = kept[1], kept[5]
    stalled = 0
    for step in range(2, iterations + 1):
        fresh_dofs = _outside(
            torch.tensor(_maxvol(coil_basis), device=grid.device), dofs
        )
        fresh_cells = _outside(_chebyshev_cells(grid, points + 2 * (step - 1)), cells)
        if fresh_dofs.numel() == 0 and fresh_cells.numel() == 0:
            break
        if fresh_dofs.numel():
            body_basis, _, _ = torch.linalg.svd(
                torch.cat([body_basis, sample(region_cells, fresh_dofs)], dim=1),
                full_matrices=False,
            )
            dofs = torch.sort(torch.cat([dofs, fresh_dofs])).values
        if fresh_cells.numel():
            coil_basis, _, _ = torch.linalg.svd(
                torch.cat(
                    [
                        coil_basis,
                        sample(fresh_cells, None).conj().transpose(0, 1),
                    ],
                    dim=1,
                ),
                full_matrices=False,
            )
            cells = torch.sort(torch.cat([cells, fresh_cells])).values

        kept = fit()
        singular, rank = kept[1], kept[5]
        if _settled(before, singular, before_rank, rank, tol) or stalled >= stalls:
            break
        stalled = stalled + 1 if abs(rank - before_rank) < 2 else 0
        before, before_rank = singular, rank

    u, singular, vh, body_map, coil_map, rank = kept
    body = body_basis @ (body_map @ (u[:, :rank] * singular[None, :rank]))
    coil = (vh[:rank] @ coil_map.conj().transpose(0, 1)) @ coil_basis.conj().transpose(
        0, 1
    )
    _log.info(
        "cross approximation: rank %d from %d cells and %d coil unknowns, "
        "leaving %.2e of the coupling's action",
        rank,
        int(cells.numel()),
        int(dofs.numel()),
        _missed(operator, body, coil, checks),
    )
    return body, coil, singular[:rank]


def _fitted_core(
    body_rows: torch.Tensor, coil_rows: torch.Tensor, submatrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit the coupling's core on a sampled submatrix, through pseudo-inverses.

    Ported from the ``M`` of ``cross_cheb_2d.m``: the submatrix is taken into
    the two bases' coordinates, each through the pseudo-inverse of its own rows
    over the sample.

    Returns
    -------
    core : torch.Tensor
        The submatrix in the two bases' coordinates.
    body_map, coil_map : torch.Tensor
        What carries the core's factors back onto the bases.
    """
    exact = 1e-15
    qu, su, vu = torch.linalg.svd(body_rows, full_matrices=False)
    qv, sv, vv = torch.linalg.svd(coil_rows.conj().transpose(0, 1), full_matrices=False)
    keep_u, keep_v = _tail_rank(su, exact), _tail_rank(sv, exact)
    left = qu[:, :keep_u].conj().transpose(0, 1) / su[:keep_u, None].to(qu.dtype)
    right = vv[:keep_v].conj().transpose(0, 1) / sv[None, :keep_v].to(qv.dtype)
    return left @ submatrix @ right, vu[:keep_u].conj().transpose(0, 1), qv[:, :keep_v]


def _settled(
    before: torch.Tensor,
    now: torch.Tensor,
    before_rank: int,
    rank: int,
    tol: float,
) -> bool:
    """Whether the core's singular values have stopped moving.

    Ported from ``Frobenius_termination_criteria``: what the spectrum moved
    between rounds, together with what the new rank added to it, against ``tol``
    of its norm.
    """
    if before.numel() == 0:
        return False
    common = min(rank, before_rank, int(before.numel()), int(now.numel()))
    moved = torch.linalg.vector_norm(before[:common] - now[:common])
    fresh = (
        torch.linalg.vector_norm(now[common:rank])
        if before_rank < rank
        else torch.zeros((), dtype=moved.dtype, device=moved.device)
    )
    reached = torch.sqrt(moved**2 + fresh**2)
    return bool(tol * torch.linalg.vector_norm(now[:rank]) > reached)


def _outside(candidate: torch.Tensor, held: torch.Tensor) -> torch.Tensor:
    """Those of ``candidate`` that ``held`` does not already carry."""
    fresh = torch.unique(candidate)
    if held.numel() == 0:
        return fresh
    return fresh[~torch.isin(fresh, held)]


def _missed(
    operator: CoupledOperator,
    body: torch.Tensor,
    coil: torch.Tensor,
    checks: int,
) -> float:
    """Give what the approximated coupling leaves of the operator's own, at worst.

    The coupling through the operator is what the approximation has to hold, so
    that is what it is measured against, on random coil currents.
    """
    generator = torch.Generator().manual_seed(1)
    shape = (operator.n_coil,)
    worst = 0.0
    for _ in range(checks):
        current = torch.complex(
            torch.randn(shape, generator=generator, dtype=torch.float64),
            torch.randn(shape, generator=generator, dtype=torch.float64),
        ).to(body.device)
        whole = operator.couple(current)
        held = body @ (coil @ current)
        worst = max(
            worst,
            float(
                torch.linalg.vector_norm(whole - held) / torch.linalg.vector_norm(whole)
            ),
        )
    return worst


def _tail_rank(singular: torch.Tensor, tol: float) -> int:
    """Smallest rank whose discarded singular values have norm within ``tol``."""
    tail = torch.sqrt(torch.flip(torch.cumsum(torch.flip(singular**2, (0,)), 0), (0,)))
    beyond = torch.cat([tail[1:], torch.zeros(1, dtype=tail.dtype, device=tail.device)])
    return max(1, int((beyond > tol * singular[0]).sum()) + 1)
