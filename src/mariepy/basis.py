"""Field bases for a body, and coil solves reduced onto them (MRGF).

Ported from MARIE 3.0's basis path: ``src_runners/BASIS_runner.m``,
``src_integral_equations/src_basis/sSVD_rwgBasis.m``, ``rSVD_dipoleBasis.m``,
``rSVD_Q.m``, ``rSVD_Q_adjoint.m`` and ``rSVD_deim.m``,
``src_geometry/basis_geometry/geo_dipole_supports/geo_ultimate_basis.m``,
``src_solver/src_ie_solver/ie_solver_vie/ie_solver_vie_basis.m`` and
``src_physics/src_electromagnetism/em_ehfield_vie.m``; and its MRGF path:
``src_runners/MRGF_runner.m``, ``svie_mrgf_assembly.m``,
``ie_solver_svie_mrgf.m`` and ``em_efield_svie_mrgf.m``.

A basis is built once per body. The fields a support surface's RWG currents
put on the body, or those of every current in a shell of voxels around it, are
compressed by SVD into incident fields; a discrete
empirical interpolation picks the voxels that pin a field's coefficients; and
the body is solved once per basis field, giving each its total field and its
current. A coil near the support then needs its coupling at the picked voxels
only: the body's whole response enters its matrix as a small dense term.

The basis is kept in field coefficients. MARIE keeps the incident basis in
tested form, the Gram matrix applied, and solves the body with it as if it
were a field; for the piecewise-constant basis the Gram matrix is a scalar and
the two agree, for the piecewise-linear one only the field form spans the
fields a coil puts on the body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from mariepy import vie
from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil
from mariepy.constants import Medium
from mariepy.solver import BodyOperator, solve_body
from mariepy.tucker import circulant_tucker

__all__ = [
    "FieldBasis",
    "MrgfSolution",
    "coupling_at",
    "dipole_basis",
    "read_marie",
    "shell",
    "solve",
    "solve_coil",
    "surface_basis",
    "ultimate_maps",
]


@dataclass(frozen=True)
class FieldBasis:
    """A body's field basis, before and after the body is solved.

    Every field is a row over the body's degrees of freedom, ordered as
    :meth:`mariepy.body.VoxelBody.to_dof` orders them, in field coefficients.

    Attributes
    ----------
    incident_electric, incident_magnetic
        The incident fields, MARIE's ``Ue`` and ``Ub`` from the SVD, shape
        ``(r, c * n_voxels)``.
    samples
        The voxels the interpolation reads, as indices into the body's voxel
        order, MARIE's ``xds``, ``yds``, ``zds``.
    interpolation
        From an incident field's values at the samples, all ``c`` components
        of each, to its coefficients: MARIE's ``X``, shape
        ``(r, c * n_samples)``.
    singular_values
        The kept singular values of the magnetic coupling, MARIE's ``SK``.
    linear
        Whether the body carries the piecewise-linear basis.
    electric, magnetic
        The total fields, MARIE's ``Ue`` and ``Ub`` after ``em_ehfield_vie.m``,
        or None before :func:`solve`.
    current
        The body current each incident field drives, or None before
        :func:`solve`.
    response
        The body's response at the samples, MARIE's ``Zbb_hat_inv`` in field
        form, shape ``(c * n_samples, c * n_samples)``, or None.
    covariance
        The basis fields' noise covariance, MARIE's ``BASIS.phi``, shape
        ``(r, r)``, or None.
    tested
        Whether the basis is in MARIE's own form, as :func:`read_marie`
        reads it: fields and interpolation carry the Gram matrix, the
        response is MARIE's ``Zbb_hat_inv``, and the incident fields and body
        currents are not stored.
    """

    incident_electric: torch.Tensor | None
    incident_magnetic: torch.Tensor | None
    samples: torch.Tensor
    interpolation: torch.Tensor
    singular_values: torch.Tensor | None
    linear: bool = False
    electric: torch.Tensor | None = None
    magnetic: torch.Tensor | None = None
    current: torch.Tensor | None = None
    response: torch.Tensor | None = None
    covariance: torch.Tensor | None = None
    tested: bool = False

    @property
    def rank(self) -> int:
        """Number of basis fields."""
        fields = (
            self.electric if self.incident_electric is None else self.incident_electric
        )
        return int(fields.shape[0])

    @property
    def n_components(self) -> int:
        """Unknowns per voxel: 3, or 12 for the linear basis."""
        return 12 if self.linear else 3

    def save(self, path) -> None:
        """Write the basis to a file :meth:`load` reads back.

        Parameters
        ----------
        path
            File to write, conventionally with a ``.pt`` suffix.
        """
        torch.save(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if getattr(self, name) is not None
            },
            path,
        )

    @classmethod
    def load(cls, path, *, device=None) -> FieldBasis:
        """Read a basis :meth:`save` wrote.

        Parameters
        ----------
        path
            File to read.
        device
            Device to place the tensors on; where they were saved by default.

        Returns
        -------
        FieldBasis
            The basis.
        """
        return cls(**torch.load(path, map_location=device, weights_only=True))


def _field(coil, dofs, points, medium, *, magnetic, order, **arguments):
    from mariepy.pfft import _field as pfft_field

    return pfft_field(
        coil, dofs, points, medium, magnetic=magnetic, order=order, **arguments
    )


def coupling_at(
    coil,
    points: torch.Tensor,
    medium: Medium,
    resolution: float,
    *,
    magnetic: bool = False,
    linear: bool = False,
    triangle_order: int = 4,
    cell_order: int = 2,
) -> torch.Tensor:
    """Integrate every coil basis function against the cells at some points.

    The tested coupling, cell-averaged and multiplied by the cell volume, as
    ``svie_mrgf_assembly.m`` and ``sSVD_rwgBasis.m`` assemble it.

    Parameters
    ----------
    coil
        A surface coil, a wire coil, or both together.
    points
        Cell centres, shape ``(n_points, 3)``.
    medium
        Supplies the wavenumber.
    resolution
        Cell pitch in metres.
    magnetic
        Give the K coupling rather than the N one.
    linear
        Test against the piecewise-linear basis.
    triangle_order, cell_order
        Quadrature orders.

    Returns
    -------
    torch.Tensor
        Shape ``(c * n_points, n_dof)``, components major, as the body's
        degrees of freedom are ordered.
    """
    terms = range(4) if linear else range(1)
    n_points, n_dof = points.shape[0], coil.n_dof
    dofs = torch.arange(n_dof, device=points.device).repeat(n_points)
    where = points.repeat_interleave(n_dof, dim=0)
    stacked = torch.stack(
        [
            _field(
                coil,
                dofs,
                where,
                medium,
                magnetic=magnetic,
                order=triangle_order,
                cell_size=resolution,
                cell_order=cell_order,
                basis_term=term,
            )
            for term in terms
        ],
        dim=-1,
    )  # (points * dofs, 3, terms)
    stacked = stacked.reshape(n_points, n_dof, 3 * len(terms))
    return resolution**3 * stacked.permute(2, 0, 1).reshape(-1, n_dof)


def _inverse_mass(n_components, n_points, resolution, device):
    mass = vie.mass(n_components, resolution).to(device)
    return (1.0 / mass).repeat_interleave(n_points).to(torch.complex128)


def _truncate(singular_values, tol):
    """``compress_SVD.m``: keep the values above ``tol`` times the largest."""
    return max(1, int((singular_values / singular_values[0] > tol).sum()))


def _deim(vectors: torch.Tensor) -> list[int]:
    """``rSVD_deim.m``: the greedy interpolation indices of the columns."""
    first = int(vectors[:, 0].abs().argmax())
    chosen = [first]
    for j in range(1, vectors.shape[1]):
        index = torch.tensor(chosen, device=vectors.device)
        coefficients = torch.linalg.lstsq(
            vectors[index, :j], vectors[index, j : j + 1]
        ).solution
        residual = vectors[:, j : j + 1] - vectors[:, :j] @ coefficients
        chosen.append(int(residual.abs().argmax()))
    return sorted(chosen)


def surface_basis(
    body: VoxelBody,
    support: SurfaceCoil,
    medium: Medium,
    *,
    tol: float = 1e-3,
    interpolation_tol: float = 1e-4,
    linear: bool = False,
    triangle_order: int = 4,
    cell_order: int = 2,
) -> FieldBasis:
    """Build the incident-field basis a support surface spans, as ``sSVD_rwgBasis.m``.

    Parameters
    ----------
    body
        The body.
    support
        The support surface, MARIE's ``SurfaceBasisSupportFile``.
    medium
        Free-space constants at the working frequency.
    tol
        Relative singular value below which fields are dropped, MARIE's
        ``tol_sSVD``, a hundred times its solver tolerance.
    interpolation_tol
        The same for the fields the interpolation points are picked from,
        MARIE's ``tol_DEIM``, ten times its solver tolerance.
    linear
        Build it over the piecewise-linear basis.
    triangle_order, cell_order
        Quadrature orders of the coupling.

    Returns
    -------
    FieldBasis
        The incident basis, not yet solved.
    """
    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    n_components = 12 if linear else 3
    n_voxels = centres.shape[0]
    inverse = _inverse_mass(n_components, n_voxels, body.resolution, centres.device)
    arguments = {
        "linear": linear,
        "triangle_order": triangle_order,
        "cell_order": cell_order,
    }
    magnetic = coupling_at(
        support, centres, medium, body.resolution, magnetic=True, **arguments
    )
    electric = coupling_at(support, centres, medium, body.resolution, **arguments)

    left, values, right_h = torch.linalg.svd(magnetic, full_matrices=False)
    right = right_h.conj().transpose(0, 1)

    picked = _truncate(values, interpolation_tol)
    rows = _deim(inverse[:, None] * left[:, :picked])
    voxels = sorted({row % n_voxels for row in rows})
    samples = torch.tensor(voxels, dtype=torch.long, device=centres.device)

    kept = _truncate(values, tol)
    combine = right[:, :kept] / values[:kept].to(right.dtype)
    incident_electric = (inverse[:, None] * (electric @ combine)).transpose(0, 1)
    incident_magnetic = (inverse[:, None] * (magnetic @ combine)).transpose(0, 1)

    sampled = _sampled(incident_electric, samples, n_voxels, n_components)
    interpolation = torch.linalg.pinv(sampled)  # (r, c * n_samples)
    return FieldBasis(
        incident_electric=incident_electric.contiguous(),
        incident_magnetic=incident_magnetic.contiguous(),
        samples=samples,
        interpolation=interpolation,
        singular_values=values[:kept],
        linear=linear,
    )


def _convex_hull_slices(mask: torch.Tensor) -> torch.Tensor:
    """Fill each slice along the last axis to its convex hull, as ``bwconvhull``.

    A pixel joins the hull when its centre lies in the convex hull of the
    object's pixel corners.
    """
    import numpy as np
    from scipy.spatial import Delaunay, QhullError

    filled = mask.cpu().numpy().copy()
    rows, columns = np.meshgrid(
        np.arange(filled.shape[0]), np.arange(filled.shape[1]), indexing="ij"
    )
    centres = np.stack([rows.ravel(), columns.ravel()], axis=1)
    corners = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    for k in range(filled.shape[2]):
        inside = np.argwhere(filled[:, :, k])
        if inside.size == 0:
            continue
        points = (inside[:, None, :] + corners[None, :, :]).reshape(-1, 2)
        try:
            hull = Delaunay(points)
        except QhullError:
            continue
        found = hull.find_simplex(centres) >= 0
        filled[:, :, k] |= found.reshape(filled.shape[:2])
    return torch.from_numpy(filled).to(mask.device)


def _ball(radius: int):
    import numpy as np

    steps = np.arange(-radius, radius + 1)
    x, y, z = np.meshgrid(steps, steps, steps, indexing="ij")
    return x**2 + y**2 + z**2 <= radius**2


def shell(
    mask: torch.Tensor, resolution: float, *, distance: float, thickness: int
) -> tuple[torch.Tensor, int]:
    """Place the dipole shell around a body, as ``geo_ultimate_basis.m``.

    The body's slices along the last axis are filled to their convex hulls;
    the shell is what dilating that by ``thickness`` voxels more than the
    distance adds, the object itself left out.

    Parameters
    ----------
    mask
        The body's mask, shape ``(n1, n2, n3)``.
    resolution
        Voxel pitch in metres.
    distance
        Gap between the body and the shell in metres, MARIE's
        ``Basis_distance``.
    thickness
        Shell thickness in voxels, MARIE's ``Basis_thickness``.

    Returns
    -------
    shell : torch.Tensor
        The shell on the padded grid, shape ``(n1 + 2 p, n2 + 2 p, n3 + 2 p)``.
    padding : int
        ``p``, the voxels added on every side; the body sits at ``p`` onwards.
    """
    from scipy.ndimage import binary_dilation

    gap = math.floor(distance / resolution)
    padding = gap + thickness
    padded = torch.nn.functional.pad(mask.to(torch.bool), (padding,) * 6, value=False)
    hull = _convex_hull_slices(padded).cpu().numpy()
    inner = binary_dilation(hull, structure=_ball(gap)) if gap else hull
    outer = binary_dilation(hull, structure=_ball(gap + thickness))
    result = outer & ~inner & ~hull
    return torch.from_numpy(result).to(mask.device), padding


def _range(apply, n_columns, n_rows, tol, block, generator, device):
    """``rSVD_Q.m``: grow a random sample of the operator until its rank shows.

    Blocks of random unit vectors are applied until the sampled matrix has a
    singular value below ``tol`` times its largest; the left singular vectors
    up to that one are its range. The sample stops growing once it has as many
    columns as the operator has columns or rows.
    """
    sample = torch.zeros((n_rows, 0), dtype=torch.complex128, device=device)
    limit = min(n_columns, n_rows)
    while True:
        size = min(block, max(1, limit - sample.shape[1]))
        vectors = torch.complex(
            torch.randn(n_columns, size, generator=generator, dtype=torch.float64),
            torch.randn(n_columns, size, generator=generator, dtype=torch.float64),
        ).to(device)
        vectors = vectors / torch.linalg.vector_norm(vectors, dim=0)
        sample = torch.cat([sample, apply(vectors.T).T], dim=1)
        left, values, _ = torch.linalg.svd(sample, full_matrices=False)
        small = (values <= tol * values[0]).nonzero()
        if small.numel():
            return left[:, : int(small[0]) + 1]
        if sample.shape[1] >= limit:
            return left


def dipole_basis(
    body: VoxelBody,
    medium: Medium,
    *,
    distance: float = 0.02,
    thickness: int = 3,
    tol: float = 1e-3,
    interpolation_tol: float = 1e-4,
    block: int = 1000,
    linear: bool = False,
    seed: int = 0,
    kernel_tol: float = 1e-7,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
) -> FieldBasis:
    """Build the incident-field basis of every current in a shell, as ``rSVD_dipoleBasis.m``.

    A randomised SVD of the magnetic operator from the shell to the body gives
    the magnetic basis and the shell currents behind it; the electric basis is
    the same currents' electric field. The kept rank must stay well below the
    body's unknowns: once it nears them, the electric fields of the kept
    currents turn linearly dependent and the noise covariance singular.

    Parameters
    ----------
    body
        The body.
    medium
        Free-space constants at the working frequency.
    distance, thickness
        Where the shell sits, as :func:`shell` takes them.
    tol
        Relative singular value below which fields are dropped, MARIE's
        ``tol_rSVD``.
    interpolation_tol
        The same for the random range finder, MARIE's ``tol_DEIM``.
    block
        Random vectors drawn at a time, MARIE's ``rSVD_blocksize``.
    linear
        Build it over the piecewise-linear basis.
    seed
        Seeds the random vectors.
    kernel_tol
        Relative tolerance of the kernels' Tucker compression.
    far_order, medium_order, near_order
        Quadrature orders of the kernels on the padded grid.

    Returns
    -------
    FieldBasis
        The incident basis, not yet solved.
    """
    around, padding = shell(
        body.mask, body.resolution, distance=distance, thickness=thickness
    )
    shape = tuple(around.shape)
    inside = torch.nn.functional.pad(body.mask.to(torch.bool), (padding,) * 6)
    orders = {
        "far_order": far_order,
        "medium_order": medium_order,
        "near_order": near_order,
        "linear": linear,
    }
    kernel_n = circulant_tucker(
        vie.kernel_n(shape, body.resolution, medium.wavenumber, **orders).to(
            body.device
        ),
        kernel_tol,
    )
    kernel_k = circulant_tucker(
        vie.kernel_k(shape, body.resolution, medium.wavenumber, **orders).to(
            body.device
        ),
        kernel_tol,
    )
    c = 12 if linear else 3
    source_cells = around.reshape(-1)
    body_cells = inside.reshape(-1)
    n_source, n_body = int(source_cells.sum()), int(body_cells.sum())
    resolution = body.resolution

    def spread(rows, cells):
        grid = torch.zeros(
            (rows.shape[0], c, *shape), dtype=torch.complex128, device=rows.device
        )
        flat = grid.reshape(rows.shape[0], c, -1)
        flat[:, :, cells] = rows.reshape(rows.shape[0], c, -1)
        return grid

    def gather(grid, cells):
        flat = grid.reshape(grid.shape[0], c, -1)
        return flat[:, :, cells].reshape(grid.shape[0], -1)

    def magnetic(rows):
        """Shell currents to the body's magnetic field, ``G^-1 K``."""
        grid = spread(rows, source_cells)
        field = vie.apply_inverse_g(vie.apply_k(kernel_k, grid), resolution)
        return gather(field, body_cells)

    def magnetic_adjoint(rows):
        """Its Hermitian adjoint; the Galerkin K matrix is symmetric."""
        grid = vie.apply_inverse_g(spread(rows, body_cells), resolution)
        field = vie.apply_k(kernel_k, grid.conj()).conj()
        return gather(field, source_cells)

    def electric(rows):
        """Shell currents to the body's electric field, ``G^-1 N / (j omega eps_0)``."""
        grid = spread(rows, source_cells)
        field = vie.apply_n(kernel_n, grid) / medium.electric_scaling
        return gather(vie.apply_inverse_g(field, resolution), body_cells)

    generator = torch.Generator().manual_seed(seed)
    device = body.device
    q = _range(
        lambda x: -magnetic(x),
        c * n_source,
        c * n_body,
        interpolation_tol,
        block,
        generator,
        device,
    )
    w = _range(
        lambda y: -magnetic_adjoint(y),
        c * n_body,
        c * n_source,
        interpolation_tol,
        block,
        generator,
        device,
    )
    if q.shape[1] < w.shape[1]:
        pulled = magnetic_adjoint(q.T).T
        middle = pulled.conj().T @ w
    else:
        middle = q.conj().T @ magnetic(w.T).T
    left, values, right_h = torch.linalg.svd(middle, full_matrices=False)
    field_h = q @ left
    currents = w @ right_h.conj().T

    rows = _deim(field_h)
    voxels = sorted({row % n_body for row in rows})
    samples = torch.tensor(voxels, dtype=torch.long, device=device)

    small = (values <= tol * values[0]).nonzero()
    kept = int(small[0]) + 1 if small.numel() else values.numel()
    field_h = field_h[:, :kept]
    scaled = (currents[:, :kept] / values[:kept].to(currents.dtype)).T
    incident_electric = electric(scaled)
    incident_magnetic = field_h.T.contiguous()

    sampled = _sampled(incident_electric, samples, n_body, c)
    return FieldBasis(
        incident_electric=incident_electric.contiguous(),
        incident_magnetic=incident_magnetic,
        samples=samples,
        interpolation=torch.linalg.pinv(sampled),
        singular_values=values[:kept],
        linear=linear,
    )


def _sampled(rows, samples, n_voxels, n_components):
    """Read every component of the sampled voxels, components major."""
    index = (
        torch.arange(n_components, device=samples.device)[:, None] * n_voxels
        + samples[None, :]
    ).reshape(-1)
    return rows[:, index].transpose(0, 1)


def solve(
    basis: FieldBasis,
    body: VoxelBody,
    medium: Medium,
    *,
    tol: float = 1e-5,
    kernel_tol: float = 1e-7,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
) -> FieldBasis:
    """Solve the body for every incident field, as ``ie_solver_vie_basis.m``.

    Also forms the total fields (``em_ehfield_vie.m``), the response at the
    samples and the basis noise covariance.

    Parameters
    ----------
    basis
        From :func:`surface_basis`.
    body
        The body it was built for.
    medium
        The same frequency.
    tol
        Target for each body solve's relative residual.
    kernel_tol
        Relative tolerance of the kernels' Tucker compression.
    far_order, medium_order, near_order
        Quadrature orders of the body kernels.

    Returns
    -------
    FieldBasis
        The basis with its total fields, currents, response and covariance.
    """
    orders = {
        "far_order": far_order,
        "medium_order": medium_order,
        "near_order": near_order,
    }
    operator = BodyOperator.build(
        body, medium, tol=kernel_tol, linear=basis.linear, **orders
    )
    kernel_k = circulant_tucker(
        vie.kernel_k(
            body.shape,
            body.resolution,
            medium.wavenumber,
            linear=basis.linear,
            **orders,
        ).to(body.device),
        kernel_tol,
    )
    currents, electric, magnetic = [], [], []
    for incident in basis.incident_electric:
        field = body.from_dof(incident)
        current = solve_body(operator, field, tol=tol).x
        currents.append(current)
        electric.append(body.to_dof(operator.total_field(current, field)))
        polarisation = body.from_dof(current)
        scattered = vie.apply_inverse_g(
            vie.apply_k(kernel_k, polarisation), body.resolution
        )
        magnetic.append(body.to_dof(body.mask * scattered))
    current = torch.stack(currents)
    electric = torch.stack(electric)
    magnetic = basis.incident_magnetic + torch.stack(magnetic)

    n_voxels = body.n_voxels
    c = basis.n_components
    mass = vie.mass(c, body.resolution).to(body.device).repeat_interleave(n_voxels)
    projected = (basis.incident_electric * mass.to(torch.complex128)) @ current.T
    x = basis.interpolation
    response = x.T @ projected @ x

    sigma = (body.conductivity[body.mask]).repeat(c).to(torch.complex128)
    covariance = (electric * (mass * sigma)) @ electric.conj().T
    return replace(
        basis,
        electric=electric,
        magnetic=magnetic,
        current=current,
        response=response,
        covariance=covariance,
    )


def ultimate_maps(
    basis: FieldBasis,
    body: VoxelBody,
    medium: Medium,
    modes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the ultimate intrinsic SNR and transmit efficiency, as ``em_ehfield_vie.m``.

    Parameters
    ----------
    basis
        A solved basis.
    body
        The body it was built for.
    medium
        The same frequency.
    modes
        Use the first this many basis fields; all by default. MARIE maps a
        logarithmic run of these to show convergence.

    Returns
    -------
    snr : torch.Tensor
        MARIE's ``UISNR``, shape ``(n1, n2, n3)``.
    efficiency : torch.Tensor
        MARIE's ``UITXE``, same shape.
    """
    from mariepy import metrics

    count = basis.rank if modes is None else modes
    magnetic = basis.magnetic[:count]
    grid = torch.stack([body.from_dof(row) for row in magnetic])
    centres = grid[:, 0::4] if basis.linear else grid
    mu = medium.permeability
    b1_minus = mu * (centres[:, 0] - 1j * centres[:, 1])
    b1_plus = mu * (centres[:, 0] + 1j * centres[:, 1])
    covariance = basis.covariance[:count, :count]
    return (
        metrics.snr(b1_minus, covariance, medium, body.resolution, body.mask),
        metrics.transmit_efficiency(b1_plus, covariance, body.mask),
    )


@dataclass(frozen=True)
class MrgfSolution:
    """A coil solved against a body through the body's field basis.

    Attributes
    ----------
    impedance
        The coil's matrix with the body's response added, MARIE's ``Zss`` in
        ``ie_solver_svie_mrgf.m``, over the shield's unknowns and then the
        coil's where there is a shield.
    coil
        Each port's coil currents, shape ``(n_ports, n_dof)``.
    coefficients
        Each port's basis coefficients, MARIE's ``a``, shape
        ``(n_ports, rank)``.
    admittance
        The port admittance, symmetrised.
    electric, magnetic
        Each port's total fields over the grid, shape
        ``(n_ports, c, n1, n2, n3)``.
    body
        Each port's body current, shape ``(n_ports, c * n_voxels)``, or None
        for a basis read from MARIE's file, which stores none.
    shield
        Each port's shield currents, or None without a shield.
    """

    impedance: torch.Tensor
    coil: torch.Tensor
    coefficients: torch.Tensor
    admittance: torch.Tensor
    electric: torch.Tensor
    magnetic: torch.Tensor
    body: torch.Tensor | None
    shield: torch.Tensor | None = None


def solve_coil(
    coil,
    system,
    basis: FieldBasis,
    body: VoxelBody,
    medium: Medium,
    *,
    shield: SurfaceCoil | None = None,
    triangle_order: int = 4,
    cell_order: int = 2,
) -> MrgfSolution:
    """Solve a coil against a body through its solved basis, as ``ie_solver_svie_mrgf.m``.

    A shield joins as MARIE joins it there: its unknowns come first, with its
    own matrix, its interaction with the coil, and its coupling to the body
    at the samples, and its driven ports, if any, come first among the ports.

    Parameters
    ----------
    coil
        A surface coil, a wire coil, or both together.
    system
        Its own system, from :func:`mariepy.sie.assemble` or its wire
        counterparts.
    basis
        A solved basis of the body.
    body
        The body.
    medium
        The same frequency.
    shield
        An RF shield around the coil and the body.
    triangle_order, cell_order
        Quadrature orders of the coupling at the samples.

    Returns
    -------
    MrgfSolution
        The port currents, admittance and fields.
    """
    from mariepy import network

    centres = body.coordinates().reshape(3, -1).transpose(0, 1)[body.mask.reshape(-1)]
    points = centres[basis.samples]
    tested = coupling_at(
        coil,
        points,
        medium,
        body.resolution,
        linear=basis.linear,
        triangle_order=triangle_order,
        cell_order=cell_order,
    )
    matrix, excitation = system.impedance, system.excitation
    if shield is not None:
        from mariepy import shield as shield_module
        from mariepy.sie import assemble as assemble_surface

        own = assemble_surface(shield, medium)
        cross = shield_module._coil_coupling(shield, coil, medium, 4)
        matrix = torch.cat(
            [
                torch.cat([own.impedance, cross], dim=1),
                torch.cat([cross.T, matrix], dim=1),
            ]
        )
        excitation = torch.cat(
            [
                torch.nn.functional.pad(own.excitation, (0, coil.n_dof)),
                torch.nn.functional.pad(excitation, (shield.n_dof, 0)),
            ]
        )
        tested = torch.cat(
            [
                coupling_at(
                    shield,
                    points,
                    medium,
                    body.resolution,
                    linear=basis.linear,
                    triangle_order=triangle_order,
                    cell_order=cell_order,
                ),
                tested,
            ],
            dim=1,
        )
    inverse = _inverse_mass(
        basis.n_components, points.shape[0], body.resolution, points.device
    )
    field = inverse[:, None] * tested  # the incident field at the samples
    if basis.tested:
        # MARIE's ie_solver_svie_mrgf.m: ZbcN.' * iG * (U S V' * ZbcN).
        impedance = matrix + tested.T @ (inverse[:, None] * (basis.response @ tested))
    else:
        impedance = matrix + field.T @ basis.response @ field
    current = torch.linalg.solve(impedance, excitation.T).T
    coefficients = (basis.interpolation @ (field @ current.T)).T
    admittance = network.symmetrise(network.port_admittance(excitation, current))
    shield_current = None
    if shield is not None:
        shield_current = current[:, : shield.n_dof]
        current = current[:, shield.n_dof :]
    electric = coefficients @ basis.electric
    magnetic = coefficients @ basis.magnetic
    return MrgfSolution(
        impedance=impedance,
        coil=current,
        shield=shield_current,
        coefficients=coefficients,
        admittance=admittance,
        electric=torch.stack([body.from_dof(row) for row in electric]),
        magnetic=torch.stack([body.from_dof(row) for row in magnetic]),
        body=None if basis.current is None else coefficients @ basis.current,
    )


def _read_matlab(group, name):
    """Read one MATLAB v7.3 array: dimensions reversed, complex as a compound.

    A MATLAB ``p``-by-``q`` matrix comes back as its transpose; a sparse one
    is a group of ``data``, ``ir`` and ``jc``.
    """
    import numpy as np

    item = group[name]
    if hasattr(item, "keys"):
        from scipy.sparse import csc_matrix

        shape = (int(item.attrs["MATLAB_sparse"]), len(item["jc"]) - 1)
        data = item["data"][()]
        if data.dtype.names:
            data = data["real"] + 1j * data["imag"]
        matrix = csc_matrix((data, item["ir"][()], item["jc"][()]), shape=shape)
        return torch.from_numpy(np.asarray(matrix.todense()))
    data = item[()]
    if data.dtype.names:
        data = data["real"] + 1j * data["imag"]
    return torch.from_numpy(np.ascontiguousarray(np.asarray(data).T))


def read_marie(path, body: VoxelBody, *, device=None) -> FieldBasis:
    """Read a basis MARIE's ``BASIS_runner.m`` saved, for the body it was built on.

    Reads the datasets ``MRGF_runner.m`` reads from the MATLAB v7.3 file:
    ``Ue`` and ``Ub``, the total fields; ``X``, the interpolation; the SVD of
    the response, ``U_hat_inv``, ``S_hat_inv`` and ``V_hat_inv``; and the
    sampled voxels' coordinates ``xds``, ``yds`` and ``zds``, with ``phi`` when
    present. MARIE orders a body's unknowns component by component, voxels in
    column-major order; they are reordered to the body's own. The basis stays
    in MARIE's tested form and :func:`solve_coil` treats it as MARIE does.

    Parameters
    ----------
    path
        The ``.mat`` file.
    body
        The body the basis was built for.
    device
        Device to place the tensors on; the body's by default.

    Returns
    -------
    FieldBasis
        With :attr:`FieldBasis.tested` set.

    Raises
    ------
    ImportError
        Without h5py.
    ValueError
        If the file's fields do not fit the body, or a sampled voxel is not
        one of its tissue voxels.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "reading MARIE's basis files needs h5py: pip install 'mariepy[marie]'"
        ) from error
    import numpy as np

    device = body.device if device is None else device
    with h5py.File(path, "r") as handle:
        group = handle["BASIS"]
        # MATLAB keeps each field as a column; here each is a row.
        electric = _read_matlab(group, "Ue").T
        magnetic = _read_matlab(group, "Ub").T
        interpolation = _read_matlab(group, "X")
        u = _read_matlab(group, "U_hat_inv")
        s = _read_matlab(group, "S_hat_inv")
        v = _read_matlab(group, "V_hat_inv")
        coordinates = [
            _read_matlab(group, name).reshape(-1).real for name in ("xds", "yds", "zds")
        ]
        covariance = _read_matlab(group, "phi") if "phi" in group else None

    n_voxels = body.n_voxels
    components, remainder = divmod(electric.shape[1], n_voxels)
    if remainder or components not in (3, 12):
        raise ValueError(
            f"the basis has {electric.shape[1]} unknowns per field, not 3 or 12 "
            f"for each of the body's {n_voxels} voxels"
        )

    # MARIE's voxel order is column-major over the grid.
    index = body.mask.cpu().nonzero()
    n1, n2, _ = body.shape
    fortran = index[:, 0] + n1 * (index[:, 1] + n2 * index[:, 2])
    marie_position = torch.empty(n_voxels, dtype=torch.long)
    marie_position[torch.argsort(fortran)] = torch.arange(n_voxels)
    order = (
        torch.arange(components)[:, None] * n_voxels + marie_position[None, :]
    ).reshape(-1)

    origin = torch.tensor(body.origin, dtype=torch.float64)
    points = torch.stack(coordinates, dim=1)
    cells = torch.round((points - origin) / body.resolution).long()
    numbering = torch.full((int(np.prod(body.shape)),), -1, dtype=torch.long)
    numbering[body.mask.cpu().reshape(-1)] = torch.arange(n_voxels)
    flat = (cells[:, 0] * n2 + cells[:, 1]) * body.shape[2] + cells[:, 2]
    samples = numbering[flat]
    if bool((samples < 0).any()):
        raise ValueError(
            "a sampled voxel of the basis is not a tissue voxel of the body"
        )

    response = u @ s.to(u.dtype) @ v.conj().T
    return FieldBasis(
        incident_electric=None,
        incident_magnetic=None,
        samples=samples.to(device),
        interpolation=interpolation.to(torch.complex128).to(device),
        singular_values=None,
        linear=components == 12,
        electric=electric[:, order].to(torch.complex128).to(device),
        magnetic=magnetic[:, order].to(torch.complex128).to(device),
        response=response.to(torch.complex128).to(device),
        covariance=None if covariance is None else covariance.to(device),
        tested=True,
    )
