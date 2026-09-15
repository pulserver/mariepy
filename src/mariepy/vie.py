"""The volume integral operator on the body grid.

Ported from MARIE 3.0's ``src_integral_equations/src_vie/src_operators_vie/``:
``cubatures/{G_N,G_K,VV_Nop,VV_Kop,weights_points}.m`` for the kernel, and
``src_solver/src_mvp/mvp_vie/`` for the products.

Two operators act on the polarisation current in the voxels. N is the
double-curl dyadic that gives the electric field, symmetric with six distinct
components in the order xx, xy, xz, yy, yz, zz. K is the curl that gives the
magnetic field, with three components in the order x, y, z. Both depend only on
the offset between two voxels, so each is stored once per offset and applied by
FFT through :mod:`mariepy.tucker`.

Voxels that touch need the surface-surface treatment of MARIE's
``surface_surface_kernels_*.m`` and its singular integrals; the quadrature here
is the volume-volume rule MARIE uses at every other offset, and it diverges as
the offset goes to zero.
"""

from __future__ import annotations

import torch

from mariepy.quadrature import gauss_legendre_1d
from mariepy.tucker import CirculantSymbol

__all__ = [
    "apply_g",
    "apply_inverse_g",
    "apply_k",
    "apply_n",
    "green_k",
    "green_n",
    "volume_volume_k",
    "volume_volume_n",
]

# Which of the six stored components of N carries each (row, column) of the
# dyad, given the order xx, xy, xz, yy, yz, zz.
_DYADIC_INDEX = ((0, 1, 2), (1, 3, 4), (2, 4, 5))

# Which component of K carries each (row, column) of the curl, and with which
# sign. The diagonal is empty, as a curl has no diagonal.
_CURL_INDEX = ((None, 2, 1), (2, None, 0), (1, 0, None))
_CURL_SIGN = ((0.0, -1.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 1.0, 0.0))

# How many kernel evaluations to hold at once. The six-dimensional rule has
# order**6 points, so the offsets are taken in chunks that keep this bounded.
_CHUNK_ELEMENTS = 1 << 22


def green_n(separation: torch.Tensor, wavenumber: float) -> torch.Tensor:
    """Return the six distinct components of the double-curl dyadic Green kernel.

    Parameters
    ----------
    separation
        Shape ``(..., 3)``, the vector from source point to observation point.
    wavenumber
        Free-space wavenumber in rad/m.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 6)``, complex, in the order xx, xy, xz, yy, yz, zz.
    """
    x, y, z = separation[..., 0], separation[..., 1], separation[..., 2]
    r = torch.linalg.vector_norm(separation, dim=-1)
    k = wavenumber

    envelope = torch.exp(-1j * k * r) / (4.0 * torch.pi)
    r2, r3, r4, r5 = r**2, r**3, r**4, r**5

    def diagonal(component: torch.Tensor) -> torch.Tensor:
        return envelope * (
            3.0 * component**2 / r5
            + 3j * k * component**2 / r4
            - (1.0 + k**2 * component**2) / r3
            - 1j * k / r2
        )

    def off_diagonal(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        product = first * second
        return envelope * (
            3.0 * product / r5 + 3j * k * product / r4 - k**2 * product / r3
        )

    g_xx, g_yy, g_zz = diagonal(x), diagonal(y), diagonal(z)
    return torch.stack(
        (
            -g_yy - g_zz,
            off_diagonal(x, y),
            off_diagonal(x, z),
            -g_xx - g_zz,
            off_diagonal(y, z),
            -g_xx - g_yy,
        ),
        dim=-1,
    )


def green_k(separation: torch.Tensor, wavenumber: float) -> torch.Tensor:
    """Return the three components of the curl Green kernel.

    Parameters
    ----------
    separation
        Shape ``(..., 3)``, the vector from source point to observation point.
    wavenumber
        Free-space wavenumber in rad/m.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)``, complex, in the order x, y, z.
    """
    r = torch.linalg.vector_norm(separation, dim=-1)
    envelope = torch.exp(-1j * r * wavenumber) / (4.0 * torch.pi)
    radial = (1.0 / r**3 + 1j * wavenumber / r**2).unsqueeze(-1)
    return envelope.unsqueeze(-1) * (-separation) * radial


def volume_volume_n(
    offsets: torch.Tensor, resolution: float, wavenumber: float, order: int = 4
) -> torch.Tensor:
    """Integrate the N kernel over two voxels separated by each offset.

    Parameters
    ----------
    offsets
        Shape ``(n_offsets, 3)``, the vector from the source voxel's centre to
        the observation voxel's centre, in metres.
    resolution
        Voxel pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Gauss-Legendre points per axis; the rule has ``order ** 6`` points.

    Returns
    -------
    torch.Tensor
        Shape ``(n_offsets, 6)``, complex.

    Raises
    ------
    ValueError
        An offset is zero, where the kernel is not integrable by this rule.
    """
    return _volume_volume(offsets, resolution, wavenumber, order, green_n, 6)


def volume_volume_k(
    offsets: torch.Tensor, resolution: float, wavenumber: float, order: int = 4
) -> torch.Tensor:
    """Integrate the K kernel over two voxels separated by each offset.

    Parameters
    ----------
    offsets
        Shape ``(n_offsets, 3)``, in metres.
    resolution
        Voxel pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Gauss-Legendre points per axis.

    Returns
    -------
    torch.Tensor
        Shape ``(n_offsets, 3)``, complex.

    Raises
    ------
    ValueError
        An offset is zero.
    """
    return _volume_volume(offsets, resolution, wavenumber, order, green_k, 3)


def _volume_volume(offsets, resolution, wavenumber, order, kernel, n_components):
    """Run the six-dimensional rule for one kernel, in chunks over the offsets."""
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(f"offsets must have shape (n, 3), got {tuple(offsets.shape)}")
    if torch.any(torch.linalg.vector_norm(offsets, dim=-1) == 0):
        raise ValueError(
            "the volume-volume rule diverges at a zero offset; voxels that touch "
            "need the surface-surface treatment"
        )

    weights, nodes = gauss_legendre_1d(
        order, device=offsets.device, dtype=offsets.dtype
    )
    # Six nested axes: three over the observation voxel, three over the source.
    grids = torch.meshgrid(*([nodes] * 6), indexing="ij")
    weight = torch.ones_like(grids[0])
    for axis in range(6):
        shape = [1] * 6
        shape[axis] = order
        weight = weight * weights.reshape(shape)
    weight = weight.reshape(-1)

    # The kernel sees only the separation, so the two triples enter as their
    # difference: the observation node less the source node, scaled to the voxel.
    separation_nodes = (
        resolution
        / 2.0
        * torch.stack(
            (
                grids[0] - grids[3],
                grids[1] - grids[4],
                grids[2] - grids[5],
            ),
            dim=-1,
        ).reshape(-1, 3)
    )

    jacobian = (resolution / 2.0) ** 6
    n_points = weight.shape[0]
    chunk = max(1, _CHUNK_ELEMENTS // n_points)

    result = torch.zeros(
        (offsets.shape[0], n_components),
        device=offsets.device,
        dtype=torch.complex128,
    )
    for start in range(0, offsets.shape[0], chunk):
        block = offsets[start : start + chunk]
        separation = block.unsqueeze(1) + separation_nodes.unsqueeze(0)
        values = kernel(separation, wavenumber)
        result[start : start + chunk] = jacobian * torch.einsum(
            "q,bqc->bc", weight.to(values.dtype), values
        )
    return result


def apply_n(
    symbols: tuple[CirculantSymbol, ...], current: torch.Tensor
) -> torch.Tensor:
    """Apply the N operator to a current on the grid.

    Parameters
    ----------
    symbols
        Six compressed symbols from :func:`mariepy.tucker.circulant_tucker`, in
        the order xx, xy, xz, yy, yz, zz.
    current
        Shape ``(..., 3, n1, n2, n3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, n1, n2, n3)``.

    Raises
    ------
    ValueError
        The wrong number of symbols was given.
    """
    if len(symbols) != 6:
        raise ValueError(f"the N operator takes six symbols, got {len(symbols)}")
    return _apply(symbols, current, _DYADIC_INDEX, None)


def apply_k(
    symbols: tuple[CirculantSymbol, ...], current: torch.Tensor
) -> torch.Tensor:
    """Apply the K operator to a current on the grid.

    Parameters
    ----------
    symbols
        Three compressed symbols, in the order x, y, z.
    current
        Shape ``(..., 3, n1, n2, n3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, n1, n2, n3)``.

    Raises
    ------
    ValueError
        The wrong number of symbols was given.
    """
    if len(symbols) != 3:
        raise ValueError(f"the K operator takes three symbols, got {len(symbols)}")
    return _apply(symbols, current, _CURL_INDEX, _CURL_SIGN)


def _apply(symbols, current, index, sign):
    """Convolve each component of the current with the symbols that reach it."""
    if current.ndim < 4 or current.shape[-4] != 3:
        raise ValueError(
            f"a current must end in (3, n1, n2, n3), got {tuple(current.shape)}"
        )
    grid = current.shape[-3:]
    padded = symbols[0].shape

    transformed = torch.fft.fftn(current, s=padded, dim=(-3, -2, -1))
    expanded = [symbol.expand() for symbol in symbols]

    out = torch.zeros_like(transformed)
    for row in range(3):
        for column in range(3):
            which = index[row][column]
            if which is None:
                continue
            scale = 1.0 if sign is None else sign[row][column]
            out[..., row, :, :, :] = out[..., row, :, :, :] + scale * (
                expanded[which] * transformed[..., column, :, :, :]
            )

    field = torch.fft.ifftn(out, dim=(-3, -2, -1))
    return field[..., : grid[0], : grid[1], : grid[2]]


def apply_g(current: torch.Tensor, resolution: float) -> torch.Tensor:
    """Apply the Galerkin mass matrix of the piecewise-constant basis.

    Parameters
    ----------
    current
        Any shape.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        ``current`` scaled by the voxel volume.
    """
    return resolution**3 * current


def apply_inverse_g(current: torch.Tensor, resolution: float) -> torch.Tensor:
    """Invert :func:`apply_g`.

    Parameters
    ----------
    current
        Any shape.
    resolution
        Voxel pitch in metres.

    Returns
    -------
    torch.Tensor
        ``current`` divided by the voxel volume.
    """
    return current / resolution**3


# The eight corners of a cell of side `resolution` centred on the origin, in
# MARIE's order, and the four corners of each face as a cycle around it. Faces
# run -x, +x, -y, +y, -z, +z.
_CORNERS = (
    (-1, -1, -1),
    (+1, -1, -1),
    (+1, +1, -1),
    (-1, +1, -1),
    (-1, -1, +1),
    (+1, -1, +1),
    (+1, +1, +1),
    (-1, +1, +1),
)
_FACE_CYCLES = (
    (0, 3, 7, 4),
    (1, 2, 6, 5),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 1, 2, 3),
    (4, 5, 6, 7),
)
_FACE_NORMALS = (
    (-1.0, 0.0, 0.0),
    (+1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, +1.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, +1.0),
)

# The six distinct components of the symmetric dyad, as the pair of Cartesian
# directions each one tests.
_COMPONENT_AXES = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def face_cycle(face: int, centre, resolution: float):
    """Return the four corners of one face of a cell, as a cycle around it.

    Parameters
    ----------
    face
        Which face, in the order -x, +x, -y, +y, -z, +z.
    centre
        The cell's centre, a sequence of three floats.
    resolution
        Cell pitch in metres.

    Returns
    -------
    list of tuple
        Four corners, each a triple of floats.
    """
    half = resolution / 2.0
    return [
        tuple(centre[axis] + half * _CORNERS[corner][axis] for axis in range(3))
        for corner in _FACE_CYCLES[face]
    ]


def face_adjacency(
    offset_cells, face_observer: int, face_source: int, resolution: float
):
    """Return how two faces of two cells touch, and the points DIRECTFN wants.

    The singular routines depend on which faces touch, on the adjacency, and on
    the shared feature sitting in the slot each routine reads it from. They do
    not depend on the winding of either square, nor on the order within a shared
    edge, so the ordering is derived here rather than tabulated.

    Parameters
    ----------
    offset_cells
        Whole-cell offset from the source cell to the observation cell.
    face_observer, face_source
        Faces of the observation and source cells.
    resolution
        Cell pitch in metres.

    Returns
    -------
    kind : str or None
        ``"self"``, ``"edge"``, ``"vertex"``, or ``None`` when the faces do not
        touch.
    points : list of tuple or None
        The four, six or seven ordered points, or ``None``.
    """
    source = face_cycle(face_source, (0.0, 0.0, 0.0), resolution)
    observer = face_cycle(
        face_observer,
        tuple(resolution * value for value in offset_cells),
        resolution,
    )

    tolerance = 1e-9 * resolution

    def same(first, second):
        return all(abs(a - b) <= tolerance for a, b in zip(first, second, strict=True))

    shared = [
        index
        for index, point in enumerate(source)
        if any(same(point, other) for other in observer)
    ]

    if len(shared) == 4:
        return "self", list(source)
    if len(shared) == 2:
        rotated = _rotate_to(source, shared, (2, 3))
        following = _cycle_from(observer, rotated[3], rotated[2])
        return "edge", [*rotated, following[3], following[2]]
    if len(shared) == 1:
        rotated = _rotate_to(source, shared, (2,))
        following = _cycle_from(observer, rotated[2], None)
        return "vertex", [*rotated, *following[1:]]
    return None, None


def _rotate_to(cycle, shared, slots):
    """Rotate a four-cycle, either way round, until the shared points land."""
    for direction in (cycle, cycle[::-1]):
        for turn in range(4):
            candidate = direction[turn:] + direction[:turn]
            landed = {candidate.index(cycle[index]) for index in shared}
            if landed == set(slots):
                return candidate
    raise ValueError("the shared points do not sit on one face of the cell")


def _cycle_from(cycle, first, second):
    """Rotate a four-cycle to start at one point, optionally reaching a second."""
    start = next(
        index
        for index, point in enumerate(cycle)
        if all(abs(a - b) <= 1e-12 for a, b in zip(point, first, strict=True))
    )
    forward = cycle[start:] + cycle[:start]
    if second is None:
        return forward
    if all(abs(a - b) <= 1e-12 for a, b in zip(forward[1], second, strict=True)):
        return forward
    return [forward[0], *forward[1:][::-1]]


def surface_surface_coefficient(
    face_observer: int, face_source: int, component: int
) -> float:
    """Return the constant the surface-surface integral of one face pair carries.

    Ported from MARIE's ``coefficients_Nop.m``. For the piecewise-constant basis
    only its first kernel survives: the other three each carry the gradient of
    the scalar basis term, which is zero when the basis is constant.

    Parameters
    ----------
    face_observer, face_source
        Faces of the observation and source cells.
    component
        Which of the six components of the symmetric dyad, in the order xx, xy,
        xz, yy, yz, zz.

    Returns
    -------
    float
        ``(n x e_p) . (n' x e_q)`` for that face pair and component.
    """
    import numpy as _np

    p, q = _COMPONENT_AXES[component]
    normal = _np.array(_FACE_NORMALS[face_observer])
    normal_source = _np.array(_FACE_NORMALS[face_source])
    return float(
        _np.dot(
            _np.cross(normal, _np.eye(3)[p]),
            _np.cross(normal_source, _np.eye(3)[q]),
        )
    )


def surface_surface_n(
    offset_cells,
    resolution: float,
    wavenumber: float,
    order: int = 15,
) -> torch.Tensor:
    """Integrate the N kernel over two cells by their faces.

    Voxels that touch make the volume-volume rule diverge. MARIE reduces the
    double-curl dyadic to integrals over the twelve faces of the two cells,
    where the remaining singularity is weak enough for DIRECTFN to handle
    exactly; face pairs that do not touch take a plain four-dimensional rule.

    Parameters
    ----------
    offset_cells
        Whole-cell offset from the source cell to the observation cell.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Points per axis, for the four-dimensional rule and for DIRECTFN alike.

    Returns
    -------
    torch.Tensor
        Shape ``(6,)``, complex, in the order xx, xy, xz, yy, yz, zz.
    """
    import numpy as _np

    from mariepy import _accelerators

    directfn = _accelerators.require(module="mariepy._directfn")
    weights, nodes = gauss_legendre_1d(order)
    weights, nodes = weights.numpy(), nodes.numpy()

    centre_observer = _np.array([resolution * value for value in offset_cells])
    centre_source = _np.zeros(3)
    total = _np.zeros(6, dtype=complex)

    for face_observer in range(6):
        for face_source in range(6):
            coefficients = _np.array(
                [
                    surface_surface_coefficient(face_observer, face_source, component)
                    for component in range(6)
                ]
            )
            if not _np.any(coefficients):
                continue

            kind, points = face_adjacency(
                offset_cells, face_observer, face_source, resolution
            )
            normal_source = _np.array(_FACE_NORMALS[face_source])
            normal_observer = _np.array(_FACE_NORMALS[face_observer])

            if kind is None:
                value = _non_singular_face_pair(
                    centre_observer + normal_observer * resolution / 2.0,
                    centre_source + normal_source * resolution / 2.0,
                    face_observer // 2,
                    face_source // 2,
                    resolution,
                    wavenumber,
                    weights,
                    nodes,
                )
            else:
                vertices = _np.array(points, dtype=float)
                routine, observer_centre = _singular_call(kind, vertices, directfn)
                value = routine(
                    vertices,
                    (vertices[0] + vertices[2]) / 2.0,
                    observer_centre,
                    normal_source,
                    normal_observer,
                    wavenumber,
                    resolution,
                    order,
                    _DIRECTFN_KERNEL_N,
                    0,
                    0,
                )
            total += coefficients * value

    return torch.tensor(total, dtype=torch.complex128)


def _non_singular_face_pair(
    centre_observer,
    centre_source,
    axis_observer,
    axis_source,
    resolution,
    wavenumber,
    weights,
    nodes,
):
    """Integrate the scalar Green function over a pair of faces that do not touch."""
    import numpy as _np

    separation, weight = _face_pair_separation(
        centre_observer,
        centre_source,
        axis_observer,
        axis_source,
        resolution,
        nodes,
        weights,
    )
    distance = _np.linalg.norm(separation, axis=1)
    green = _np.exp(-1j * wavenumber * distance) / (4.0 * _np.pi * distance)
    return (resolution / 2.0) ** 4 * _np.sum(weight * green)


def _singular_call(kind, vertices, directfn):
    """Pick the DIRECTFN routine and the observation centre for one adjacency.

    The centres are the ones MARIE forms in ``singular_ST_lin.m``,
    ``singular_EA_lin.m`` and ``singular_VA_lin.m``: each is the midpoint of a
    diagonal of its own square, which the ordering puts in a different slot for
    each adjacency.
    """
    if kind == "self":
        return directfn.voxel_self, (vertices[0] + vertices[2]) / 2.0
    if kind == "edge":
        return directfn.voxel_edge, (vertices[3] + vertices[5]) / 2.0
    return directfn.voxel_vertex, (vertices[2] + vertices[5]) / 2.0


def kernel_n(
    shape: tuple[int, int, int],
    resolution: float,
    wavenumber: float,
    *,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
) -> torch.Tensor:
    """Assemble the N kernel at every offset of a grid.

    Three regimes, as in MARIE's ``assembly_N.m``: a coarse volume-volume rule
    far out, a finer one over the first ``medium_order`` offsets along each
    axis, and the surface-surface reduction over the two-by-two-by-two block
    where the cells touch and the volume rule diverges.

    Parameters
    ----------
    shape
        Grid shape; the kernel is stored at every non-negative offset.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    far_order, medium_order
        Points per axis for the two volume-volume passes.
    near_order
        Points per axis for the surface-surface pass.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3, 6)``, complex, in the order xx, xy, xz, yy, yz, zz.
    """
    return _assemble(
        shape,
        resolution,
        wavenumber,
        far_order,
        medium_order,
        near_order,
        volume_volume_n,
        surface_surface_n,
        6,
    )


def _assemble(
    shape,
    resolution,
    wavenumber,
    far_order,
    medium_order,
    near_order,
    by_volume,
    by_surface,
    n_components,
):
    """Fill every offset of a grid from the regime that is valid there."""
    import itertools as _itertools

    kernel = torch.zeros((*shape, n_components), dtype=torch.complex128)
    near = tuple(_itertools.product(*(range(min(2, n)) for n in shape)))
    offsets = [
        cell
        for cell in _itertools.product(*(range(n) for n in shape))
        if cell not in near
    ]

    if offsets:
        index = torch.tensor(offsets, dtype=torch.float64)
        far = by_volume(resolution * index, resolution, wavenumber, far_order)
        for row, cell in enumerate(offsets):
            kernel[cell] = far[row]

        medium = [
            cell for cell in offsets if all(value < medium_order for value in cell)
        ]
        if medium:
            index = torch.tensor(medium, dtype=torch.float64)
            refined = by_volume(
                resolution * index, resolution, wavenumber, medium_order
            )
            for row, cell in enumerate(medium):
                kernel[cell] = refined[row]

    for cell in near:
        kernel[cell] = by_surface(cell, resolution, wavenumber, near_order)

    return kernel


# The three components of the curl, as the pair of Cartesian directions each one
# tests. Anti-symmetry leaves three distinct interactions rather than six.
_CURL_COMPONENT_AXES = ((2, 1), (0, 2), (1, 0))

# Which reduced kernel of DIRECTFN each operator's surviving surface-surface
# term asks for. `Kernels.cpp` branches on 0 through 8.
_DIRECTFN_KERNEL_N = 1
_DIRECTFN_KERNEL_K = 5


def curl_surface_coefficient(face_source: int, component: int) -> float:
    """Return the constant the surface-surface integral of one face pair carries.

    Ported from MARIE's ``coefficients_Kop.m``. As for the N operator, only the
    first of its four surface-surface kernels survives the piecewise-constant
    basis, and this one depends on the source face alone.

    Parameters
    ----------
    face_source
        Face of the source cell.
    component
        Which component of the curl, in the order x, y, z.

    Returns
    -------
    float
        ``(e_p x e_q) . n'`` for that face and component.
    """
    import numpy as _np

    p, q = _CURL_COMPONENT_AXES[component]
    return float(
        _np.dot(
            _np.cross(_np.eye(3)[p], _np.eye(3)[q]),
            _np.array(_FACE_NORMALS[face_source]),
        )
    )


def surface_surface_k(
    offset_cells,
    resolution: float,
    wavenumber: float,
    order: int = 15,
) -> torch.Tensor:
    """Integrate the K kernel over two cells by their faces.

    The counterpart of :func:`surface_surface_n` for the curl operator. Its
    integrand contracts the separation with the *observation* face normal where
    the dyadic operator's contracts with the source's.

    Parameters
    ----------
    offset_cells
        Whole-cell offset from the source cell to the observation cell.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    order
        Points per axis, for the four-dimensional rule and for DIRECTFN alike.

    Returns
    -------
    torch.Tensor
        Shape ``(3,)``, complex, in the order x, y, z.
    """
    import numpy as _np

    from mariepy import _accelerators

    directfn = _accelerators.require(module="mariepy._directfn")
    weights, nodes = gauss_legendre_1d(order)
    weights, nodes = weights.numpy(), nodes.numpy()

    centre_observer = _np.array([resolution * value for value in offset_cells])
    total = _np.zeros(3, dtype=complex)

    for face_observer in range(6):
        for face_source in range(6):
            coefficients = _np.array(
                [
                    curl_surface_coefficient(face_source, component)
                    for component in range(3)
                ]
            )
            if not _np.any(coefficients):
                continue

            kind, points = face_adjacency(
                offset_cells, face_observer, face_source, resolution
            )
            normal_source = _np.array(_FACE_NORMALS[face_source])
            normal_observer = _np.array(_FACE_NORMALS[face_observer])

            if kind is None:
                value = _non_singular_curl_face_pair(
                    centre_observer + normal_observer * resolution / 2.0,
                    normal_source * resolution / 2.0,
                    face_observer // 2,
                    face_source // 2,
                    normal_observer,
                    resolution,
                    wavenumber,
                    weights,
                    nodes,
                )
            else:
                vertices = _np.array(points, dtype=float)
                routine, observer_centre = _singular_call(kind, vertices, directfn)
                value = routine(
                    vertices,
                    (vertices[0] + vertices[2]) / 2.0,
                    observer_centre,
                    normal_source,
                    normal_observer,
                    wavenumber,
                    resolution,
                    order,
                    _DIRECTFN_KERNEL_K,
                    0,
                    0,
                )
            total += coefficients * value

    return torch.tensor(total, dtype=torch.complex128)


def _non_singular_curl_face_pair(
    centre_observer,
    centre_source,
    axis_observer,
    axis_source,
    normal_observer,
    resolution,
    wavenumber,
    weights,
    nodes,
):
    """Integrate the curl kernel over a pair of faces that do not touch."""
    import numpy as _np

    separation, weight = _face_pair_separation(
        centre_observer,
        centre_source,
        axis_observer,
        axis_source,
        resolution,
        nodes,
        weights,
    )
    distance = _np.linalg.norm(separation, axis=1)
    green = _np.exp(-1j * wavenumber * distance) / (4.0 * _np.pi * distance)
    static = 1.0 / (4.0 * _np.pi * distance)

    radial = (
        -1j * wavenumber * green / distance - green / distance**2 + static / distance**2
    ) / (1j * wavenumber) ** 2
    contracted = separation @ normal_observer * radial
    return (resolution / 2.0) ** 4 * _np.sum(weight * contracted)


def _face_pair_separation(
    centre_observer,
    centre_source,
    axis_observer,
    axis_source,
    resolution,
    nodes,
    weights,
):
    """Return the separation at every point of the four-dimensional rule."""
    import numpy as _np

    def place(centre, axis, first, second):
        free = [value for value in range(3) if value != axis]
        point = _np.empty((first.size, 3))
        point[:, axis] = centre[axis]
        point[:, free[0]] = centre[free[0]] + resolution / 2.0 * first
        point[:, free[1]] = centre[free[1]] + resolution / 2.0 * second
        return point

    a, b, c, d = (
        grid.reshape(-1) for grid in _np.meshgrid(*([nodes] * 4), indexing="ij")
    )
    weight = _np.ones(a.size)
    for grid in _np.meshgrid(*([weights] * 4), indexing="ij"):
        weight = weight * grid.reshape(-1)

    separation = place(centre_observer, axis_observer, a, b) - place(
        centre_source, axis_source, c, d
    )
    return separation, weight


def kernel_k(
    shape: tuple[int, int, int],
    resolution: float,
    wavenumber: float,
    *,
    far_order: int = 4,
    medium_order: int = 8,
    near_order: int = 15,
) -> torch.Tensor:
    """Assemble the K kernel at every offset of a grid.

    The three regimes of :func:`kernel_n`, for the curl operator.

    Parameters
    ----------
    shape
        Grid shape; the kernel is stored at every non-negative offset.
    resolution
        Cell pitch in metres.
    wavenumber
        Free-space wavenumber in rad/m.
    far_order, medium_order
        Points per axis for the two volume-volume passes.
    near_order
        Points per axis for the surface-surface pass.

    Returns
    -------
    torch.Tensor
        Shape ``(n1, n2, n3, 3)``, complex, in the order x, y, z.
    """
    return _assemble(
        shape,
        resolution,
        wavenumber,
        far_order,
        medium_order,
        near_order,
        volume_volume_k,
        surface_surface_k,
        3,
    )
