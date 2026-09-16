# Milestone 1 — surface coil, voxel body, pFFT coupling

The design for the first milestone of the MARIE 3.0 port: the path MARIE takes
by default for a surface coil around a piecewise-constant voxel body, coupled by
precorrected FFT, ending with each port's body currents, network parameters and
E and H fields.

`PLAN.md` states the scope, the validation criteria and the constraints. This
note fixes the module layout, the order in which the stages are built and
checked, the MARIE-to-mariepy mapping, the signatures of the coupling kernels,
and the readings taken where MARIE's reference leaves a choice.

MARIE's reference is <https://github.com/cloudmrhub/marie-tools> (MIT). File
paths below are relative to its `src/`.

## 1. What the milestone computes

For a coil of `n_ports` ports, discretised into `n_rwg` RWG edge basis functions
on a triangular surface mesh, around a body of `n_voxels` voxels on a uniform
grid of pitch `res`, at the Larmor frequency of the nucleus and field strength
given:

- the coil surface current and the body polarisation current for a unit drive at
  each port, from one restarted GMRES solve per port;
- the port admittance, impedance and scattering matrices;
- the electric and magnetic field in each body voxel, per port.

The body uses the piecewise-constant basis, so each voxel carries three current
degrees of freedom and `n_dof = 3 · n_voxels`. There is no wire coil, no RF
shield and no co-simulation; those are milestones 2 and 3.

The coupled system MARIE solves, in the block form of `mvp_svie_pfft.m`, is

```
[  Z_cc      Z_bc^T  ] [ J_c ]   [ V ]
[ -Z_bc      Z_bb    ] [ J_b ] = [ 0 ]
```

where `Z_cc` is the coil EFIE matrix, `Z_bb` the volume integral operator on the
body, `Z_bc` the coil-to-body coupling, and `V` the delta-gap port excitation.
Precorrected FFT never forms `Z_bc` or the far part of `Z_cc` densely: coil
currents are projected onto a grid of expansion voxels, the whole grid is
convolved with the body Green kernel by FFT, and the near interactions that the
projection gets wrong are corrected by a sparse matrix — `direct minus
projected`, assembled in `pfft_surface_assemble_direct_bc.m` and
`pfft_assemble_voxel_bc.m`.

## 2. Module layout

```
src/mariepy/
    __init__.py
    _accelerators.py        loads mariepy._ext and mariepy._directfn
    constants.py            nucleus table, Larmor frequency, medium constants
    quadrature.py           Gauss-Legendre, triangle rules, Dunavant, Lebedev
    tucker.py               HOSVD, n-mode product, circulant FFT embedding
    mesh.py                 GMSH 2.2 reader, triangle geometry, meshes in code
    coil.py                 RWG basis, ports, lumped elements
    body.py                 voxel grid, tissue contrast, degree-of-freedom map
    vie.py                  body kernels N and K, and their products
    coupling.py             coil-to-body kernels and the collocation matrix
    sie.py                  coil EFIE matrix, lumped loads, port excitation
    pfft.py                 extended grid, projection, precorrection
    system.py               the coupled operator
    preconditioner.py       coil LU block, body diagonal block
    gmres.py                restarted GMRES with a split preconditioner
    network.py              Y, Z and S at the ports
    fields.py               E and H on the body grid
    solver.py               drives geometry, operators, solve, network, fields

src/cpp/                    -> mariepy._ext, MIT
    module.cpp              bindings; a kernel moves here when a profile asks

src/cpp_lgpl/               -> mariepy._directfn, LGPL, notices kept
    NOTICE.md               what is carried, and the two build changes
    module_directfn.cpp     bindings
    directfn_vie/           the voxel family, linking as its sources stand
    directfn_rwg/           direct_ws_{st,ea,va}_rwg and their headers
    rwg_namespace_*.cpp     one wrapper per RWG source, giving it a namespace

tests/
    test_quadrature.py  test_tucker.py  test_mesh.py   test_coil.py
    test_body.py        test_vie.py     test_mie.py    test_sie.py
    test_pfft.py        test_coupling.py               test_gmres.py
    test_network.py     test_fields.py  test_power_balance.py
    mie.py              analytic reference for a dielectric sphere
    pec.py              analytic reference for a conducting sphere
    parity.py           MARIE's coupling sources, compiled without MATLAB
```

The two extensions are separate targets in `CMakeLists.txt` and separate
`install(TARGETS ...)` lines. `_ext` is MIT and links nothing from
`src/cpp_lgpl/`; `_directfn` carries the LGPL notice of each source it compiles
and is loaded through the same `_accelerators.require` path, which names the
module in its error.

### Array layout

Body quantities are torch tensors, complex128, C-ordered, with the port axis
first and the three spatial axes last and contiguous:

```
body currents and fields   (n_ports, 3, n1, n2, n3)
coil currents              (n_ports, n_rwg)
port matrices              (n_ports, n_ports)
```

so that `torch.fft.fftn(J, s=(m1, m2, m3), dim=(-3, -2, -1))` batches over ports
and vector components in one call. MARIE's trailing port axis and its
`(n1,n2,n3,ql)` ordering are not carried over. Sparse operators — the projection
`P`, the precorrection `Z_bc` and `Z_cc` — are torch sparse CSR tensors.

## 3. Order of work

Each stage is finished, checked and merged before the next starts. The body
solver and its Mie test come first: it is the only stage with a closed-form
reference, and every later stage reuses its kernels.

**Stage 1 — quadrature, Tucker and the LGPL extension.**
`quadrature.py`, `tucker.py`, `_directfn`. Checks: a Gauss-Legendre rule of
order *p* integrates polynomials of degree `2p-1` exactly; a Dunavant rule of
degree *d* integrates bivariate monomials up to degree *d* over the unit
triangle exactly and its weights sum to the triangle area; the 26-point Lebedev
set is invariant under the octahedral group; `hosvd` followed by `to_full`
reproduces a random tensor within `tol_HOSVD`; the circulant embedding of a
Toeplitz tensor reproduces the dense product.

`_directfn` has no smooth limit to be compared against: its edge-adjacent and
vertex-adjacent kernels take elements that touch, so there is no separation to
take large. It is checked instead by the invariants that catch a binding fault
and by convergence. Every kernel is invariant under a rigid motion of the whole
configuration, which a vertex array read with the wrong stride or the wrong
order would break, and every kernel converges as the quadrature order rises,
which swapped weights and nodes would break. The triangle self term reaches
machine precision by order 12; the edge and vertex terms converge more slowly
and not monotonically, so their check is a trend rather than a step-by-step
decrease. The voxel kernels are invariant under translation but *not* under
rotation, because their reduced kernel index and scalar basis terms are defined
against the cell's own axes; a test states that, so the body grid is not
quietly turned later.

These kernels divide by the wavenumber, which therefore may not be zero.

**Stage 2 — body solver and the Mie test.**
`body.py`, `vie.py`, `gmres.py`, `preconditioner.py`, and the incident-field and
body-only solve paths. A homogeneous dielectric sphere and then a two-layer
sphere are built in code, illuminated by a plane wave, and solved. The relative
L2 error of E inside the sphere against the analytic Mie series must fall
monotonically over three voxel sizes and, on the coarsest, stay at or below the
value that first converged run records as a test constant with its grid. The
refinement legs carry the `slow` marker; the coarsest grid runs by default. The
compressed body operator is checked against the uncompressed kernel on a small
grid, applied to random currents, within `tol_HOSVD`.

This stage carries the whole numerical core: the N and K kernels, the Tucker
compression, the FFT product, the diagonal body preconditioner and GMRES. The
Mie comparison is the only place in milestone 1 where an absolute answer is
known, so nothing downstream is trusted until it passes.

**Stage 3 — coil matrix.**
`mesh.py`, `coil.py`, `sie.py`, `network.py`. The GMSH reader and the RWG
construction are checked on meshes built in code: every interior edge carries
exactly one basis function, the two triangles of a basis function carry opposite
signs, and the normal current of a basis function is continuous across its own
edge. The four interaction blocks — non-singular, edge-adjacent,
vertex-adjacent, self — are each checked against the transposed pair computed
independently, before `Assembly_SIE_par.m`'s `Z + Zᵀ` is formed, see decision 1
in section 6; the non-singular block is checked against a direct quadrature of
the Galerkin entry, and the surface-impedance block against the Gram matrix of
the basis functions. The assembled matrix is checked absolutely: a perfectly
conducting sphere in a plane wave scatters the cross section the Mie series
gives, and the error falls with mesh refinement. `network.py` arrives here
rather than at stage 5 because the port path needs an impedance to be checked
against the loop's own inductance.

**Stage 4 — coupling kernels.**
`src/cpp/coupling.cpp`, `src/cpp/collocation.cpp`, bound into `_ext`. Checked
against MARIE's own 24 sources, compiled without MATLAB, before any pFFT code
uses them. Section 5 gives the signatures and section 4.3 the parity harness.

**Stage 5 — pFFT coupling and the coupled solve.**
`pfft.py`, `system.py`, `solver.py`. Checks: the projection
reproduces the field of an RWG basis function at the collocation sphere to the
tolerance the least-squares solve achieves; the sparse precorrection is zero
where the expansion cells and the body do not overlap; the final GMRES relative
residual is at or below `tol` for every port.

**Stage 6 — fields and power balance.**
`fields.py`. Checks: the field in the body agrees with a direct integration of
the coupling kernel and the body operator, taken without the projection; the
power the body takes out of the coil's field equals the ohmic loss integrated
from E as `½ σ |E|²` plus the power its own current puts back; and the port
delivers more than the body absorbs. `PLAN.md`'s **Why the power balance is not
closed at the port** paragraph says why the last of those is an inequality and
not an equation.

## 4. Mapping from MARIE

Every ported function names its MARIE source file in its docstring, as
`PLAN.md` requires. `THIRD_PARTY.md` gains no new rows: MARIE, DIRECTFN and
Dunavant are already listed.

### 4.1 Python side

| MARIE | mariepy |
|---|---|
| `src_utils/src_loaders/load_inputs.m` | not ported as a file: MARIE's tolerances and quadrature orders are the defaults of the functions that use them, and `solver.solve` takes `tol` and derives the rest as `load_inputs.m` does |
| `src_physics/em_constants.m` | `constants.NUCLEI` table and `constants.Medium(b0, nucleus)` |
| `src_utils/src_loaders/parse_inputs.m` | not ported: milestone 1 builds one path, so the dispatch has no branches to make. The branch it selects — coil present, no wire, no shield, `pFFT_flag` set — is the whole of `solver.solve` |
| `src_geometry/geo_assembly.m` | `solver.build_geometry` |
| `src_geometry/body_geometry/geo_body_domain.m`, `grid3d.m` | `body.VoxelBody`: grid coordinates, mask, voxel index and degree-of-freedom map |
| `src_physics/src_electromagnetism/em_assembly.m` | `body.contrast`, returning `Mr`, `Mc`, `Mcr` and the inverses |
| `src_geometry/scoil_geometry/mesh_geo/Mesh_Parse.m` | `mesh.SurfaceMesh.read_gmsh22` |
| `Mesh_Permute.m` | `mesh.SurfaceMesh.align_to_lines` |
| `Mesh_CLP.m`, `rwg_geo/Triangle_area.m` | `mesh.SurfaceMesh.centroids`, `edge_vectors`, `edge_lengths`, `rho`, `areas` |
| `Mesh_PreProc.m`, `ProcessLoops.m` | `coil.SurfaceCoil.build`: edges, signs, the dof numbering and the adjacency classes |
| `rwg_geo/get_rwg_vertices.m` | `coil.SurfaceCoil.rwg_vertices`, shape `(n_rwg, 4, 3)` for `rp, rn, r2, r3` |
| `ports_geo/geo_scoil_lumped_elements.m` | `coil.read_lumped_elements` |
| `scoil_geometry/geo_scoil.m` | `coil.SurfaceCoil.build(mesh.SurfaceMesh.read_gmsh22(...), coil.read_lumped_elements(...))` |
| `src_integral_equations/src_vie/src_operators_vie/assembly_N.m` | `vie.kernel_n` |
| `assembly_K.m` | `vie.kernel_k` |
| `cubatures/VV_Nop.m`, `VV_Kop.m`, `kernels_*.m`, `coefficients_*.m`, `weights_points.m`, `points_const_4D.m` | `vie.volume_volume_n`, `vie.volume_volume_k` |
| `cubatures/surface_surface_*.m` | `vie._surface_surface`, singular parts through `_directfn` |
| `singular/singular_{ST,EA,VA}_lin.m`, `points_mapping.m` | `vie.singular_block` |
| `assembly_fft_circ_tucker_pwc.m` | `tucker.circulant_tucker` |
| `src_tucker/hosvd.m`, `nmp.m`, `hosvd_to_full.m` | `tucker.hosvd`, `tucker.mode_product`, `tucker.to_full` |
| `mvp_N_pwc_tucker.m`, `mvp_K_pwc_tucker.m` | `vie.apply_n`, `vie.apply_k` |
| `mvp_G_pwc.m`, `mvp_invG_pwc.m` | `vie.apply_g`, `vie.apply_inv_g` |
| `mvp_vie.m` | `vie.apply_vie`, the body-only operator the Mie test solves |
| `src_sie/src_operators_sie/Assembly_SIE_par.m` | `sie.impedance` |
| `assembly_ns_par.m` | `sie.near_block` |
| `assembly_{ea,va,st}_par.m` | `sie._edge_adjacent`, `_vertex_adjacent`, `_self`, through `_directfn` |
| `assembly_le.m` | `sie.lumped_loads` |
| `excitation_coil.m` | `sie.port_excitation` |
| `src_sie/sie_assembly.m` | `sie.assemble` |
| `src_wsvie/src_pfft/src_svie_pfft/pfft_surface_domain.m` | `pfft.extended_domain` |
| `src_pfft_supporting/pfft_extend_vie_domain.m` | folded into `pfft.extended_domain` |
| `pfft_proj_surface_create_near_lists.m` | `pfft.near_lists` and `pfft.near_body_pairs` |
| `src_pfft_supporting/pfft_proj_find_RWG_centers.m` | folded into `pfft.near_lists`: an RWG's centre is its own edge's midpoint |
| `pfft_proj_find_nearest_voxel.m`, `pfft_proj_find_expansion_cell.m`, `pfft_proj_get_near_indecies.m` | folded into `pfft.near_lists` and `pfft.near_body_pairs` |
| `pfft_proj_pwx_to_collocation.m` | `coupling.collocation_matrix` |
| `pfft_projection_surface_assembly.m` | `pfft.projection` and `pfft.projection_matrix`, with `pfft.scatter_matrix` for `S` |
| `pfft_surface_assemble_direct_bc.m` | `pfft.direct_coupling`, calling `coupling.coupling_n` and `coupling.coupling_k` |
| `src_pfft_coil/pfft_assemble_voxel_bc.m` | `pfft.projected_coupling`, over `pfft.expansion_response` |
| `src_pfft_coil/pfft_assemble_voxel_cc.m` | `pfft.coil_precorrection` |
| `src_wsvie/wsvie_coupling_assembly.m` | `pfft.assemble`, with `pfft.kernels` building both grids' kernels from one table |
| `src_solver/src_ie_solver/solver_wsvie.m`, `src_runners/MARIE_runner.m` | `solver.solve` |
| `src_solver/src_rhs/rhs_assembly.m` | `system.CoupledOperator.right_hand_side` |
| `src_preconditioners/prec_wsvie.m`, `prec_LU.m` | `system.CoupledOperator.preconditioner`, which carries both blocks |
| `src_preconditioners/prec_vie.m` | `preconditioner.body_diagonal` |
| `src_mvp/mvp_svie/mvp_svie_pfft.m` | `system.CoupledOperator.__call__` |
| `src_ie_solver/ie_solver_svie/ie_solver_svie_pfft.m` | `solver.solve_ports` |
| `src_ie_solver/ie_solver_vie/ie_solver_vie.m`, `is_gmres_vie.m` | `solver.solve_body`, the Mie path |
| `src_iterative_solvers/is_gmres_svie.m`, `is_iter_gmres_svie.m` | `gmres.gmres` |
| `src_numerical_linear_algebra/src_inverse/fast_pinv.m` | `gmres._least_squares` |
| `src_iterative_solvers/iterchk.m`, `iterapp.m` | excluded, MathWorks. `gmres.gmres` calls the operator directly |
| `src_electronics/src_network_parameters/np_compute.m` | `network.port_parameters` |
| `np_y2z.m`, `np_z2s.m`, `np_z2y.m` | `network.y_to_z`, `z_to_s`, `z_to_y` |
| `src_electromagnetism/em_ehfield_wsvie.m` | `fields.compute` |
| `em_efield/em_efield_svie/em_efield_svie_pfft.m` | `fields.compute`, `fields.power_balance`, `fields.absorbed_power` |
| `em_hfield/em_hfield_svie/em_hfield_svie_pfft.m` | `fields.compute`, `fields.circular_components` |
| `em_efield/em_efield_vie/em_efield_vie_excitation.m` | `solver.BodyOperator.total_field`, the Mie path |
| `src_utils/src_transformers/to_GPU.m`, `hh_mm_ss.m` | not ported; torch carries the device |

Out of milestone 1 and not ported here: `co_simulation` and everything under
`src_electronics/co_simulation`; `src_wie`, `src_wvie`, `src_wsie` (wire);
`shield_sie_assembly.m` and `src_tt` (shield); `src_basis`, `*_mrgf*`,
`BASIS_runner.m`, `MRGF_runner.m` (reduced-order bases); `em_SNR.m`,
`em_TXE.m`, `em_g_factor.m`, `src_visualizer` (coil evaluation);
`src_utils/src_loaders/CloudMR` and its bundled NIfTI toolbox; the GMSH binary
shipped under `src_geometry/scoil_geometry/GMSH`.

### 4.2 Quadrature provenance

Three of MARIE's quadrature files cannot be ported as they stand, and none of
them needs to be.

- `src_mathematics/src_quads/gauss_1d.m` is Burkardt's `LEGENDRE_SET` with no
  licence line in MARIE's copy. Gauss-Legendre nodes and weights are computed
  from the Golub-Welsch eigenvalue problem instead, in `quadrature.py`, and
  checked by polynomial exactness.
- `gauss_2d_sie.m` is a Duffy map of the tensor-product Gauss rule onto the unit
  triangle, six lines of arithmetic, written anew on top of the above.
- `getLebedevSphere.m` is a File Exchange translation of Laikov's routines with
  no licence line. Only the 26 point *positions* are used —
  `pfft_projection_surface_assembly.m` takes `x`, `y`, `z` and discards the
  weights — and those 26 positions are the three octahedral orbits: the six
  axis directions, the twelve `(±1, ±1, 0)/√2` permutations and the eight
  `(±1, ±1, ±1)/√3` corners. `quadrature.lebedev_26_directions` generates them,
  with no table and no licence question.
- `dunavant_*.m` carries the LGPL notice. Its rules are taken from Dunavant
  (1985), as `PLAN.md` and `THIRD_PARTY.md` record, and checked by polynomial
  exactness rather than against the MATLAB files. Degrees 1 to 6 cover
  milestone 1, which uses `Quad_order_sie = 4`.

The DIRECTFN sources and the RWG singular integrals stay LGPL, unmodified, in
`_directfn`.

### 4.3 The parity harness for the coupling kernels

MARIE ships 24 coupling sources under
`src_integral_equations/src_svie/Cpp_Assembly/src/`. They are one file: the
inner loop of each carries every variant, with all but one commented out. The
variation is exactly two-dimensional — operator N or K, vector component x, y or
z, basis term constant or linear in x, y or z — which is why one N kernel and
one K kernel with `component` and `basis_term` arguments reproduce all 24.

`tests/parity.py` compiles the originals without MATLAB and
`tests/test_coupling.py` compares. Two obstacles, both handled at test-build
time:

- `mex.h` is absent. A four-line header supplies `mxComplexDouble` as a struct
  of two doubles, which is the only thing the 24 sources use it for.
- `get_source_coords_mat`, `vec_norm_l2` and `compute_edge_length_v` are defined
  at file scope with external linkage in all 24, so linking them together is a
  multiple definition. Each copy has those three given internal linkage, which
  needs no `objcopy` and works wherever a compiler does.

The test builds a random RWG geometry and a random set of cells and asserts
that `coupling.coupling_n(..., basis_term=b)[:, c]` reproduces
`Assemble_rwg_coupling_matrix_N_{x,y,z}{,1,2,3}` for every `(c, b)` pair, and
likewise for K. It runs on CPU and skips only when no C++ compiler is on the
path.

The sources are kept in `tests/marie/`. MARIE 3.0 is MIT, so unlike the
DIRECTFN family there is no licence reason to hold them at arm's length, and a
comparison that needs no setup is a comparison every pull request runs. They are
test-only: nothing under `tests/` reaches the wheel, and `_ext` never links
them.

## 5. The coupling kernels

Two functions in `coupling.py` replace the 24 sources. Both take the RWG
geometry and one observer per basis function, so a near list is a flat list of
pairs, and both run on either device.

```python
def coupling_n(
    corners,                 # (n, 4, 3): r_p, r_n, r_2, r_3
    points,                  # (n, 3): the observer paired with each basis function
    medium,                  # supplies k0 and j omega eps_0
    *,
    triangle_order=4,        # degree of the Dunavant rule on each triangle
    cell_size=None,          # cell pitch, or None to observe at the point
    cell_order=2,            # points per axis of the Gauss rule over the cell
    basis_term=0,            # 0 constant, 1-3 linear in the cell's own x, y, z
) -> torch.Tensor            # (n, 3), complex

def coupling_k(...)          # same parameters, same shape
```

MARIE's `res` and `k0` arrive inside `medium` and `cell_size`; its packed
`sie_quads` and `vie_quads`, whose leading element carried the point count, are
replaced by the orders, so a mismatch cannot read past the end. The cell
average is not multiplied by the cell volume: the caller does that when it
wants the integral, as `pfft_surface_assemble_direct_bc.m` does and
`pfft_projection_surface_assembly.m` does not.

`collocation_matrix` is the companion that `pfft_proj_pwx_to_collocation.m`
builds:

```python
def collocation_matrix(
    centres,                 # (n_cells, 3)
    points,                  # (n_points, 3)
    medium,
    *,
    cell_size,
    cell_order=2,
    n_basis=1,               # 1 for the piecewise-constant cell basis, 4 for linear
) -> torch.Tensor            # (3 * n_points, 3 * n_basis * n_cells), complex
```

returning the matrix whose least-squares solve gives the projection weights of
one RWG function. Rows run component-major over the points; columns run
component-major, then basis term, then cell, as MARIE orders them.

## 6. Decisions taken

Points where MARIE's reference leaves a choice. `PLAN.md` carries the ones
that are design decisions; the rest are recorded here.

1. **Two of `PLAN.md`'s milestone 1 validation criteria did not test what they
   claimed, and the table now states the checks that do.**
   `Assembly_SIE_par.m` computes the non-singular, edge-adjacent and
   vertex-adjacent blocks for one triangle ordering and then sets
   `Z = Z + Z.'`; `np_compute.m` sets `YP_s = (Ip + Ip.')/2`. A port that copies
   both — and it should, they are how MARIE builds the operators — satisfies
   "port impedance matrix symmetric within `tol`" and "coupled port matrix
   symmetric within `tol`" for any physics whatever. The criteria are now
   reciprocity of each interaction block against the independently computed
   transposed pair, and reciprocity of `Ip` before the symmetrisation. Stage 3
   and stage 6 check those.

2. **Both coupling operators are assembled, as MARIE assembles them.**
   `mvp_svie_pfft.m` applies only `pfft_Z_bc_N`; `pfft_Z_bc_K` is first used in
   `em_hfield_svie_pfft.m`. Since milestone 1 delivers the H field, K is needed
   whether or not the solve touches it, and assembling it alongside N costs one
   more pass over the same near lists and the same quadrature points.

3. **GMRES exits on a happy breakdown.**
   `is_iter_gmres_svie.m` divides by `H(k+1,k)` without testing it, and solves
   the least-squares problem with an economy SVD of the Hessenberg
   (`fast_pinv.m`) rather than Givens rotations. The port keeps the SVD, which
   torch does well, and returns the iterate when `H(k+1,k)` falls below the
   working precision — the Krylov space is then invariant and the iterate is
   exact. `PLAN.md` records the deviation under **GMRES**.

4. **The Mie criterion splits across the two test legs.**
   `PLAN.md` puts refinement studies behind the `slow` marker, and the milestone
   1 criterion is monotone decrease over three voxel sizes *and* a bound on the
   coarsest. The default run checks the coarsest grid against its recorded
   value; the `slow` leg checks that the error falls with refinement.
   `PLAN.md`'s **Test layout** now says so.

5. **A lumped element is matched to its mesh edges by its own number.**
   `Mesh_PreProc.m` matches the *i*-th entry of the element file to the *i*-th
   smallest physical line tag in the mesh, so an element file listed out of tag
   order silently drives the wrong edges. `coil.SurfaceCoil.build` matches on
   the element's `number`, which `geo_scoil_lumped_elements.m` already reads and
   MARIE then ignores, and raises when a number names no interior edge.

6. **Every edge shared by two triangles carries a basis function.**
   `Mesh_PreProc.m` reaches the same set through a boundary flag `kn` that is
   0 on a rim edge, the physical tag on a tagged edge and −1 elsewhere, and it
   would give a half basis function to a tagged edge that lies on a rim. The
   port takes the geometric rule instead: two triangles, one basis function;
   one triangle, none. A tag on a rim edge therefore leaves its port empty,
   which `build` reports rather than solving a coil whose current leaves the
   sheet.

7. **The conductor's surface resistance is counted once.**
   `assembly_st_par.m` builds `Z_ST_local` with `ZR_DE + ZR_DE_losses` in it and
   then stores `Z_ST_local + Z_ST_local_losses`, so the loss term reaches the
   matrix twice while `ZR_DE_losses_matrix`, which the network parameters read,
   carries it once. `sie.surface_block` adds it once. It is also derived rather
   than transcribed: the overlap of two basis functions over their shared
   triangle is exact from the barycentric moment, the integral of
   `lambda_u lambda_v` over a triangle of area `A` being `A (1 + delta_uv) / 12`,
   and a test states it as the Gram matrix of the basis functions. Evaluating
   MARIE's three-branch `staticq` on a random triangle reproduces it exactly,
   which is how the reading was confirmed.

8. **The edge-adjacent rule runs at order 10, where MARIE uses 6.**
   The DIRECTFN edge-adjacent integral converges at a rate set by the shape of
   the two triangles. On the near-equilateral triangles of a sphere, order 6
   already leaves the block reciprocal to 4e-7. On the elongated triangles a
   loop coil's strip produces, order 6 leaves 1e-4 — above `PLAN.md`'s `tol` —
   and order 10 brings it under. The default is 10 here; a mesh of very
   elongated elements may still need more, and a test states that the defect
   falls with the order rather than fixing a number.

9. **The coil matrix is checked on a closed conductor, not at a port.**
   `PLAN.md`'s **Why the coil matrix is not checked at a port** paragraph and
   its milestone 1 table now carry the reason and the criterion.

10. **The coupling kernels stayed in torch.**
   `PLAN.md`'s **C++ kernels** constraint keeps in torch the work torch can
   batch, and moves it to C++ when a profile shows it dominates. The 24 sources
   are a quadrature over the same `N` and `K` kernels `vie.py` already carries:
   the coupling of a basis function `f` to a point is
   `E = 1 / (j omega eps_0) * integral N(r - r') f(r') dS'` and
   `H = integral K(r - r') x f(r') dS'`, which torch batches over pairs in one
   contraction. `coupling.py` therefore reuses `vie.green_n` and `vie.green_k`,
   which the Mie test already validated, and reproduces all 24 originals to
   3e-14. `src/cpp/` keeps its bindings module for the first kernel a profile
   sends there.

11. **MARIE halves Burkardt's Dunavant weights, and that is what carries the
   basis function's own factor.** `dunavant_rule.m` returns `0.5 * w`, so the
   surface rule sums to the reference triangle's area rather than to one, and
   the coupling sources multiply by the edge length alone rather than by
   `L / (2 A)`. `quadrature.dunavant` keeps `PLAN.md`'s convention, weights
   summing to one, and `coupling.py` halves them where MARIE's rule already is
   halved. Read the other way round — the rule normalised to one and the basis
   function written out — the two agree, and a test states the coupling as the
   field the basis function itself radiates.

12. **The power balance is closed inside the body, not at the port.**
   `PLAN.md`'s milestone 1 table asked for the body-absorbed power to equal
   "the absorbed power predicted from the port currents". There is no such
   prediction: the power a port delivers is spent on the conductor, on the body
   and on radiation, and subtracting the coil's own dissipation leaves the
   body's absorption plus the interference between the coil's radiation and the
   body's, which needs a far field to separate. The criterion is now the balance
   that is exact — extinction equals absorption plus scattering, inside the
   body — together with the inequality at the port and a direct-integration
   check on the field itself.

A further decision sits in section 4.2 and in `PLAN.md`'s **Excluded** list rather
than here: `gauss_1d.m` and `getLebedevSphere.m` ship without a licence, so
neither is ported, and the rules they carry are obtained from first principles.
