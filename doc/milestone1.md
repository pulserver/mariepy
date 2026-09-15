# Milestone 1 — surface coil, voxel body, pFFT coupling

The design for the first milestone of the MARIE 3.0 port: the path MARIE takes
by default for a surface coil around a piecewise-constant voxel body, coupled by
precorrected FFT, ending with each port's body currents, network parameters and
E and H fields.

`PLAN.md` states the scope, the validation criteria and the constraints. This
note fixes the module layout, the order in which the stages are built and
checked, the MARIE-to-mariepy mapping, and the signatures of the compiled
coupling kernels. No solver code lands with it.

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
    settings.py             tolerances and quadrature orders
    quadrature.py           Gauss-Legendre, triangle rules, Dunavant, Lebedev
    tucker.py               HOSVD, n-mode product, circulant FFT embedding
    mesh.py                 GMSH 2.2 reader, triangle geometry, meshes in code
    coil.py                 RWG basis, ports, lumped elements
    body.py                 voxel grid, tissue contrast, degree-of-freedom map
    vie.py                  body kernels N and K, and their products
    sie.py                  coil EFIE matrix, lumped loads, port excitation
    pfft.py                 extended grid, projection, precorrection
    system.py               the coupled operator
    preconditioner.py       coil LU block, body diagonal block
    gmres.py                restarted GMRES with a split preconditioner
    network.py              Y, Z and S at the ports
    fields.py               E and H on the body grid
    solver.py               drives geometry, operators, solve, network, fields

src/cpp/                    -> mariepy._ext, MIT
    module.cpp              bindings
    threads.hpp             partitioning of independent work
    coupling.cpp            RWG-to-voxel N and K kernels
    collocation.cpp         voxel-to-collocation-point dyadic kernel
    sie_nonsingular.cpp     coil matrix, non-singular triangle pairs
    vie_volume.cpp          body kernel, volume-volume quadrature

src/cpp_lgpl/               -> mariepy._directfn, LGPL, notices kept
    NOTICE.md               what is carried, and the two build changes
    module_directfn.cpp     bindings
    directfn_vie/           the voxel family, linking as its sources stand
    directfn_rwg/           direct_ws_{st,ea,va}_rwg and their headers
    rwg_namespace_*.cpp     one wrapper per RWG source, giving it a namespace

tests/
    test_quadrature.py  test_tucker.py  test_mesh.py  test_coil.py
    test_body.py        test_vie.py     test_mie.py   test_sie.py
    test_pfft.py        test_coupling_kernels.py      test_gmres.py
    test_network.py     test_fields.py  test_power_balance.py
    mie.py              analytic reference, scipy special functions
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
`fields.py`. Checks: the power absorbed in the body, integrated from E as
`½ σ |E|²` over the voxels, equals the absorbed power predicted from the port
currents, within `tol`; and the coupled port matrix is reciprocal before the
symmetrisation `np_compute.m` applies.

## 4. Mapping from MARIE

Every ported function names its MARIE source file in its docstring, as
`PLAN.md` requires. `THIRD_PARTY.md` gains no new rows: MARIE, DIRECTFN and
Dunavant are already listed.

### 4.1 Python side

| MARIE | mariepy |
|---|---|
| `src_utils/src_loaders/load_inputs.m` | `settings.Settings`, a frozen dataclass of `tol`, `tol_HOSVD`, quadrature orders and the pFFT kernel width, with `Settings.from_json` |
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
| `cubatures/VV_Nop.m`, `VV_Kop.m`, `kernels_*.m`, `coefficients_*.m`, `weights_points.m`, `points_const_4D.m` | `_ext.vie_volume_block` |
| `cubatures/surface_surface_*.m` | `vie._surface_surface`, singular parts through `_directfn` |
| `singular/singular_{ST,EA,VA}_lin.m`, `points_mapping.m` | `vie.singular_block` |
| `assembly_fft_circ_tucker_pwc.m` | `tucker.circulant_tucker` |
| `src_tucker/hosvd.m`, `nmp.m`, `hosvd_to_full.m` | `tucker.hosvd`, `tucker.mode_product`, `tucker.to_full` |
| `mvp_N_pwc_tucker.m`, `mvp_K_pwc_tucker.m` | `vie.apply_n`, `vie.apply_k` |
| `mvp_G_pwc.m`, `mvp_invG_pwc.m` | `vie.apply_g`, `vie.apply_inv_g` |
| `mvp_vie.m` | `vie.apply_vie`, the body-only operator the Mie test solves |
| `src_sie/src_operators_sie/Assembly_SIE_par.m` | `sie.impedance` |
| `assembly_ns_par.m` | `_ext.sie_nonsingular` |
| `assembly_{ea,va,st}_par.m` | `sie._edge_adjacent`, `_vertex_adjacent`, `_self`, through `_directfn` |
| `assembly_le.m` | `sie.lumped_loads` |
| `excitation_coil.m` | `sie.port_excitation` |
| `src_sie/sie_assembly.m` | `sie.assemble` |
| `src_wsvie/src_pfft/src_svie_pfft/pfft_surface_domain.m` | `pfft.extended_domain` |
| `src_pfft_supporting/pfft_extend_vie_domain.m` | `pfft._extend_grid` |
| `pfft_proj_surface_create_near_lists.m` | `pfft.near_lists` |
| `src_pfft_supporting/pfft_proj_find_RWG_centers.m` | `coil.rwg_centres` |
| `pfft_proj_find_nearest_voxel.m`, `pfft_proj_find_expansion_cell.m`, `pfft_proj_get_near_indecies.m` | `pfft._nearest_voxel`, `pfft._expansion_cells`, `pfft._near_cells` |
| `pfft_proj_pwx_to_collocation.m` | `_ext.collocation_matrix` |
| `pfft_projection_surface_assembly.m` | `pfft.projection`, returning the sparse `P` and `S` blocks |
| `pfft_surface_assemble_direct_bc.m` | `pfft.direct_coupling`, calling `_ext.coupling_n` and `_ext.coupling_k` |
| `src_pfft_coil/pfft_assemble_voxel_bc.m` | `pfft.projected_coupling` |
| `src_pfft_coil/pfft_assemble_voxel_cc.m` | `pfft.coil_precorrection` |
| `src_wsvie/wsvie_coupling_assembly.m` | `pfft.assemble` |
| `src_solver/src_ie_solver/solver_wsvie.m` | `solver.solve` |
| `src_solver/src_rhs/rhs_assembly.m` | `solver.right_hand_side` |
| `src_preconditioners/prec_wsvie.m`, `prec_LU.m` | `preconditioner.coil_lu` |
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
| `em_efield/em_efield_svie/em_efield_svie_pfft.m` | `fields.electric` |
| `em_hfield/em_hfield_svie/em_hfield_svie_pfft.m` | `fields.magnetic` |
| `em_efield/em_efield_vie/em_efield_vie_excitation.m` | `fields.electric_from_body`, the Mie path |
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

`tests/test_coupling_kernels.py` compiles the originals without MATLAB and
compares. Two obstacles, both handled at test-build time:

- `mex.h` is absent. A shim header supplies `mxComplexDouble` as a struct of two
  doubles, which is the only thing the 24 sources use it for.
- `get_source_coords_mat`, `vec_norm_l2` and `compute_edge_length_v` are defined
  at file scope with external linkage in all 24, so linking them together is a
  multiple definition. Each object gets `objcopy --localize-symbol` on those
  three before the link.

The test builds a random RWG geometry and a random voxel set and asserts that
`_ext.coupling_n(..., component=c, basis_term=b)` reproduces
`Assemble_rwg_coupling_matrix_N_{x,y,z}{,1,2,3}` to floating-point precision,
for every `(c, b)` pair, and likewise for K. It carries the `slow` marker, runs
on CPU, and skips when the environment variable naming a MARIE checkout is unset
or no C++ compiler is on the path — the sources are not vendored into this
repository.

## 5. The compiled coupling kernels

Two functions in `_ext` replace the 24 sources. Both take the RWG geometry for a
whole batch of basis functions and a set of observation points, both evaluate on
CPU buffers, and both return an array the caller moves to its device.

```cpp
// src/cpp/coupling.cpp

py::array_t<std::complex<double>> coupling_n(
    py::array_t<double, py::array::c_style | py::array::forcecast> observation_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> rwg_vertices,
    py::array_t<double, py::array::c_style | py::array::forcecast> triangle_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> triangle_points,
    py::array_t<double, py::array::c_style | py::array::forcecast> voxel_weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> voxel_nodes,
    double voxel_size,
    double wavenumber,
    int component,
    int basis_term,
    int threads);

py::array_t<std::complex<double>> coupling_k(  // same parameters
    ...);
```

| Parameter | Shape and meaning |
|---|---|
| `observation_points` | `(n_obs, 3)` — voxel centres for the direct coupling, collocation points for the projection |
| `rwg_vertices` | `(n_rwg, 4, 3)` — the free vertex of the positive triangle, the free vertex of the negative triangle, and the two shared-edge vertices, in that order. MARIE's `r_p, r_n, r_2, r_3` in one array |
| `triangle_weights` | `(n_tri,)` — Dunavant weights on the reference triangle |
| `triangle_points` | `(n_tri, 3)` — the matching barycentric coordinates |
| `voxel_weights` | `(n_gauss,)` — Gauss-Legendre weights on `[-1, 1]` |
| `voxel_nodes` | `(n_gauss,)` — the matching nodes; the cubature over the voxel is their tensor cube |
| `voxel_size` | voxel pitch in metres, MARIE's `res` |
| `wavenumber` | free-space `k0` in rad/m. The kernel forms `ce = i k0 c0 ε0` from it, as MARIE's sources do, so the scalings `-1/(4π ce)` for N and `-1/(4π)` for K need no further argument |
| `component` | `0`, `1`, `2` for the x, y or z component of the tested field |
| `basis_term` | `0` for the constant basis, `1`, `2`, `3` for the term linear in x, y or z within the voxel. Milestone 1 passes `0`; the others are milestone 2's piecewise-linear basis |
| `threads` | number of worker threads; `0` asks for `std::thread::hardware_concurrency()` |

Returns `(n_rwg, n_obs)` complex128, C-ordered: the batch axis first, as
`PLAN.md` requires. MARIE's Fortran-ordered `(N_vox, N_rwg)` output is this
array's transpose, which is why the parity test compares against a transpose.
The two quadrature arrays replace MARIE's packed `sie_quads` and `vie_quads`
vectors, whose leading element carried the point count; here the shapes carry
it, and a mismatch raises rather than reading past the end.

The observation-point cubature is not applied: as in
`pfft_surface_assemble_direct_bc.m`, the caller multiplies by `res³` when it
wants the voxel integral, and does not when it wants the value at a collocation
point.

`collocation.cpp` binds the companion kernel, the voxel-to-collocation dyadic
Green matrix that `pfft_proj_pwx_to_collocation.m` builds:

```cpp
py::array_t<std::complex<double>> collocation_matrix(
    py::array_t<double, ...> voxel_centres,      // (n_cells, 3)
    py::array_t<double, ...> collocation_points, // (n_col, 3)
    py::array_t<double, ...> voxel_weights,      // (n_gauss,)
    py::array_t<double, ...> voxel_nodes,        // (n_gauss,)
    double voxel_size,
    double wavenumber,
    int n_basis_terms,                           // 1 for PWC, 4 for PWL
    int threads);
```

returning `(3 · n_col, 3 · n_basis_terms · n_cells)` complex128, the matrix whose
least-squares solve gives the projection weights of one RWG function.

Threading in both follows the package template: `threads.hpp` partitions the
observation-point loop across `std::thread` workers, each writing a disjoint
slice of the output, replacing MARIE's `#pragma omp parallel for`. No locks and
no atomics, so the result does not depend on the thread count — a test asserts
that for one and for many.

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

A further decision sits in section 4.2 and in `PLAN.md`'s **Excluded** list rather
than here: `gauss_1d.m` and `getLebedevSphere.m` ship without a licence, so
neither is ported, and the rules they carry are obtained from first principles.
