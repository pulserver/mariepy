# mariepy plan

## Scope

This package computes virtual observation points (VOPs) for the local and
head-average SAR of multi-channel RF transmit coils. Given a coil model (a
surface mesh with ports) and a population of head models, it:

1. solves the electromagnetic problem for each port;
2. forms each channel's field and the per-voxel Q matrices;
3. averages them over 10 g;
4. compresses them into VOPs;
5. writes a file that pypulseqpp's `safety.check_sar` reads.

The first milestone is a Python port of the path MARIE 3.0 takes by default: a
surface-integral-equation coil around a piecewise-constant
volume-integral-equation body, coupled by precorrected FFT. The later stages
build on that solver once it reproduces an analytic sphere: Q matrices,
averaging, compression, and head models from public MRI data.

Out of scope: evaluating the SAR of pulse sequences, which pypulseqpp does;
integration with any scanner; and models of specific commercial coils.

## Output contract

A VOP file is a NumPy `.npz` archive that pypulseqpp loads with
`pypulseqpp.safety.read_vops`. It holds matrices and metadata. The metadata is a
JSON string, so the archive loads without pickle.

| Entry | Shape and type | Meaning |
|---|---|---|
| `vops` | (N, Nc, Nc), complex, Hermitian | Virtual observation points for 10 g local SAR |
| `global_matrix` | (N_body, Nc, Nc), complex, Hermitian | Head-average SAR matrix of each body model |
| `metadata` | JSON string | Coil identity, frequency, drive, method and provenance, below |

**Units and convention.** Local SAR in W/kg for channel drive phasors v at peak
amplitude is vᴴQv, with

Q_ij = σ / (2ρ) · e_i* · e_j,

where e_c is channel c's electric field per unit drive, σ the conductivity and ρ
the mass density. The global matrices follow the same convention, averaged over
each body model's head mass. Every matrix therefore has units of W/kg per unit
drive squared.

**Drive.** A channel's unit drive is set by how the channel is excited:

- for a port-driven coil, 1 √W incident at the channel's input after the
  matching network;
- for a coil defined by fixed current patterns, one unit of that pattern.

The unit must be the same for every channel. Channel order is the coil model's.

**Metadata.**

| Key | Content |
|---|---|
| `coil` | Identity string of the coil model |
| `frequency_hz` | Frequency the fields were solved at |
| `drive_unit` | Definition of a channel's unit drive |
| `channels` | Channel names in matrix order |
| `averaging` | Target mass and method |
| `bodies` | Identifiers of the body models, in the order of `global_matrix` |
| `compression_margin` | Overestimation allowed in VOP compression |
| `mariepy_version` | Version that wrote the file |
| `data_licence` | Licence of the file, set by its body models |

Today `read_vops` reads a single (Nc, Nc) global matrix and no metadata. The
stacked global matrices and the metadata need a pypulseqpp release that reads
them.

## MARIE port

The port follows MARIE 3.0 (<https://github.com/cloudmrhub/marie-tools>, MIT).
Milestone 1 is its case for a surface coil around a voxel body, with no wire
coil and no shield, and precorrected FFT coupling. MARIE picks that case in
`src_utils/src_loaders/parse_inputs.m`. Its runner,
`src_runners/MARIE_runner.m`, calls the stages in order:

| Stage | MARIE entry point | Functions on this path |
|---|---|---|
| Inputs | `load_inputs` | JSON simulation file, body `.mat`, coil mesh and its lumped-element JSON |
| Geometry | `geo_assembly` | body voxel grid; GMSH 2.2 surface mesh parsed into RWG edges and ports |
| Operators | `wsvie_assembly` | body kernel tensors `assembly_N`, `assembly_K` and their Tucker/FFT form `assembly_fft_circ_tucker_pwc`; coil matrix `Assembly_SIE_par`; coil–body coupling in `src_wsvie/src_pfft/src_svie_pfft`, `src_pfft_coil` and `src_pfft_supporting` |
| Solve | `solver_wsvie` | preconditioner `prec_wsvie`, right-hand sides `rhs_assembly`, one GMRES solve per port through `ie_solver_svie_pfft` and `mvp_svie_pfft` |
| Network | `np_compute` | Y, Z and S parameters of the ports |
| Fields | `em_ehfield_wsvie` | `em_efield_svie_pfft`, `em_hfield_svie_pfft` |

**Milestones.** The port covers MARIE 3.0 except its CloudMR export, in four
milestones, each building on the previous one:

| Milestone | Content | Needed for |
|---|---|---|
| 1. Default path | Surface coil, piecewise-constant body, pFFT coupling, network parameters, E and H fields | Coils defined by fixed current patterns; the solver every later milestone reuses |
| 2. Accuracy and enclosures | Piecewise-linear body basis; RF shields with tensor-train cross coupling, porting TT-Toolbox's `dmrg_cross` | Shielded coils; measuring how the basis changes peak 10 g SAR at tissue boundaries |
| 3. Circuits and wires | Co-simulation (tuning, matching, decoupling, preamplifier decoupling, frequency sweeps), with a scipy global optimiser in place of `particleswarm`; wire coils | Port-driven coils, whose channel drive depends on the matching network |
| 4. Speed and coil evaluation | Incident- and total-field bases and MRGF reduced-order solves; SNR, transmit-efficiency and g-factor maps; visualizer | Many coil configurations in one body; coil design comparisons |

Milestone 1 ends with each port's body currents, fields and network parameters.
From milestone 3 onward, co-simulation forms the channel drive inside mariepy.

**Compiled kernels.** MARIE's C++ sources are bound with pybind11 into the
package's single `_ext` module, following the package template and pypulseqpp.

- **Coupling kernels.** For a surface coil, MARIE ships 24 coupling sources that
  differ only in field component, piecewise-linear basis term and operator (N or
  K). They become one N kernel and one K kernel taking the component and basis
  term as arguments. Both are batched quadratures over the same `N` and `K`
  kernels the body operator already uses, so under **C++ kernels** below they
  stay in torch; the 24 sources remain the reference they are checked against.
  The wire-coil sources are checked for the same structure when milestone 3
  ports them.
- **Singular integrals.** The DIRECTFN sources (`direct_ws_*_rwg` for the coil,
  `solve_ea`, `solve_st`, `solve_va` and their headers for the body) carry an
  LGPL notice. They build as a separate extension module with that notice kept,
  never inside `_ext`.
- **Threading.** OpenMP loops become standard C++ threads partitioning
  independent work, as in the template.

**MATLAB-side code.**

- **Numerical building blocks.** Arrays, FFTs, sparse projection matrices and
  dense factorisations use torch (`torch.fft`, sparse CSR tensors,
  `torch.linalg`), so one code path runs on CPU or CUDA.
- **GMRES.** MARIE's own GMRES (`is_gmres_svie.m`, `is_iter_gmres_svie.m`) is
  ported to torch together with its split preconditioner: an LU-factored coil
  block and a diagonal body block. The Arnoldi division by the subdiagonal
  Hessenberg entry, which MARIE leaves unguarded, exits on a happy breakdown.
  MathWorks' `iterchk.m` and `iterapp.m`, which MARIE calls only to apply the
  operator, are not ported.
- **Loops.** The loops MATLAB runs with `parfor` are the body kernel tensor and
  the coil matrix's non-singular terms. They go to C++ in `_ext` when a profile
  shows they dominate.
- **Array layout.** Arrays follow C order with the batch dimension first:
  (ports, …) rather than MATLAB's trailing port axis.

**Order of work.** No MATLAB reference is available, so each stage is checked
against physics:

1. **Body solver.** A homogeneous sphere, then a layered one, in a plane-wave
   incident field, compared with the analytic Mie series.
2. **Coil matrix.** Checked by reciprocity of each interaction block, and by
   convergence as the surface mesh is refined.
3. **Coupling and fields.** Checked by reciprocity of the coupled port matrix and
   by power balance: the power absorbed in the body, integrated from the fields,
   matches the absorbed power predicted from the port currents.

MARIE symmetrises two of these quantities by construction: `Assembly_SIE_par.m`
forms `Z + Zᵀ`, and `np_compute.m` forms `(Ip + Ipᵀ)/2`. A reciprocity check
applied after either step passes whatever the physics says, so both checks are
made on the quantity before its symmetrisation.

Validation sets the tolerances.

## Validation

No MATLAB reference run is available. Each stage is validated against physics,
against closed-form results, or against the stage beneath it. Numerical
tolerances default to MARIE's (`tol` for GMRES, with `tol_HOSVD`, `tol_TT` and
`tol_ACA` derived from it in `load_inputs.m`) and are exposed as parameters.

Criteria are stated as invariants. Where a threshold can only come from a
converged run, the first such run records it as a test constant, with the grid
it was measured on.

**Milestone 1.**

| Check | Reference | Pass criterion |
|---|---|---|
| Body solver | Analytic Mie series for a homogeneous sphere, then a layered sphere, in a plane-wave incident field | Relative L2 error of E inside the sphere falls monotonically over three voxel sizes, and on the coarsest stays at or below its recorded value. Measured over the interior, for the reason below |
| Linear solve | – | Final GMRES relative residual at or below `tol` for every port |
| Compressed body operator | The uncompressed kernel on a small grid | Operator applied to random currents agrees within `tol_HOSVD` |
| Coupling kernels | MARIE's original C++ coupling sources, kept in `tests/marie/` and compiled without MATLAB: a header defining `mxComplexDouble` replaces `mex.h`, and the three helper functions each source repeats are given internal linkage so all 24 variants link into one library | For every component and basis term, the new N and K kernels reproduce the corresponding original to floating-point precision on random geometry |
| Coil matrix | Analytic Mie series for a perfectly conducting sphere in a plane-wave incident field | Each interaction block equals the independently computed transposed pair within `tol`, before `Z + Zᵀ` is formed; the scattering cross section falls monotonically towards the series over three mesh refinements and on the coarsest stays at or below its recorded value |
| Coupled system | – | The port matrix is reciprocal within `tol` before `(Ip + Ipᵀ)/2` is formed; the field in the body agrees with a direct integration of the coupling kernel and the body operator, taken without the projection; the power the body takes out of the coil's field equals the ohmic loss integrated from E plus the power the body's own current puts back, within `tol` |

**Why the power balance is not closed at the port.** The power a port
delivers is spent on the conductor, on the body and on radiation, and the three
cannot be separated from the port alone: the coil's field and the body's
interfere, so what the pair radiates is not what the coil would radiate by
itself plus what the body would. Subtracting the coil's own dissipation from the
delivered power therefore leaves the body's absorption plus that interference,
and the two cannot be told apart without a far field, which milestone 1 does not
compute. What is exact, and is what the criterion above states, is the balance
inside the body: the power it takes out of the coil's field equals what it turns
to heat plus what its own current puts back. The port is checked instead by the
inequality that must hold — it delivers more than the body absorbs — and by the
field itself, against a direct integration.

**Why the coil matrix is not checked at a port.** A delta-gap feed puts the
whole drive on one ring of edges, and the charge that piles up there grows as
the mesh is refined, so a port impedance does not settle: refining a loop coil
sixfold moves its reactance by a few per cent and shows no sign of stopping. The
closed conductor has no such feed. Its scattering cross section is the power the
incident field does on the induced current, which the solve delivers without a
far field, and the perfectly conducting sphere has that cross section in closed
form. The coil matrix is therefore checked there, and the port path is checked
against the inductance the loop's own geometry implies.

**What the piecewise-constant basis costs at a boundary.** The normal electric
field jumps across a dielectric boundary by the contrast ratio, and this basis
puts that jump on a staircase. The error is therefore concentrated in the
boundary voxels, falls about as fast as the voxel size, and grows with contrast:
a sphere at brain-like permittivity needs far more than ten voxels across its
radius before its interior field is worth quoting, while at a permittivity of 2
ten voxels already suffice. MARIE's own example runs the piecewise-linear basis
at 2 mm for this reason. The Mie criterion is therefore measured over the
interior rather than over every voxel, and the refinement legs carry the grids
they were measured on.

**Milestone 2.**

- The piecewise-linear basis meets the Mie criterion, and its error is reported
  against the piecewise-constant basis on the same grids.
- Shielded coils meet the reciprocity and mesh-convergence criteria.
- Tensor-train coupling agrees with the fully assembled coupling on a small
  case, within `tol_TT`.

**Milestone 3.**

- Co-simulation reproduces closed-form results for simple lumped networks: a
  series RLC, and an L-section matching a known load.
- On a coil, the tuned and matched reflection at the Larmor frequency meets the
  target set in the coil's element file.

**Milestone 4.**

- A solve in a precomputed field basis agrees with the direct solve for the same
  coil and body, within the basis tolerance.
- SNR and transmit-efficiency maps agree with their definitions evaluated
  directly from the fields.

**Q matrices, averaging and VOPs.**

- 10 g averaging agrees with the IEC/IEEE 62704-1 reference implementation at
  random drives.
- For random drives, the largest VOP SAR is at least the largest averaged voxel
  SAR and at most that plus the compression margin.
- Each body's global matrix reproduces the head-average SAR integrated directly
  from the fields.

**Test layout.** Every check runs on CPU on coarse grids, and the CUDA leg skips
on a machine without a device. Refinement studies carry the `slow` marker and
stay out of the default run, so the default run checks the coarsest grid against
its recorded value and the `slow` leg checks that the error falls with
refinement. A failing check reports the measured error and the grid.

**Acceptance.** A stage is accepted on the evidence its own checks produce, not
on a reading of the physics by whoever merges it. A stage lands as one or more
pull requests, each carrying the checks its table above states and merging when
they pass; a stage splits where vendored sources would otherwise bury the code
written here. Where a stage meets something these criteria do not settle — a quantity
with no reference, a choice between two defensible conventions — the pull
request states the reading it took and why, in `PLAN.md` where the reading is a
design decision and in the test name where it is an invariant.

What this leaves uncovered is recorded rather than implied away:

- A port that reproduces MARIE reproduces its errors with it. The Mie series is
  the one independent reference here, and it reaches the body solver only.
- Reciprocity and power balance are necessary and not sufficient. A wrong sign
  or a wrong edge length can satisfy both.
- Nothing above checks that the inputs are physically sensible: a coil geometry,
  a tissue property at the working frequency, a body model's labels. Those are
  checked where they enter, against their own sources.

## Constraints

**Code licence.** mariepy is MIT.

- **Permissive code** (MIT, BSD, Apache-2.0) may be ported with its copyright
  notice kept, and each ported source is listed in `THIRD_PARTY.md`. This covers
  MARIE 3.0 itself and TT-Toolbox's `dmrg_cross`.
- **MARIE files kept verbatim as a test oracle** are listed the same way. MARIE
  3.0 is MIT, so a source a check compares against is kept rather than described:
  a comparison a developer must set up is a comparison that does not run. Such
  files live under `tests/`, never under `src/`, so nothing in them reaches the
  wheel or links into `_ext`. The 24 coupling sources in `tests/marie/` are the
  first of them.
- **MARIE's LGPL files** stay outside the MIT code:
  - the DIRECTFN singular integrals, built as a separate extension module with
    their notice;
  - the Dunavant triangle quadrature files, whose rules are taken from Dunavant
    (1985) and checked by polynomial exactness, so the LGPL files themselves are
    not carried.
- **Excluded:** GPL, AGPL, non-commercial or unlicensed code, and model weights
  under such terms. MathWorks' `iterapp.m` and `iterchk.m` are excluded too, and
  so are two quadrature files MARIE ships without a licence: `gauss_1d.m`,
  Burkardt's `LEGENDRE_SET`, and `getLebedevSphere.m`, a translation of Laikov's
  routines. Gauss–Legendre nodes and weights come from the Golub–Welsch
  eigenvalue problem instead, and the 26 Lebedev directions — the only part of
  that rule the precorrected FFT projection uses, since it discards the weights —
  are the three octahedral orbits, generated in code.

**Data.**

- **MARIE's example bodies.** The human body models and basis meshes shipped with
  MARIE are not copied, because their licences are unknown or restricted. Tests
  build their spheres and simple geometries in code.
- **IXI-derived models.** Body models and VOP files from IXI data are CC BY-SA,
  kept separate from the code, with the licence written into each file's
  metadata.
- **Tissue properties.** Permittivity and conductivity are evaluated in code from
  the published Gabriel et al. 1996 Cole–Cole parameters. Mass densities are
  ICRU Report 44 values as tabulated by NIST, credited to NIST. The IT'IS tissue
  database is not bundled.
- **Coil models.** No vendor or commercial coil data enters the repository.

**Implementation.**

- **torch.** It carries the numerical work on CPU or CUDA, with complex128 as the
  working precision.
- **Other dependencies.** numpy is used for file input and output only, where VOP
  files are `.npz`. scipy is a test and optional dependency: special functions
  for the Mie reference, and the global optimiser for co-simulation from
  milestone 3. No numba, no CuPy.
- **C++ kernels.** Work over voxels, mesh elements or quadrature points that
  torch cannot batch runs in C++ in the pybind11 extension, on CPU buffers, and
  its result moves to the caller's device. Work that torch can batch stays in
  torch, where one code path runs on either device, and moves to C++ when a
  profile shows it dominates, as **Loops** above says. Milestone 1's own profile names the
  first candidate: assembling the body kernel takes tens of seconds, nearly all
  of it the six-dimensional volume-volume rule at `Np_1D_medium_V`, whose cost
  is set by that order and by the 512 offsets it covers rather than by the size
  of the grid.
- **Arrays and units.** Arrays are C-ordered with the batch dimension first:
  (ports, …) and (N, Nc, Nc). Units are SI, and frequencies are in Hz.
- **Provenance.** Every ported function names its MARIE source file in its
  docstring.

**Tests.**

- Every test runs on a CPU-only machine on coarse grids. The CUDA leg of a test
  skips when no device is present.
- Mesh and grid refinement studies carry the `slow` marker.
- Build, lint and test commands are those in `AGENTS.md`, and a change is
  reported complete only with their output.

## Later stages

These stages follow the solver. They are here so the solver's interfaces serve
them; the first task is milestone 1.

**Head models.**

- **Source.** IXI T1 and T2 volumes, downloaded with torchio's `ixi()`, which
  fetches the raw tarballs without preprocessing.
- **Bone.** From a pseudo-CT predicted by mr-to-pct (Apache-2.0 code, CC BY 4.0
  weights).
- **Brain.** From FastSurfer (Apache-2.0 code and weights).
- **Skin, fat, muscle and the remaining tissues.** From intensity rules, or from
  SimNIBS charm run as an external tool: charm is GPL-3.0, but its output label
  maps are not covered by that licence.
- **Labels.** The label merge is written anew, using PHASE (Apache-2.0) as a
  reference.
- **Properties.** Each label receives permittivity and conductivity from Gabriel
  et al. 1996 at the coil's frequency, and a mass density from NIST's ICRU Report
  44 table. Tissues NIST does not list take a recorded substitute: soft tissue
  for skin, dura and mucosa, water for CSF and vitreous humour, cortical bone for
  cancellous bone.
- **Open check.** Whether IXI volumes keep the face is unverified, so it is
  checked before the pipeline relies on it.

**Q matrices and averaging.**

- **Local Q.** Per voxel, from each channel's electric field, in the convention
  of the output contract.
- **10 g averaging.** Follows the IEC/IEEE 62704-1 algorithm of the Apache-2.0
  reference implementation
  (umbertozanovello/IEC-IEEE-62704-1-spatial-average-SAR). Averaging a Q matrix
  over a cube is linear. The two steps that depend on SAR both assign a voxel
  the largest average over a set of cubes:
  - step 1: the valid cubes enclosing the voxel;
  - step 2: up to six face-centred cubes.
- **Pooling.** Pooling every valid cube's matrix with every step-2 cube's matrix
  therefore gives the standard's peak spatial-average SAR for any drive.

**VOP compression.**

- **Pool.** The averaged matrices of all body models, compressed together, so the
  VOPs bound local SAR across the population.
- **Algorithm.** Eichfelder and Gebhardt (MRM 2011, doi 10.1002/mrm.22927),
  written from the paper. The open implementations found are copyleft or
  unlicensed, so none is used.
- **Margin.** The compression margin is a parameter recorded in the VOP file.

**Head-average matrices.** One per body model, integrated over that model's head
mass, in the stacked layout of the output contract.
