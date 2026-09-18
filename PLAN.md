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

**Order.** The four milestones run in order, and the SAR milestone follows them:
Q matrices, averaging, compression and head models, described under **SAR
milestone** below.

**Co-simulation source.** MARIE ships its co-simulation twice:
`src_physics/src_electronics/co_simulation/` and a copy under
`src_physics/src_electronics/src_electronics/`. The two share every function
name; the nested copy moves the cost functions onto two shared helpers,
`match_optim.m` and `preamplifier_match_optim.m`, and in `match.m` adds each
matching stage only to the ports whose network carries it, where the top-level
copy adds it to every port. The two agree when every port has the same matching
topology. Milestone 3 ports the nested copy alone.

**Shields.** A shield is a surface around the coil and the body. Its own matrix is
the coil's, `sie.assemble`; its coupling to the coil is assembled whole, where
MARIE compresses it by adaptive cross approximation, since a coil and its shield
together carry few enough unknowns for the dense block; its coupling to the body
is a tensor train per body unknown, built by the DMRG cross method ported into
`tt.py`. The port makes two changes to TT-Toolbox as MARIE carries it:
`maxvol2.m` never advances its iteration count, so its loop can run without
end, and here it is bounded; `reort.m` compares squared entries without
conjugating, which on complex data compares real parts, and here it compares
squared moduli. A shield may carry lumped elements and driven ports of its
own; its ports come first, as in MARIE's `rhs_assembly.m`, and its element
file is merged first for co-simulation. A simulation file that names a
shield and no coil is solved with the shield as the coil, through the
precorrected FFT rather than tensor trains.

**Wire coils.** `wire.py` ports MARIE's wire coil: its geometry, its own
matrix with the closed forms for a segment against itself, its lumped loads,
its port drive and its coupling kernels. The precorrected FFT takes a wire coil
as it takes a surface one, since the two differ only in where a basis function
sits, how wide it is and which kernel gives its field. It departs from MARIE
in three places:

- MARIE's open-wire branch of `ProcessLoops.m` assigns rows of mismatched
  size and cannot run. An open wire here carries a basis function at every
  interior node and none at its two ends, where the current vanishes; a port
  on an end segment is refused. A centre-fed half-wave dipole converges, in
  the segment count, to an input impedance above the infinitely thin dipole's
  73 + j42 ohms, as a wire of finite radius has.
- MARIE's wire coupling sources sample the falling half of each basis function
  with the rising ramp, so the current they couple to the body is not the one
  the wire's own matrix solves for. The torch kernel gives the falling half its
  falling ramp, and the oracle test compiles MARIE's sources with that one
  change; a second test shows the sources as shipped differ.
- A port or element on a loop's last segment spans that segment's two basis
  functions, the last and the loop's first; MARIE takes the next basis
  function by index, which there is the next loop's first.

A wire coil goes with a surface coil or inside a shield through its
interaction with a surface, assembled whole where MARIE compresses it by
adaptive cross approximation. The port does not follow MARIE's row assembly
(`assembly_wire_surf_ns_row.m`), which holds the triangle basis at one half on
both segments and gives both segments' charge the same sign; it integrates the
integrand both self-matrices share, with each basis as the self-matrices
define it. A wire loop and a surface loop reproduce Neumann's mutual
inductance.

**Co-simulation.** `cosim.py` ports MARIE's tuning, matching, decoupling and
preamplifier-decoupling searches and its calibrations, with the circuit
numerics in `circuit.py`. MARIE's index masks become a table of rows read once
from the element file. MARIE writes one master file per mix of port roles; the
port writes the idea they share once. The transmit side is every port that
transmits, with the receive-only ports detuned; the receive side is every port
that receives, with the transmit-only ports detuned. With one role throughout
nothing is detuned. The port departs from MARIE in these places:

- `calibration_tune_match_decouple.m` uses `SPs` without defining it, and
  MARIE's `Tx` search with optimisation stops there; the port defines it as
  the preamplifier variant does, from the tuned admittance.
- `toeplitz`, which the decoupling costs call, is not defined anywhere in
  `marie-tools`; `circuit.toeplitz` builds the matrix it names.
- `calibration_matching.m` rebuilds a lossless matching network from the
  scattering parameters alone. Power conservation fixes that network only up
  to a unitary, which on coupled ports can mix channels, and the map it gives
  ends at the wave into the coil while the solver's currents are per volt. The
  port computes the coil voltage per incident wave exactly, from each port's
  chain of matching elements (`circuit.coil_voltage`). On one port the two
  agree in magnitude.
- The preamplifier terminations add `1/R` to every entry of the other ports'
  admittance block, because MATLAB adds a scalar to a matrix entrywise; the
  port adds it to the diagonal, one resistor per port.
- `particleswarm` becomes scipy's differential evolution, with MARIE's swarm
  size, iteration limit, restarts and bound narrowing. Each search's first
  population holds the values the previous search handed it, so the joint
  search cannot end worse than the per-entity ones.
- Coupling-coefficient variables are numbered past every other variable,
  where MARIE's numbering can collide with a symmetry-offset one.
- Merging a wire coil's element file with a surface coil's, MARIE adds the
  wire's element count to each mutual inductor's mutual inductance, where it
  means the partner's number; the port moves the partner's number.
- Where roles mix, MARIE's four cost and calibration copies differ in ways
  the idea does not explain. The copy for all three roles detunes the
  receive-only ports on its receive side, where it means the transmit-only
  ones; the port detunes the other side's own ports throughout. The copy for
  transmit-only with transmit-and-receive ports weighs no coupling, and the
  port keeps that. MARIE adds a side's terms to the joint cost when that side's
  own search found values; the port adds them when the side has ports.

**Field bases and performance maps.** `basis.py` ports MARIE's basis and MRGF
paths. The incident-field basis is spanned by a support surface, or by a shell
of voxel currents that hugs the body or is spherical; then come its
interpolation voxels, the body solved once per basis field, the ultimate
intrinsic SNR and transmit efficiency, and a coil solved through the basis with
its coupling integrated at the interpolation voxels only. `metrics.py` ports
the SNR, transmit-efficiency and g-factor maps, and `plot.py` MARIE's figures,
with matplotlib as an optional dependency. The port departs from MARIE in five
places:

- MARIE keeps the incident basis in tested form, the Gram matrix applied, and
  solves the body with it as if it were a field. For the piecewise-constant
  basis the Gram matrix is a scalar and the two agree; for the piecewise-linear
  one the tested vectors do not span the fields a coil puts on the body. The
  port keeps the basis in field coefficients and carries the Gram matrix where
  the coupling needs it.
- MARIE's noise covariance weighs the body by `integral sigma |E|^2`, twice the
  power it dissipates, and the conductor and lumped elements by half that
  scale and on the diagonal only. The port weighs all three alike, with their
  channel-to-channel terms.
- MARIE's randomised range finder (`rSVD_Q.m`) draws blocks until the
  sample's spectrum drops below its tolerance, and on an operator whose
  spectrum never does it never stops; the port stops once the sample has as
  many columns as the operator has columns or rows.
- MARIE's spherical shell (`geo_spherical_basis.m`) reads its inputs from a
  variable it never defines, centres its padded grid at the shell's thickness
  rather than at the body, and sizes the enclosing sphere from the body's
  position. The port pads the grid about the body and takes the grid's
  half-diagonal, MARIE's value for a body centred at the origin.
- MARIE's saved basis files are read with h5py and kept in MARIE's tested
  form, which the reduced coil solve then follows as MARIE does, piecewise-
  linear inconsistency included; bases built here are saved with
  `FieldBasis.save`.

On a coarse body the ultimate SNR does not settle as basis fields are added:
the discrete electric field under-resolves the high-order fields, which then
look nearly noiseless. MARIE maps it over a logarithmic run of mode counts for
that reason, and `ultimate_maps` takes the count.

**Coil-implicit solve.** `implicit.py` ports MARIE 2.0's perturbation solve
(GPL-3.0-or-later; Guryev et al., IEEE TBME 70 (2023) 1575), which eliminates
the coil currents and leaves an equation in the body current alone,

`(Zbb + Zbc Zc^-1 Zbc^T) Jb = Zbc Zc^-1 F`.

The second term is the coil's perturbation of the body. It depends on the coil,
the frequency and the grid, never on the tissue, so it is compressed once over
the region of a grid a body may occupy and serves every body within it: a
population on one grid pays for the coil once. The unknown is then the body's
alone, and the coil rows, whose preconditioned single-precision rounding is what
stops the coupled system from gaining by mixed precision, are gone from the
iteration.

The port departs from MARIE 2.0 in three places:

- The coupling is approximated by MARIE 2.0's own cross approximation
  (`cross_cheb_2d.m`, with `sample_rows.m`, `sample_cols.m` and `maxvol.m`):
  rows and columns of `Zbc` are integrated directly, at one kernel evaluation
  per entry and no convolution, sampled columns span a body-side basis and
  sampled rows a coil-side one, and the core is fitted on the sampled submatrix
  through their pseudo-inverses. Each round takes the columns where the
  coil-side basis has the largest volume and the rows of a Chebyshev grid one
  step finer, and it stops when the core's singular values settle to `tol`.
  Where MARIE 2.0 subsamples the rows of its chosen cells at random to bound
  their number, every component of a chosen cell is kept here: the rows only
  have to span the row space, and keeping them whole leaves the submatrix in
  the order the body numbers its own unknowns.

  The pivoting is what makes the sampling affordable, and is not incidental: a
  random sample of rows estimates the Gram of `Zbc` with an error that falls
  only as its square root, and measured on a head at 6 mm, a basis from 1024
  cells of 40316 leaves 1.4e-1 of the coupling's action, 4096 leaves 2.8e-2,
  and every row leaves 3.0e-3, whichever way the cells are drawn — so a random
  sample reaches `tol` only by taking nearly the whole region. Holding
  rank-many grid-sized columns while it pivots is what that costs, and is why
  MARIE 2.0 keeps factors the size of the grid.

  The perturbation is then the small matrix `C Zc^-1 C^T` between the
  coupling's factors, truncated at `tol`, which leaves the factors over the
  whole region, in complex64: a body selects the rows of the voxels it
  occupies, and no product on the extended grid is taken per body. Their
  right-hand side stays in complex128, where the solve starts from it.
- The coupling the perturbation holds is the one the quadrature gives, where
  the coupled operator projects its far interactions onto the grid, so the two
  differ by that projection — as they do already in the coil matrix below.
  What the approximation leaves of the operator's own coupling is measured on
  every build, on random coil currents, since the perturbation inherits it.
- The region must keep its distance from the conductors. Tissue beside a
  conductor sees the near field of each of its edges, which no low-rank
  perturbation holds: a head mask dilated into the coil kept more than 480
  singular values above the tolerance. The region defaults to the grid's own
  mask, takes the union of a population's masks, and `tissue_region` gives a
  clearance envelope.
- The coil is eliminated with its own method-of-moments matrix `Zc` by default.
  Passing the coupled operator's coil block instead makes the solve reproduce
  `solver.solve_ports` exactly, which is how the two truncations are told apart
  from the elimination.

**Compiled kernels.** MARIE's C++ sources are bound with pybind11 into the
package's single `_ext` module, following the package template and pypulseqpp.

- **Coupling kernels.** For a surface coil, MARIE ships 24 coupling sources that
  differ only in field component, piecewise-linear basis term and operator (N or
  K). They become one N kernel and one K kernel taking the component and basis
  term as arguments, written twice: in C++ in `_ext`, which runs on CPU, and in
  torch over the `N` and `K` kernels the body operator already uses, which runs
  on CUDA. Each is checked against the other and against the 24 sources, kept in
  `tests/marie/`. The torch form is assembled from kernels the Mie series
  validated rather than transcribed, so the check between the two is a check
  between independent formulations.
  The 24 wire-coil sources, kept in `tests/marie/wire/`, have the same
  structure and are checked the same way against the torch wire kernel.
- **Singular integrals.** The DIRECTFN sources (`direct_ws_*_rwg` for the coil,
  `solve_ea`, `solve_st`, `solve_va` and their headers for the body) carry an
  LGPL notice. They build as a separate extension module with that notice kept,
  never inside `_ext`.
- **Threading.** OpenMP loops become standard C++ threads partitioning
  independent work, as in the template.

**MATLAB-side code.**

- **Data licences.** The same terms as code: no MathWorks, GPL, AGPL or
  unlicensed data, and no non-commercial clause. The IT'IS tissue database is
  CC BY-NC and therefore excluded; tissue properties come from Gabriel et al.
  1996, as the SAR milestone records.
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
for a 5 cm sphere at brain-like permittivity the interior field's error falls
from 31% to 16% to 7.5% as the voxel halves from 10 mm to 2.5 mm, while at a
permittivity of 2 ten voxels across the radius already suffice. MARIE's own example runs the piecewise-linear basis
at 2 mm for this reason. The Mie criterion is therefore measured over the
interior rather than over every voxel, and the refinement legs carry the grids
they were measured on.

**Milestone 2.**

- The piecewise-linear basis meets the Mie criterion, and its error is reported
  against the piecewise-constant basis on the same grids.
- The coupled solve in the piecewise-linear basis meets milestone 1's
  reciprocity, power-balance and direct-integration criteria.
- Each basis's absorbed power is measured against the Mie series for a lossy
  sphere at brain-like permittivity, split between the interior and the shell
  of boundary voxels, since the staircase error sits in that shell. Peak 10 g
  SAR is compared in the same way once 10 g averaging exists, in the SAR
  milestone.
- Shielded coils meet the reciprocity and mesh-convergence criteria.
- Tensor-train coupling agrees with the fully assembled coupling on a small
  case, within `tol_TT`.

**Milestone 3.**

- Co-simulation reproduces closed-form results for simple lumped networks: a
  series RLC, and an L-section matching a known load.
- The loaded port matrix and the fields at the matching network's input agree
  with CoSimPy's `RF_Coil` connections for the same element values. CoSimPy
  evaluates a given circuit in S-parameters and cannot tune one, so it checks
  the ported circuit algebra rather than replacing it. The check runs when
  CoSimPy enters, with the VOP compression stage; until then the circuit
  algebra is checked by power conservation through lossless networks and by
  the coil voltage against its closed form.
- On a coil, the tuned and matched reflection at the Larmor frequency meets the
  target set in the coil's element file.

**Milestone 4.**

- A solve in a precomputed field basis agrees with the direct solve for the same
  coil and body, within the basis tolerance.
- SNR and transmit-efficiency maps agree with their definitions evaluated
  directly from the fields.
- The coil-implicit solve reproduces the coupled solve's body current when the
  coil is eliminated with the coupled operator's own coil block and the
  perturbation is truncated tightly. With the coil's method-of-moments matrix
  and the default tolerance the two answers differ, and each case reports that
  difference in the port matrix and in the field, separated between the two
  causes: the elimination matrix and the truncation.
- A body smaller than the region gets the same currents from the restricted
  coupling as from one assembled for it alone.

**Q matrices, averaging and VOPs.**

- 10 g averaging agrees with the IEC/IEEE 62704-1 reference implementation at
  random drives.
- For random drives, the largest VOP SAR is at least the largest averaged voxel
  SAR and at most that plus the compression margin.
- On fields of rank three or less, the compression reproduces CoSimPy's
  `compVOP` output, conjugated.
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

**Code licence.** mariepy is GPL-3.0-or-later. It is a standalone program,
and what other projects take from it is its output: a VOP file, or a field map,
which the licence of the program does not reach. pypulseqpp reads the file and
never imports mariepy.

- **Code under a licence compatible with GPL-3.0** — MIT, BSD, Apache-2.0,
  LGPL, and GPL-2.0-or-later or GPL-3.0-or-later — may be ported with its
  copyright notice kept, and each ported source is listed in
  `THIRD_PARTY.md`. This covers MARIE 3.0 itself, TT-Toolbox's `dmrg_cross`,
  and MARIE 2.0 (GPL-3.0-or-later), whose coil-implicit solve the package
  takes up.
- **MARIE files kept verbatim as a test oracle** are listed the same way. MARIE
  3.0 is MIT, so a source a check compares against is kept rather than described:
  a comparison a developer must set up is a comparison that does not run. Such
  files live under `tests/`, never under `src/`, so nothing in them reaches the
  wheel or links into `_ext`. The 24 coupling sources in `tests/marie/` are the
  first of them.
- **CoSimPy** (MIT, <https://github.com/umbertozanovello/CoSimPy>) enters with
  the VOP compression stage, as the source of the compression core and as a
  test dependency, and is listed in `THIRD_PARTY.md` then.
- **MARIE's LGPL files** keep their own notice:
  - the DIRECTFN singular integrals, built as a separate extension module that
    carries it;
  - the Dunavant triangle quadrature files, whose rules are taken from Dunavant
    (1985) and checked by polynomial exactness, so the LGPL files themselves are
    not carried.
- **Excluded:** GPL-2.0-only, AGPL, non-commercial or unlicensed code, and model
  weights under such terms. MathWorks' `iterapp.m` and `iterchk.m` are excluded
  too, and so are two quadrature files MARIE ships without a licence:
  `gauss_1d.m`, Burkardt's `LEGENDRE_SET`, and `getLebedevSphere.m`, a
  translation of Laikov's routines. Gauss–Legendre nodes and weights come from
  the Golub–Welsch eigenvalue problem instead, and the 26 Lebedev directions —
  the only part of that rule the precorrected FFT projection uses, since it
  discards the weights — are the three octahedral orbits, generated in code.

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
  working precision: the operators are assembled in it, every Krylov basis is
  kept in it, and every result is returned in it. A solve may take its
  matrix-vector products in complex64 (`precision="mixed"`), which halves what
  the products move through memory. Their rounding then caps the residual the
  iteration can reach, so that iteration stops at the tolerance or a margin
  above the rounding measured through the preconditioner on a random vector,
  whichever is larger, and a complex128 solve started from its iterate reaches
  the tolerance. The two solves' residual histories are reported as one.
- **Other dependencies.** numpy is used for file input and output only, where VOP
  files are `.npz`. scipy is a test and optional dependency: special functions
  for the Mie reference, reading MARIE's MATLAB body files, and the global
  optimiser for co-simulation from milestone 3. cosimpy is a test dependency
  from the VOP compression stage. No numba, no CuPy.
- **C++ kernels.** Work over voxels, mesh elements or quadrature points that
  torch cannot batch runs in C++ in the pybind11 extension, on CPU buffers, and
  its result moves to the caller's device. Work that torch can batch stays in
  torch, where one code path runs on either device, and moves to C++ when a
  profile shows it dominates, as **Loops** above says. Milestone 1's profile
  named the first: the six-dimensional volume-volume rule at `Np_1D_medium_V`.
  The kernel sees only the separation of two points, so the port departs from
  MARIE's `q⁶`-point product rule and integrates over the separation, weighted
  by the two cells' overlap, with `(2q + 2)³` points; at every order tested
  its error against a converged product rule is the smaller. It runs in C++
  on CPU and in torch otherwise, each checked against the other. The DIRECTFN
  face integrals of touching cells release the GIL and run on threads. A
  body kernel on a 60³ grid then assembles in about ten seconds for the
  piecewise-constant basis and half a minute for the piecewise-linear one,
  a cost set by the near offsets rather than by the grid. The solve's profile
  named the second: the Fourier-domain multiply by the body symbols, which
  torch takes with every symbol expanded at once and therefore with several
  copies of the extended grid in memory. In C++ a symbol stays in its Tucker
  form, contracted along each z-line as it is needed, and the multiply runs in
  place on the transformed current over threads, so the product holds one
  buffer of the extended grid and one component of it. It runs in complex128
  and complex64; torch takes the product on CUDA, and on a grid small enough
  that expanding the symbols costs less than starting the threads.
- **Circulant embedding.** A Toeplitz block of `n` offsets sits inside any
  circulant of `2n - 1` rows or more, where MARIE always takes `2n`. The port
  takes the shortest length with no prime factor above seven, since an FFT of a
  length carrying a large prime factor costs several times the one it needs.
  The symbol, not the caller, names the length, so the choice is local to
  `tucker.transform_length`.
- **Body solve.** The body operator is scaled by its own Galerkin mass term, so
  what GMRES sees is the identity minus a compact term and no preconditioner is
  applied. MARIE's `prec_vie.m` inverts that mass term, which is what the
  coupled system still needs, since there the mass term stands in the matrix.
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

## SAR milestone

This milestone follows milestone 4, as the **Order** paragraph of the MARIE port
section says. It is described here so the solver's interfaces serve it.

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
- **Where the numbers live.** `tissue.py` carries the four-Cole-Cole model and
  reads the parameters from a table beside the label volume, rather than
  bundling them. A measured parameter the package cannot check against its
  source would be a silent error in every SAR number that follows, and the
  pipeline writes the table it names its own labels against anyway.
  `tissue.read_table` states the columns and the paper's units.
- **Which measurements.** Gabriel et al. 1996, and not the IT'IS database, whose
  licence is CC BY-NC and so outside this package's terms. The two differ: at
  127.74 MHz MARIE's own head model, which carries IT'IS values, has grey matter
  at 65.4 and 0.518 S/m where Gabriel's fit gives 73.6 and 0.587. Muscle, white
  matter and cortical bone agree between them to the three figures MARIE stores,
  so those are a check on the model rather than on the table.
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
  ported from CoSimPy's `EM_Field.compVOP` (MIT). `compVOP` builds each point's
  matrix from one field, so it holds rank three at most and cannot take the
  pooled averaged matrices, and its compression steps are nested inside it. The
  port is a function of a stack of Hermitian matrices.
- **Convention.** `compVOP` returns the complex conjugates of the matrices that
  bound `vᴴQv` in the convention of the output contract, which CoSimPy's own
  `compQMatrix` follows. The port writes that convention, and the check against
  `compVOP` conjugates its output; conjugating the output and conjugating the
  input field give the same matrices, since the algorithm uses only eigenvalues
  and their order.
- **Margin.** The compression margin is a parameter recorded in the VOP file.

**Head-average matrices.** One per body model, integrated over that model's head
mass, in the stacked layout of the output contract.
