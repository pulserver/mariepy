# mariepy

Electromagnetic simulation of MRI transmit coils and virtual observation points for SAR, ported from MARIE 3.0.

[![Tests](https://github.com/pulserver/mariepy/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/mariepy/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/pulserver/mariepy/branch/main/graph/badge.svg)](https://codecov.io/gh/pulserver/mariepy)
[![PyPI](https://img.shields.io/pypi/v/mariepy.svg)](https://pypi.org/project/mariepy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Install

```bash
pip install mariepy
```

## Usage

Build a coil and a body, and solve them together at the Larmor frequency of the
field strength you name. Both are built in code here. A case laid out as MARIE
lays it out — a simulation file in `data/inputs/`, its body in `data/bodies/`,
its coil in `data/coils/coil_files/` — is read whole:

```python
from mariepy.inputs import read_case

case = read_case("data/inputs/my_case.json")
result = solve(
    case.body, case.coil, case.medium, linear=case.linear, shield=case.shield
)
```

`VoxelBody.read_marie`, `SurfaceMesh.read_gmsh22` and `read_lumped_elements`
read each file on its own.

```python
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import solve

medium = Medium(3.0)  # 1H at 3 T
body = VoxelBody.sphere(0.07, 0.005, 52.0, 0.55, padding=2)
coil = SurfaceCoil.build(
    SurfaceMesh.loop(radius=0.12, width=0.01, n_around=48, n_across=2),
    (Port(tag=1, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0),),
)

result = solve(body, coil, medium)
```

`result.impedance`, `result.admittance` and `result.scattering` are the port
matrices; `result.fields.electric` and `result.fields.magnetic` are each port's
field over the body grid, shaped `(n_ports, 3, n1, n2, n3)`. From there:

```python
from mariepy.fields import absorbed_power, circular_components

watts = absorbed_power(result.operator, result.fields)
b1_plus, b1_minus = circular_components(result.operator, result.fields)
```

A wire coil, `WireCoil.read_gmsh22` or `WireCoil.loop`, goes to `solve` in the
same place, and so does a `CombinedCoil` of a wire coil and a surface coil; a
simulation file naming a `WireFile`, with or without a `CoilFile`, reads into
one of them.

`linear=True` gives the body the piecewise-linear basis, twelve unknowns per
voxel, which carries the field's variation inside each voxel; the fields then
come back as those coefficients, and `fields.at_centres` gives their values at
the voxel centres.

The solve runs on either device: build the body and the coil with
`device="cuda"` and everything downstream follows.

### Tuning, matching and calibration

A coil's lumped elements are closed by co-simulation, as MARIE does it. Solve
the coil with its tunable elements opened into ports (`"TMD": 1` in the
simulation file), then search their values, the matching networks' and the
decoupling, and calibrate the fields to the wave driving each matched port:

```python
from mariepy.cosim import calibrate, co_simulate

closed = co_simulate(case.network, result.admittance, case.medium.angular_frequency)
electric = calibrate(result.fields.electric, closed.transmit)
```

With `"TMD": 0` the file's values are placed as they are. `closed.transmit`
has a column per transmitting port and `closed.receive` one per receiving
port, as the element file assigns `Tx`, `Rx` and `TxRx`; `closed.scattering`
is the transmitting ports' reflection and coupling; `cosim.sweep` gives the
matched ports across a band. The searches need scipy:
`pip install "mariepy[cosim]"`.

### Field bases, SNR and figures

A body's field basis is built once from a support surface around it
(`basis.surface_basis`) or from a shell of currents around it
(`basis.dipole_basis`), and any coil near that support is then solved through
it:

```python
from mariepy import basis, metrics, plot
from mariepy.solver import assemble_coil

incident = basis.surface_basis(case.body, case.basis_support, case.medium)
solved = basis.solve(incident, case.body, case.medium)
system = assemble_coil(case.coil, case.medium)
reduced = basis.solve_coil(case.coil, system, solved, case.body, case.medium)
ultimate_snr, ultimate_efficiency = basis.ultimate_maps(solved, case.body, case.medium)
```

`metrics.noise_covariance`, `metrics.snr`, `metrics.transmit_efficiency` and
`metrics.g_factor` map a coil's performance from its fields; `plot.geometry`,
`plot.coil_currents`, `plot.scattering`, `plot.sweep` and `plot.slices` draw
the model and the maps (`pip install "mariepy[plot]"`).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).
