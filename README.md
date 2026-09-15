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
field strength you name. Both are built in code here; a real coil comes from a
GMSH file and a JSON list of its lumped elements, through
`SurfaceMesh.read_gmsh22` and `read_lumped_elements`.

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

The solve runs on either device: build the body and the coil with
`device="cuda"` and everything downstream follows.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).
