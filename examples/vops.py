"""Write the virtual observation points of a coil and a body.

usage, from the repository root:
    python examples/vops.py --sphere --out vops.npz
    python examples/vops.py --case <input.json> --data <marie-tools>/data \
        --labels labels.npy --table tissue.csv --out vops.npz

``--sphere`` builds a loop coil around a uniform ball and runs the whole chain
in a minute, which is enough to see the file it writes. A real head takes its
tissue labels and their properties from ``--labels`` and ``--table``, which
``mariepy.tissue`` reads; without them the case's own permittivity and
conductivity are used and the density is whatever ``--density`` says.

The matrices follow PLAN.md's output contract: local SAR for a drive ``v`` at
peak amplitude is ``v^H Q v`` in W/kg.
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from mariepy import __version__, averaging, sar, sie, vop
from mariepy.body import VoxelBody
from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh
from mariepy.solver import solve

parser = argparse.ArgumentParser()
parser.add_argument("--sphere", action="store_true", help="a loop coil around a ball")
parser.add_argument("--case", help="a MARIE simulation file")
parser.add_argument("--data", help="marie-tools data folder, with --case")
parser.add_argument("--labels", help="a .npy volume of tissue labels")
parser.add_argument("--table", help="the tissue table those labels are named against")
parser.add_argument("--density", default="1000", help="kg/m^3, a number or a .npy grid")
parser.add_argument("--target", type=float, default=10.0, help="averaging mass in grams")
parser.add_argument("--margin", type=float, default=0.05, help="VOP overestimation")
parser.add_argument("--device", default="cpu")
parser.add_argument("--out", required=True)
args = parser.parse_args()

t0 = time.time()
if args.sphere:
    medium = Medium(3.0)
    body = VoxelBody.sphere(0.06, 0.006, 52.0, 0.55, padding=2, device=args.device)
    ports = tuple(
        Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
        for tag in (1, 2)
    )
    coil = SurfaceCoil.build(
        SurfaceMesh.loop(radius=0.11, width=0.02, n_around=12, n_across=1, ports=2),
        ports,
    )
    channels = ["loop-a", "loop-b"]
    name, licence = "loop around a ball", "none, the body is analytic"
    drive_unit = "1 V across the port, unmatched"
    result = solve(body, coil, medium, far_order=2, medium_order=2, near_order=4)
else:
    from mariepy.inputs import read_case

    case = read_case(args.case, data=args.data, device=args.device)
    medium, coil = case.medium, case.coil
    body = case.body
    if args.labels and args.table:
        from mariepy import tissue

        labels = torch.from_numpy(np.load(args.labels)).to(args.device)
        built = tissue.build(
            labels, tissue.read_table(args.table), medium, body.resolution,
            origin=body.origin,
        )
        body, density_grid = built.body, built.density
    channels = [f"port{index + 1}" for index in range(coil.n_driven)]
    name, licence = args.case, "set by the body model"
    drive_unit = "1 V across the port, unmatched"
    result = solve(body, coil, medium, linear=case.linear, shield=case.shield)

print(
    f"solved in {time.time() - t0:.0f}s: {body.n_voxels} voxels, "
    f"{len(channels)} channels, iterations {result.ports.iterations}",
    flush=True,
)

if args.sphere or not (args.labels and args.table):
    if args.density.endswith(".npy"):
        density_grid = torch.from_numpy(np.load(args.density)).to(body.device)
    else:
        density_grid = torch.full_like(
            body.conductivity, float(args.density), dtype=torch.float64
        )
        print(f"no density given, taking {args.density} kg/m^3 everywhere", flush=True)

mass = torch.where(body.mask, density_grid * body.resolution**3, 0.0)
target = args.target * 1e-3

t1 = time.time()
local = sar.local_matrices(
    result.fields.electric, body.conductivity, density_grid, body.mask
)
pool = averaging.cube_pool(mass, body.mask, target)
averaged = averaging.averaged_matrices(pool, local, mass, body.mask)
t2 = time.time()
print(
    f"{len(pool)} averaging cubes over {args.target:g} g in {t2 - t1:.0f}s; "
    f"largest eigenvalue {float(torch.linalg.eigvalsh(averaged)[:, -1].max()):.4g} W/kg",
    flush=True,
)

points, cluster = vop.compress(averaged, args.margin)
whole = sar.average(local, sar.voxel_mass(density_grid, body.resolution, body.mask))
print(
    f"{points.shape[0]} virtual observation points from {len(pool)} cubes "
    f"at a margin of {args.margin:g} in {time.time() - t2:.0f}s",
    flush=True,
)

vop.write(
    args.out,
    points,
    whole[None],
    coil=name,
    frequency_hz=medium.frequency,
    drive_unit=drive_unit,
    channels=channels,
    averaging=f"{args.target:g} g, IEC/IEEE 62704-1",
    bodies=[name],
    compression_margin=args.margin,
    data_licence=licence,
)
back = vop.read(args.out)
generator = torch.Generator().manual_seed(0)
drive = torch.randn(len(channels), dtype=torch.complex128, generator=generator)
print(
    f"wrote {args.out} with mariepy {__version__}; on one drive the points bound "
    f"{float(sar.peak(back.vops, drive)):.4g} W/kg against the cubes' "
    f"{float(sar.peak(averaged, drive)):.4g} W/kg",
    flush=True,
)
print(f"total {time.time() - t0:.0f}s")
