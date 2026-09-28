"""Solve BrainWeb's normal brain in a body coil and three head arrays, and write their maps and VOPs.

usage, from the repository root:
    python examples/brainweb.py --out fields
    python examples/brainweb.py --out fields --resolution 10 --loops 8 12 16

The head is BrainWeb's normal brain (Collins et al., IEEE Trans Med Imaging
17:463, 1998), whose fuzzy model gives every 1 mm voxel the fraction each of
its ten classes fills; brainweb-dl downloads it on first use. The fractions are
averaged over cubes ``--resolution`` millimetres wide and mixed into a body with
the Gabriel properties ``brainweb_tissues.csv`` gives each class, where CSF,
grey and white matter, fat, muscle, dry skin, cortical bone and dura carry the
parameters of Gabriel, Lau and Gabriel (Phys. Med. Biol. 41:2271, 1996) as the
IFAC implementation of that model tabulates them. Glial matter is taken as grey
matter and connective tissue as dura. Mass densities are those of NIST's table
of ICRU Report 44 tissues, with the substitutes PLAN.md records: water for
CSF, soft tissue for skin and dura.

The frame is the scanner's for a subject lying head first and supine, with
BrainWeb's MNI origin, the anterior commissure, at the isocentre: x points to
the subject's left, y posterior and z superior, in metres.

Four coils are written, each as ``<coil>.npz`` of :mod:`mariepy.maps`, and the
two that transmit also as ``<coil>_vops.npz`` of :mod:`mariepy.vop`:

- ``body``: an infinitely long quadrature birdcage around the head, its two
  linear modes as two channels, in the piecewise-linear basis;
- ``head8``: eight loops on an elliptic cylinder around the head, a transmit
  array, in the piecewise-linear basis;
- ``head32`` and ``head48``: receive arrays of loops on a helmet, in the
  piecewise-constant basis, whose magnetic field agrees with the linear one's.

A loop's unit drive is one ampere into its port with every other port of its
array open, the preamplifier-decoupled idealisation of an array. A birdcage
mode's is the mode whose flux density on the axis, in free space, is 1 uT.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from mariepy import averaging, fields, incident, maps, sar, tissue, vop
from mariepy.constants import Medium
from mariepy.solver import solve, solve_incident
from mariepy.wire import WireCoil

FRAME = (
    "scanner frame of a head-first supine subject: x to the subject's left, "
    "y posterior, z superior, in metres; origin at BrainWeb's MNI origin"
)
BODY = "BrainWeb normal brain, fuzzy model"
LICENCE = "BrainWeb's terms of use (https://brainweb.bic.mni.mcgill.ca/)"
LOOP_DRIVE = "1 A into the channel's port, every other port of the coil open"
MODE_DRIVE = "the birdcage mode whose flux density on its axis in free space is 1 uT"

# MNI coordinates, in mm, of the fuzzy model's first voxel; its axes run
# z, y, x, x fastest.
FIRST_VOXEL_MM = (-72.0, -126.0, -90.0)

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True, help="directory to write")
parser.add_argument("--field", type=float, default=3.0, help="static field in T")
parser.add_argument("--resolution", type=int, default=5, help="voxel pitch in mm")
parser.add_argument(
    "--loops",
    type=int,
    nargs=3,
    default=(8, 32, 48),
    metavar=("TRANSMIT", "RECEIVE", "RECEIVE"),
    help="loops of the transmit array and of the two receive arrays",
)
parser.add_argument("--coils", nargs="*", help="coils to solve; all by default")
parser.add_argument("--tol", type=float, default=1e-5, help="solver residual")
parser.add_argument("--margin", type=float, default=0.05, help="VOP overestimation")
parser.add_argument("--brainweb-dir", help="brainweb-dl's cache")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()


def say(text):
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def head(step: int):
    """Return the head's fractions in the scanner frame at ``step`` mm, and their origin."""
    import brainweb_dl
    from scipy import ndimage

    fine = np.asarray(
        brainweb_dl.get_mri(0, "fuzzy", brainweb_dir=args.brainweb_dir),
        dtype=np.float64,
    )
    # Voxels outside the head's own connected region are the model's border
    # artefacts; they are made background.
    labelled, _ = ndimage.label(fine[..., 0] < 0.5)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    stray = (labelled != 0) & (labelled != sizes.argmax())
    fine[stray] = 0.0
    fine[stray, 0] = 1.0
    # (z, y, x) in MNI, x to the right and y anterior, to (x, y, z) in the
    # scanner frame, x to the left and y posterior.
    scanner = np.ascontiguousarray(fine.transpose(2, 1, 0, 3)[::-1, ::-1])
    first = np.array(
        [
            -(FIRST_VOXEL_MM[2] + fine.shape[2] - 1),
            -(FIRST_VOXEL_MM[1] + fine.shape[1] - 1),
            FIRST_VOXEL_MM[0],
        ]
    )
    coarse = tissue.coarsen(torch.from_numpy(scanner), step)
    origin = 1e-3 * (first + 0.5 * (step - 1))
    return coarse.to(args.device), tuple(float(value) for value in origin)


def outside(body, points: torch.Tensor) -> bool:
    """Whether no point falls in a voxel of the body."""
    index = torch.round(
        (points.cpu() - torch.tensor(body.origin)) / body.resolution
    ).long()
    shape = torch.tensor(body.shape)
    inside = ((index >= 0) & (index < shape)).all(dim=1)
    index = index[inside]
    return not bool(body.mask.cpu()[index[:, 0], index[:, 1], index[:, 2]].any())


def extent(body, below: float):
    """Return the centre and semi-axes of the head's box above z = ``below``."""
    points = body.coordinates()[:, body.mask].T.cpu()
    points = points[points[:, 2] >= below]
    low, high = points.min(dim=0).values, points.max(dim=0).values
    return (low + high) / 2, (high - low) / 2


def cylinder_loops(body, count: int, clearance: float):
    """Loops around an elliptic cylinder about the head, at the height of the brain."""
    centre, half = extent(body, 0.0)
    semi = half[:2] + clearance
    angle = 2 * math.pi * torch.arange(count, dtype=torch.float64) / count
    x = centre[0] + semi[0] * torch.cos(angle)
    y = centre[1] + semi[1] * torch.sin(angle)
    z = torch.full_like(x, 0.02)
    normals = torch.stack(
        [torch.cos(angle) / semi[0], torch.sin(angle) / semi[1], torch.zeros_like(x)],
        dim=1,
    )
    spacing = 2 * math.pi * float(semi.mean()) / count
    return torch.stack([x, y, z], dim=1), normals, 0.45 * spacing


def helmet_loops(body, count: int, clearance: float):
    """Loops spread evenly over a helmet that follows the head.

    The loop centres lie along directions spread over the cap from the vertex
    to 100 degrees, as a Fibonacci spiral spreads them, from a centre level
    with the origin, each ``clearance`` past the farthest head voxel within 15
    degrees of its direction; each loop faces its direction, and its radius is
    half the median distance between neighbouring centres.
    """
    centre, _ = extent(body, 0.0)
    centre = torch.tensor([float(centre[0]), float(centre[1]), 0.0], dtype=torch.float64)
    offsets = body.coordinates()[:, body.mask].T.cpu() - centre
    distances = torch.linalg.vector_norm(offsets, dim=1)
    reach = math.radians(100.0)
    index = torch.arange(count, dtype=torch.float64) + 0.5
    polar = torch.arccos(1 - (1 - math.cos(reach)) * index / count)
    azimuth = math.pi * (3 - math.sqrt(5)) * index
    directions = torch.stack(
        [
            torch.sin(polar) * torch.cos(azimuth),
            torch.sin(polar) * torch.sin(azimuth),
            torch.cos(polar),
        ],
        dim=1,
    )
    within = (offsets @ directions.T) >= math.cos(math.radians(15.0)) * distances[:, None]
    farthest = torch.where(within, distances[:, None], 0.0).amax(dim=0)
    centres = centre + (farthest + clearance)[:, None] * directions
    spacing = torch.cdist(centres, centres).fill_diagonal_(math.inf).amin(dim=1)
    return centres, directions, 0.5 * float(spacing.median())


def loop_coil(body, centres, normals, radius):
    # A basis function spans two segments, and GMRES slows once one spans
    # more than three voxels.
    segments = max(12, math.ceil(2 * math.pi * radius / (1.5 * body.resolution)))
    coil = WireCoil.loops(centres, normals, radius, segments, device=args.device)
    if not outside(body, coil.points()):
        raise SystemExit("a loop passes through the head; widen its clearance")
    return coil


def write_maps(name, drives, body, medium, channels, drive_unit):
    plus, minus = fields.circular_components(medium, drives)
    maps.write(
        args.out / f"{name}.npz",
        plus,
        minus,
        body.mask,
        coil=name,
        channels=channels,
        frequency_hz=medium.frequency,
        drive_unit=drive_unit,
        origin=body.origin,
        resolution=body.resolution,
        frame=FRAME,
        bodies=[BODY],
        data_licence=LICENCE,
    )


def write_vops(name, drives, built, channels, drive_unit):
    body = built.body
    mass = torch.where(body.mask, built.density * body.resolution**3, 0.0)
    local = sar.local_matrices(drives.electric, body.conductivity, built.density, body.mask)
    pool = averaging.cube_pool(mass, body.mask, 10e-3)
    averaged = averaging.averaged_matrices(pool, local, mass, body.mask)
    points, _ = vop.compress(averaged, args.margin)
    whole = sar.average(local, sar.voxel_mass(built.density, body.resolution, body.mask))
    vop.write(
        args.out / f"{name}_vops.npz",
        points,
        whole[None],
        coil=name,
        frequency_hz=Medium(args.field).frequency,
        drive_unit=drive_unit,
        channels=channels,
        averaging="10 g, IEC/IEEE 62704-1",
        bodies=[BODY],
        compression_margin=args.margin,
        data_licence=LICENCE,
    )
    say(f"{name}: {points.shape[0]} VOPs from {len(pool)} cubes")


def solve_loops(name, coil, built, medium, linear, transmit):
    say(f"{name}: {coil.n_driven} loops, {coil.n_dof} wire unknowns, linear={linear}")
    result = solve(built.body, coil, medium, tol=args.tol, linear=linear)
    say(f"{name}: solved, iterations {result.ports.iterations}")
    drives = fields.combine(result.fields, result.impedance)
    channels = [f"{name}-{index + 1}" for index in range(coil.n_driven)]
    write_maps(name, drives, built.body, medium, channels, LOOP_DRIVE)
    if transmit:
        write_vops(name, drives, built, channels, LOOP_DRIVE)


def solve_birdcage(built, medium):
    say("body: two birdcage modes, linear")
    driving = [
        incident.birdcage(built.body, medium, angle=angle, linear=True)
        for angle in (0.0, math.pi / 2)
    ]
    drives = solve_incident(
        built.body,
        medium,
        torch.stack([electric for electric, _ in driving]),
        torch.stack([magnetic for _, magnetic in driving]),
        tol=args.tol,
        linear=True,
    )
    channels = ["body-I", "body-Q"]
    write_maps("body", drives, built.body, medium, channels, MODE_DRIVE)
    write_vops("body", drives, built, channels, MODE_DRIVE)


args.out.mkdir(parents=True, exist_ok=True)
start = time.time()
medium = Medium(args.field)
fractions, origin = head(args.resolution)
table = tissue.read_table(Path(__file__).with_name("brainweb_tissues.csv"))
built = tissue.mix(fractions, table, medium, 1e-3 * args.resolution, origin=origin)
say(
    f"head: {built.body.n_voxels} voxels of {args.resolution} mm on a "
    f"{built.body.shape} grid at {medium.frequency / 1e6:.2f} MHz"
)
wanted = set(args.coils or ("body", "head8", "head32", "head48"))
transmit, first, second = args.loops
clearance = 0.02
if "body" in wanted:
    solve_birdcage(built, medium)
if "head8" in wanted:
    coil = loop_coil(built.body, *cylinder_loops(built.body, transmit, clearance))
    solve_loops("head8", coil, built, medium, linear=True, transmit=True)
for name, count in (("head32", first), ("head48", second)):
    if name in wanted:
        coil = loop_coil(built.body, *helmet_loops(built.body, count, clearance))
        solve_loops(name, coil, built, medium, linear=False, transmit=False)
say(f"done in {time.time() - start:.0f} s")
