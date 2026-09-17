"""MARIE's multilayer sphere under a plane wave, against the layered Mie series.

usage, from the repository root:
    python examples/multilayer_mie.py <marie-tools>/data/bodies/Scatterers/2mm/Multilayer_Sphere.mat [linear] [mixed] [cpu|cuda]
    python examples/multilayer_mie.py --pitch 0.008 [linear] [mixed] [cpu|cuda]

``linear`` gives the body the piecewise-linear basis, twelve unknowns per voxel
where the constant basis has three; ``mixed`` applies the operator in single
precision and finishes in double. MARIE's file is 2 mm, where the linear basis wants a
workstation's memory; ``--pitch`` builds the same sphere at a coarser pitch,
where both bases finish in a couple of minutes, and the comparison between
bases on one grid is the same question.
"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, ".")
from tests import mie
from mariepy import vie
from mariepy.body import VoxelBody
from mariepy.constants import Medium
from mariepy.incident import plane_wave
from mariepy.solver import BodyOperator, solve_body

radii = [0.06, 0.08, 0.10, 0.12]
eps = [60.0, 20.0, 40.0, 30.0]
sigma = [0.45, 0.25, 0.35, 0.30]


def built(pitch, device):
    """The same four-layer sphere, on a grid of the pitch asked for."""
    n = int(round(2 * radii[-1] / pitch)) + 3
    half = (n - 1) / 2
    axis = pitch * (torch.arange(n, dtype=torch.float64) - half)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    distance = torch.sqrt(x**2 + y**2 + z**2)
    permittivity = torch.ones((n, n, n), dtype=torch.float64)
    conductivity = torch.zeros((n, n, n), dtype=torch.float64)
    for inner, outer, e, s in zip([0.0] + radii[:-1], radii, eps, sigma):
        shell = (distance > inner) & (distance <= outer)
        permittivity[shell] = e
        conductivity[shell] = s
    return VoxelBody(
        permittivity=permittivity.to(device),
        conductivity=conductivity.to(device),
        mask=(distance <= radii[-1]).to(device),
        resolution=pitch,
        origin=(-half * pitch,) * 3,
    )


arguments = sys.argv[1:]
medium = Medium(3.0063)
if arguments[0] == "--pitch":
    source, arguments = float(arguments[1]), arguments[2:]
else:
    source, arguments = arguments[0], arguments[1:]
linear = "linear" in arguments
precision = "mixed" if "mixed" in arguments else "double"
device = "cuda" if "cuda" in arguments else "cpu"
if isinstance(source, float):
    body = built(source, device)
else:
    body = VoxelBody.read_marie(source, device=device)
omega_eps0 = medium.angular_frequency * medium.permittivity
indices = [np.conj(np.sqrt(complex(e, -s / omega_eps0))) for e, s in zip(eps, sigma)]

t0 = time.time()
op = BodyOperator.build(body, medium, linear=linear)
t1 = time.time()
inc = plane_wave(body, medium, linear=linear)
sol = solve_body(op, inc, tol=1e-6, maxit=400, precision=precision)
t2 = time.time()
total = op.total_field(sol.x, inc)
print(f"assembly {t1-t0:.0f}s solve {t2-t1:.0f}s iterations {len(sol.residuals)-1} converged {sol.converged}", flush=True)

centres = total[0::4] if linear else total
c = body.coordinates().cpu()
mask = body.mask.cpu()
pts = torch.stack([c[a][mask] for a in range(3)], 1).numpy()
got = torch.stack([centres[a].cpu()[mask] for a in range(3)], 1).numpy()
ref = mie.layered_internal_field(pts, radii, indices, medium.wavenumber)
r = np.linalg.norm(pts, axis=1)
res = body.resolution
away = np.min(np.abs(r[:, None] - np.array(radii)[None, :]), axis=1) > 1.5 * res
err = np.linalg.norm(got[away] - ref[away]) / np.linalg.norm(ref[away])
print(f"interior field error (>1.5 voxels from interfaces) {err:.4f}")
layer = np.searchsorted(radii, r)
cond = body.conductivity.cpu()[mask].numpy()
mass = vie.mass(total.shape[0], res)
weighted = torch.einsum("c,cxyz->xyz", mass, total.abs().cpu() ** 2)[mask].numpy()
for l in range(4):
    sel = layer == l
    p_solver = 0.5 * np.sum(cond[sel] * weighted[sel])
    p_mie = 0.5 * np.sum(cond[sel] * np.sum(np.abs(ref[sel]) ** 2, axis=1)) * res**3
    inside = sel & away
    # A layer thinner than the margin either side of its interfaces leaves no
    # voxel to compare, which a coarse pitch makes easy to reach.
    lerr = (
        f"{np.linalg.norm(got[inside] - ref[inside]) / np.linalg.norm(ref[inside]):.4f}"
        if inside.any()
        else "-, no voxel is that far from an interface"
    )
    print(f"layer {l}: absorbed solver/Mie(voxel centres) {p_solver/p_mie:.3f}  field error away from interfaces {lerr}")
