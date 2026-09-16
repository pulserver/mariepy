"""MARIE's multilayer sphere under a plane wave, against the layered Mie series.

usage, from the repository root:
    python examples/multilayer_mie.py <marie-tools>/data/bodies/Scatterers/2mm/Multilayer_Sphere.mat [linear] [cpu|cuda]
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

path = sys.argv[1]
linear = len(sys.argv) > 2 and sys.argv[2] == "linear"
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
medium = Medium(3.0063)
body = VoxelBody.read_marie(path, device=device)
radii = [0.06, 0.08, 0.10, 0.12]
eps = [60.0, 20.0, 40.0, 30.0]
sigma = [0.45, 0.25, 0.35, 0.30]
omega_eps0 = medium.angular_frequency * medium.permittivity
indices = [np.conj(np.sqrt(complex(e, -s / omega_eps0))) for e, s in zip(eps, sigma)]

t0 = time.time()
op = BodyOperator.build(body, medium, linear=linear)
t1 = time.time()
inc = plane_wave(body, medium, linear=linear)
sol = solve_body(op, inc, tol=1e-6, maxit=400)
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
    lerr = np.linalg.norm(got[sel & away] - ref[sel & away]) / np.linalg.norm(ref[sel & away])
    print(f"layer {l}: absorbed solver/Mie(voxel centres) {p_solver/p_mie:.3f}  field error away from interfaces {lerr:.4f}")
