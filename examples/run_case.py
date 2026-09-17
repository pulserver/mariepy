"""Run one MARIE example case end to end and report the physics checks.

usage: run_case.py <input.json> --data <marie-tools>/data [--device cpu|cuda] [--quick]
                   [--precision double|mixed] [--shift DX DY DZ]

The data folder is that of https://github.com/cloudmrhub/marie-tools.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch

from mariepy import cosim, fields, metrics, network, sie
from mariepy.inputs import read_case
from mariepy.solver import solve

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--data", required=True, help="marie-tools data folder")
parser.add_argument("--device", default="cpu")
parser.add_argument("--quick", action="store_true", help="low quadrature orders")
parser.add_argument("--precision", default="double", choices=("double", "mixed"))
parser.add_argument("--out", default=None)
parser.add_argument("--shift", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"),
                    help="translate the body, in metres, to place it inside the coil")
args = parser.parse_args()

t0 = time.time()
case = read_case(args.input, data=args.data, device=args.device)
if args.shift is not None:
    import dataclasses
    origin = tuple(o + d for o, d in zip(case.body.origin, args.shift))
    case = dataclasses.replace(case, body=dataclasses.replace(case.body, origin=origin))
coil = case.coil
print(f"case read in {time.time()-t0:.0f}s: voxels {case.body.n_voxels}, grid {case.body.shape}, "
      f"coil unknowns {coil.n_dof}, driven {coil.n_driven}, shield {case.shield is not None}, "
      f"linear {case.linear}, tmd {case.network.tmd}, roles {case.network.roles}", flush=True)

# Collision: any coil point inside a tissue voxel?
from mariepy.pfft import _nodes
nodes = _nodes(coil).to(case.body.device)
index = torch.round((nodes - torch.tensor(case.body.origin, device=nodes.device)) / case.body.resolution).long()
limits = torch.tensor(case.body.shape, device=nodes.device)
inside = ((index >= 0) & (index < limits)).all(dim=1)
hits = case.body.mask[tuple(index[inside].T)].sum() if bool(inside.any()) else 0
print(f"coil nodes inside tissue voxels: {int(hits)}", flush=True)

orders = {"far_order": 2, "medium_order": 2, "near_order": 4} if args.quick else {}
t1 = time.time()
result = solve(
    case.body, coil, case.medium, linear=case.linear, shield=case.shield,
    precision=args.precision, **orders,
)
t2 = time.time()
op = result.operator
raw = network.port_admittance(op.excitation, op.conductors(result.ports.coil, result.ports.shield))
asym = float((raw - raw.T).abs().max() / raw.abs().max())
taken, absorbed, scattered = fields.power_balance(op, result.fields, result.ports.body)
balance = float(((taken - absorbed - scattered).abs() / taken.abs()).max())
print(f"solve {t2-t1:.0f}s; iterations {result.ports.iterations}; residual max {max(result.ports.residual):.1e}", flush=True)
print(f"reciprocity (before symmetrising) {asym:.2e}; power balance {balance:.2e}; "
      f"absorbed per port [W per V^2] {absorbed.cpu().numpy()}", flush=True)

omega = case.medium.angular_frequency
t3 = time.time()
closed = cosim.co_simulate(case.network, result.admittance, omega)
t4 = time.time()
print(f"co-simulation {t4-t3:.0f}s costs {json.dumps(closed.costs, default=float)}", flush=True)
if closed.transmit is not None:
    s = closed.scattering
    print("transmit |S| dB:\n", np.round(20 * np.log10(s.abs().cpu().numpy()), 1))
if closed.receive_scattering is not None:
    print("receive |S_check| dB:", np.round(20 * np.log10(closed.receive_scattering.abs().cpu().numpy()), 1))

mask = case.body.mask
for side, mapping in (("transmit", closed.transmit), ("receive", closed.receive)):
    if mapping is None:
        continue
    e = cosim.calibrate(result.fields.electric, mapping)
    h = cosim.calibrate(result.fields.magnetic, mapping)
    from mariepy.fields import Fields, absorbed_power, at_centres
    cal = Fields(electric=e, magnetic=h, incident=e, scattered=e)
    p_abs = absorbed_power(op, cal)
    centres = at_centres(h)
    mu = case.medium.permeability
    b1p = mu * (centres[:, 0] + 1j * centres[:, 1])
    b1m = mu * (centres[:, 0] - 1j * centres[:, 1])
    print(f"{side}: absorbed per unit incident wave [W] {p_abs.cpu().numpy()}")
    print(f"{side}: mean |B1+| per port [uT per sqrt(W)] {(b1p.abs()[:, mask].mean(1) * 1e6 / math.sqrt(0.5)).cpu().numpy()}")
    coil_current = cosim.calibrate(op.conductors(result.ports.coil, result.ports.shield), mapping)
    loss = op.system.loss
    if case.shield is not None:
        loss = sie.block_diagonal(op.shield.system.loss, loss)
    psi = metrics.noise_covariance(e, case.body.conductivity, mask, case.body.resolution, coil=coil_current, loss=loss)
    if side == "receive":
        snr = metrics.snr(b1m, psi, case.medium, case.body.resolution, mask)
        print(f"receive: combined SNR mean/max {float(snr[mask].mean()):.3e} {float(snr[mask].max()):.3e}")
    else:
        txe = metrics.transmit_efficiency(b1p, psi, mask)
        print(f"transmit: efficiency mean/max [T^2/W] {float(txe[mask].mean()):.3e} {float(txe[mask].max()):.3e}")
if args.out and closed.scattering is not None:
    np.savez(args.out, admittance=result.admittance.cpu().numpy(), scattering=closed.scattering.cpu().numpy())
print(f"total {time.time()-t0:.0f}s")
