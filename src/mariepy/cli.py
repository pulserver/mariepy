"""The ``mariepy`` command: solve a MARIE case, or write its virtual observation points.

``mariepy solve`` reads a MARIE simulation file, solves it, co-simulates its
network and prints the physics checks. ``mariepy vops`` solves a case, or a
loop coil around a uniform ball with ``--sphere``, and writes the VOP file
:func:`mariepy.vop.read` and pypulseqpp read. The data folder of a case is
that of https://github.com/cloudmrhub/marie-tools.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np
import torch

from mariepy import __version__


def _read(args):
    from mariepy.inputs import read_case

    case = read_case(args.case, data=args.data, device=args.device)
    if args.shift is not None:
        origin = tuple(o + d for o, d in zip(case.body.origin, args.shift, strict=True))
        case = dataclasses.replace(
            case, body=dataclasses.replace(case.body, origin=origin)
        )
    return case


def _coil_nodes_in_tissue(case) -> int:
    from mariepy.pfft import _nodes

    body = case.body
    nodes = _nodes(case.coil).to(body.device)
    origin = torch.tensor(body.origin, device=nodes.device)
    index = torch.round((nodes - origin) / body.resolution).long()
    limits = torch.tensor(body.shape, device=nodes.device)
    inside = ((index >= 0) & (index < limits)).all(dim=1)
    if not bool(inside.any()):
        return 0
    return int(body.mask[tuple(index[inside].T)].sum())


def _solve(args) -> None:
    from mariepy import cosim, fields, metrics, network, sie
    from mariepy.solver import solve

    t0 = time.time()
    case = _read(args)
    coil = case.coil
    print(
        f"case read in {time.time() - t0:.0f}s: voxels {case.body.n_voxels}, "
        f"grid {case.body.shape}, coil unknowns {coil.n_dof}, driven {coil.n_driven}, "
        f"shield {case.shield is not None}, linear {case.linear}, "
        f"tmd {case.network.tmd}, roles {case.network.roles}",
        flush=True,
    )
    print(f"coil nodes inside tissue voxels: {_coil_nodes_in_tissue(case)}", flush=True)

    orders = {"far_order": 2, "medium_order": 2, "near_order": 4} if args.quick else {}
    t1 = time.time()
    result = solve(
        case.body,
        coil,
        case.medium,
        linear=case.linear,
        shield=case.shield,
        precision=args.precision,
        **orders,
    )
    op = result.operator
    raw = network.port_admittance(
        op.excitation, op.conductors(result.ports.coil, result.ports.shield)
    )
    asym = float((raw - raw.T).abs().max() / raw.abs().max())
    taken, absorbed, scattered = fields.power_balance(
        op, result.fields, result.ports.body
    )
    balance = float(((taken - absorbed - scattered).abs() / taken.abs()).max())
    print(
        f"solve {time.time() - t1:.0f}s; iterations {result.ports.iterations}; "
        f"residual max {max(result.ports.residual):.1e}\n"
        f"reciprocity (before symmetrising) {asym:.2e}; power balance {balance:.2e}; "
        f"absorbed per port [W per V^2] {absorbed.cpu().numpy()}",
        flush=True,
    )

    medium, mask = case.medium, case.body.mask
    closed = cosim.co_simulate(
        case.network, result.admittance, medium.angular_frequency
    )
    print(f"co-simulation costs {json.dumps(closed.costs, default=float)}", flush=True)
    if closed.transmit is not None:
        decibels = 20 * np.log10(closed.scattering.abs().cpu().numpy())
        print("transmit |S| dB:\n", np.round(decibels, 1))
    if closed.receive_scattering is not None:
        decibels = 20 * np.log10(closed.receive_scattering.abs().cpu().numpy())
        print("receive |S_check| dB:", np.round(decibels, 1))

    loss = op.system.loss
    if case.shield is not None:
        loss = sie.block_diagonal(op.shield.system.loss, loss)
    for side, mapping in (("transmit", closed.transmit), ("receive", closed.receive)):
        if mapping is None:
            continue
        e = cosim.calibrate(result.fields.electric, mapping)
        h = cosim.calibrate(result.fields.magnetic, mapping)
        p_abs = fields.absorbed_power(
            op, fields.Fields(electric=e, magnetic=h, incident=e, scattered=e)
        )
        centres = fields.at_centres(h)
        b1p = medium.permeability * (centres[:, 0] + 1j * centres[:, 1])
        b1m = medium.permeability * (centres[:, 0] - 1j * centres[:, 1])
        b1p_mean = b1p.abs()[:, mask].mean(1) * 1e6 / math.sqrt(0.5)
        print(f"{side}: absorbed per unit incident wave [W] {p_abs.cpu().numpy()}")
        print(f"{side}: mean |B1+| per port [uT per sqrt(W)] {b1p_mean.cpu().numpy()}")
        current = cosim.calibrate(
            op.conductors(result.ports.coil, result.ports.shield), mapping
        )
        psi = metrics.noise_covariance(
            e,
            case.body.conductivity,
            mask,
            case.body.resolution,
            coil=current,
            loss=loss,
        )
        if side == "receive":
            snr = metrics.snr(b1m, psi, medium, case.body.resolution, mask)
            print(
                f"receive: combined SNR mean/max "
                f"{float(snr[mask].mean()):.3e} {float(snr[mask].max()):.3e}"
            )
        else:
            txe = metrics.transmit_efficiency(b1p, psi, mask)
            print(
                f"transmit: efficiency mean/max [T^2/W] "
                f"{float(txe[mask].mean()):.3e} {float(txe[mask].max()):.3e}"
            )
    if args.out and closed.scattering is not None:
        np.savez(
            args.out,
            admittance=result.admittance.cpu().numpy(),
            scattering=closed.scattering.cpu().numpy(),
        )
    print(f"total {time.time() - t0:.0f}s")


def _vops(args) -> None:
    from mariepy import averaging, sar, vop
    from mariepy.solver import solve

    t0 = time.time()
    drive_unit = "1 V across the port, unmatched"
    density = None
    if args.sphere:
        from mariepy.body import VoxelBody
        from mariepy.coil import Port, SurfaceCoil
        from mariepy.constants import Medium
        from mariepy.mesh import SurfaceMesh

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
        result = solve(body, coil, medium, far_order=2, medium_order=2, near_order=4)
    else:
        case = _read(args)
        medium, coil, body = case.medium, case.coil, case.body
        if args.labels and args.table:
            from mariepy import tissue

            labels = torch.from_numpy(np.load(args.labels)).to(args.device)
            built = tissue.build(
                labels,
                tissue.read_table(args.table),
                medium,
                body.resolution,
                origin=body.origin,
            )
            body, density = built.body, built.density
        channels = [f"port{index + 1}" for index in range(coil.n_driven)]
        name, licence = args.case, "set by the body model"
        result = solve(
            body,
            coil,
            medium,
            linear=case.linear,
            shield=case.shield,
            precision=args.precision,
        )
    print(
        f"solved in {time.time() - t0:.0f}s: {body.n_voxels} voxels, "
        f"{len(channels)} channels, iterations {result.ports.iterations}",
        flush=True,
    )

    if density is None:
        if args.density.endswith(".npy"):
            density = torch.from_numpy(np.load(args.density)).to(body.device)
        else:
            density = torch.full_like(
                body.conductivity, float(args.density), dtype=torch.float64
            )
            print(f"density {args.density} kg/m^3 everywhere", flush=True)

    mass = torch.where(body.mask, density * body.resolution**3, 0.0)
    t1 = time.time()
    local = sar.local_matrices(
        result.fields.electric, body.conductivity, density, body.mask
    )
    pool = averaging.cube_pool(mass, body.mask, args.target * 1e-3)
    averaged = averaging.averaged_matrices(pool, local, mass, body.mask)
    largest = float(torch.linalg.eigvalsh(averaged)[:, -1].max())
    print(
        f"{len(pool)} averaging cubes over {args.target:g} g in {time.time() - t1:.0f}s; "
        f"largest eigenvalue {largest:.4g} W/kg",
        flush=True,
    )

    t2 = time.time()
    points, _ = vop.compress(averaged, args.margin)
    whole = sar.average(local, sar.voxel_mass(density, body.resolution, body.mask))
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
    print(f"wrote {args.out} with mariepy {__version__} in {time.time() - t0:.0f}s")


def _case_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "case", nargs=None if required else "?", help="a MARIE simulation file (JSON)"
    )
    parser.add_argument("--data", help="the marie-tools data folder")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--precision", default="double", choices=("double", "mixed"))
    parser.add_argument(
        "--shift",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="translate the body, in metres, to place it inside the coil",
    )


def main(argv: list[str] | None = None) -> None:
    """Run the ``mariepy`` command line.

    Parameters
    ----------
    argv
        Arguments after the program name; None reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(prog="mariepy", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    solve = commands.add_parser("solve", help="solve a case and print its checks")
    _case_arguments(solve, required=True)
    solve.add_argument("--quick", action="store_true", help="low quadrature orders")
    solve.add_argument("--out", help="a .npz for the admittance and scattering")
    solve.set_defaults(run=_solve)

    vops = commands.add_parser("vops", help="write a coil's VOP file")
    _case_arguments(vops, required=False)
    vops.add_argument("--sphere", action="store_true", help="a loop coil around a ball")
    vops.add_argument("--labels", help="a .npy volume of tissue labels")
    vops.add_argument("--table", help="the tissue table the labels are named against")
    vops.add_argument(
        "--density", default="1000", help="kg/m^3, a number or a .npy grid"
    )
    vops.add_argument("--target", type=float, default=10.0, help="averaging mass in g")
    vops.add_argument("--margin", type=float, default=0.05, help="VOP overestimation")
    vops.add_argument("--out", required=True, help="the VOP file to write")
    vops.set_defaults(run=_vops)

    args = parser.parse_args(argv)
    if args.command == "vops" and args.sphere == (args.case is not None):
        parser.error("vops takes a case file or --sphere, not both")
    args.run(args)
