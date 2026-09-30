"""The 8-channel transmit head coil of a 7 T scanner, as microstrips driven by current sources.

usage, from the repository root:
    python examples/nova8tx.py geometry --out nova
    python examples/nova8tx.py phantom --out nova --resolution 4
    python examples/nova8tx.py bodies --out nova --resolution 4 --subjects 4 5 6

The coil follows the model of Ozkara et al., "Generating an EM Simulation Model
of a Clinically-Used RF Head Coil at 7T", ISMRM 2025, abstract 3957: eight
channels on a cylinder 350 mm across, each channel four copper microstrips
173 mm long and 12 mm wide with 6, 17 and 6 mm between them, and current
sources at both ends -- one in series with each strip and one across each gap
between neighbouring strips, seven at each end.

Each channel's drive is the current every one of its sources carries. Those
currents are the model's only free parameters: Ozkara et al. fit them, per
channel and by least squares, to the field the vendor's own simulation puts in
a spherical phantom. ``--weights`` reads a file of the same shape, fitted here
to measured maps by ``fit`` once phantom measurements exist; without it the
strips carry equal currents and the gaps none, which sets the pattern of each
channel but not the scale of the whole.

The three subcommands are:

- ``geometry``: build the coil, solve it in free space, and report its port
  matrix and the currents of each channel;
- ``phantom``: solve the sphere the model was fitted in -- 82 mm across,
  permittivity 80 and conductivity 1 S/m -- and write each channel's B1+;
- ``bodies``: solve BrainWeb subjects, write each one's maps and field images,
  and compress the population's SAR matrices into virtual observation points.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from mariepy.coil import Port, SurfaceCoil
from mariepy.constants import Medium
from mariepy.mesh import SurfaceMesh

# The abstract's geometry, in metres.
DIAMETER = 0.350
STRIP_LENGTH = 0.173
STRIP_WIDTH = 0.012
GAPS = (0.006, 0.017, 0.006)
SOURCE_LENGTH = 0.0325
CHANNELS = 8

# 7 T, the field the coil is built for.
FIELD_STRENGTH = 6.98

# The sphere Ozkara et al. fit the source currents in.
PHANTOM_RADIUS = 0.082
PHANTOM_PERMITTIVITY = 80.0
PHANTOM_CONDUCTIVITY = 1.0


def build_mesh(
    *,
    diameter: float = DIAMETER,
    channels: int = CHANNELS,
    strip_length: float = STRIP_LENGTH,
    strip_width: float = STRIP_WIDTH,
    gaps=GAPS,
    source_length: float = SOURCE_LENGTH,
    along: int = 18,
    across: int = 2,
    device: torch.device | str = "cpu",
) -> SurfaceMesh:
    """Build the array's conductors and tag the edges its current sources drive.

    A channel is ``len(gaps) + 1`` strips along the cylinder's axis, each
    divided ``along`` times down its length and ``across`` times over its
    width, joined near each end by a patch across every gap. Its sources are
    the rows of edges ``source_length`` from each end of each strip, and the
    edge down the middle of each patch: with four strips, seven at each end.

    Tags run channel by channel, and within a channel the near end's strips,
    the near end's gaps, then the far end's, so channel ``c`` owns tags
    ``2 * (len(gaps) + 1 + len(gaps)) * c + 1`` upwards.

    Parameters
    ----------
    diameter
        Diameter of the cylinder the strips lie on, in metres.
    channels
        Channels equally spaced about the cylinder.
    strip_length, strip_width
        Length along the axis and width around it of one strip, in metres.
    gaps
        Distance between neighbouring strips of a channel, in metres, one
        fewer than the strips.
    source_length
        Distance from each end of a strip to the source in series with it.
    along, across
        Divisions of a strip along its length and across its width.
    device
        Device the arrays are built on.

    Returns
    -------
    SurfaceMesh
        The conductors, with one physical line per current source.

    Raises
    ------
    ValueError
        If the strips would overlap, or a division is too coarse to carry a
        source away from the ends.
    """
    strips = len(gaps) + 1
    span = strips * strip_width + sum(gaps)
    radius = 0.5 * diameter
    if span >= 2.0 * math.pi * radius / channels:
        raise ValueError(
            f"the channels overlap: {span * 1e3:.1f} mm of strip and gap in "
            f"{2e3 * math.pi * radius / channels:.1f} mm of circumference"
        )
    if along < 4 or across < 1:
        raise ValueError(f"need along >= 4 and across >= 1; got {along}, {across}")

    step = strip_length / along
    source_row = round(source_length / step)
    if not 1 <= source_row < along // 2:
        raise ValueError(
            f"a source {source_length * 1e3:.1f} mm from the end falls on row "
            f"{source_row} of {along}; divide the strip differently"
        )

    heights = torch.linspace(
        -0.5 * strip_length, 0.5 * strip_length, along + 1, dtype=torch.float64
    )
    nodes: list[torch.Tensor] = []
    triangles: list[tuple[int, int, int]] = []
    lines: list[tuple[int, int]] = []
    tags: list[int] = []

    # Where each strip of a channel starts, as an angle about the axis.
    offsets = []
    edge = -0.5 * span
    for strip in range(strips):
        offsets.append(edge)
        edge += strip_width + (gaps[strip] if strip < len(gaps) else 0.0)

    def corner(base: int, column: int, row: int) -> int:
        return base + column * (along + 1) + row

    tag = 0
    for channel in range(channels):
        centre = 2.0 * math.pi * channel / channels
        first = len(nodes)
        for strip in range(strips):
            base = len(nodes)
            starts = offsets[strip] + torch.linspace(
                0.0, strip_width, across + 1, dtype=torch.float64
            )
            for arc in starts:
                angle = centre + arc / radius
                for height in heights:
                    nodes.append(
                        torch.tensor(
                            [
                                radius * math.cos(float(angle)),
                                radius * math.sin(float(angle)),
                                float(height),
                            ],
                            dtype=torch.float64,
                        )
                    )
            for column in range(across):
                for row in range(along):
                    here = corner(base, column, row)
                    up = corner(base, column, row + 1)
                    over = corner(base, column + 1, row)
                    both = corner(base, column + 1, row + 1)
                    triangles.append((here, up, both))
                    triangles.append((here, both, over))

        # A source in series with each strip, at each end: the row of edges
        # across the strip, which every current along it crosses.
        for row in (source_row, along - source_row):
            for strip in range(strips):
                base = first + strip * (across + 1) * (along + 1)
                tag += 1
                for column in range(across):
                    lines.append(
                        (corner(base, column, row), corner(base, column + 1, row))
                    )
                    tags.append(tag)
            # A patch across each gap, hung on the nodes of the strips it
            # joins and divided so that no edge of it outgrows a strip's. Its
            # source drives the edge down the middle.
            for gap, width in enumerate(gaps):
                left = first + gap * (across + 1) * (along + 1)
                right = first + (gap + 1) * (across + 1) * (along + 1)
                columns = max(2, 2 * round(0.5 * width / (strip_width / across)))
                arc = offsets[gap] + strip_width
                angle = centre + arc / radius
                bridge = [corner(left, across, row), corner(left, across, row + 1)]
                for column in range(1, columns):
                    step_angle = angle + column * width / (columns * radius)
                    for height in (heights[row], heights[row + 1]):
                        nodes.append(
                            torch.tensor(
                                [
                                    radius * math.cos(float(step_angle)),
                                    radius * math.sin(float(step_angle)),
                                    float(height),
                                ],
                                dtype=torch.float64,
                            )
                        )
                    bridge += [len(nodes) - 2, len(nodes) - 1]
                bridge += [corner(right, 0, row), corner(right, 0, row + 1)]
                for column in range(columns):
                    near, far = bridge[2 * column], bridge[2 * column + 1]
                    other_near, other_far = (
                        bridge[2 * column + 2],
                        bridge[2 * column + 3],
                    )
                    triangles.append((near, other_near, other_far))
                    triangles.append((near, other_far, far))
                tag += 1
                middle = columns // 2
                lines.append((bridge[2 * middle], bridge[2 * middle + 1]))
                tags.append(tag)

    # Wind every triangle so its normal points away from the axis: two
    # neighbours that traverse their shared edge the same way carry no basis
    # function between them.
    placed = torch.stack(nodes)
    faces = torch.tensor(triangles, dtype=torch.int64)
    corners = placed[faces]
    normals = torch.linalg.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    outward = corners.mean(dim=1)
    outward[:, 2] = 0.0
    inward = (normals * outward).sum(dim=1) < 0.0
    faces[inward] = faces[inward][:, [0, 2, 1]]

    mesh = SurfaceMesh(
        nodes=placed.to(device),
        triangles=faces.to(device),
        triangle_tags=torch.ones(len(triangles), dtype=torch.int64, device=device),
        lines=torch.tensor(lines, dtype=torch.int64, device=device),
        line_tags=torch.tensor(tags, dtype=torch.int64, device=device),
    )
    return mesh.align_to_lines()


def build_coil(mesh: SurfaceMesh, *, sources: int) -> SurfaceCoil:
    """Give every tagged line a port of its own."""
    elements = tuple(
        Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)
        for tag in range(1, sources + 1)
    )
    return SurfaceCoil.build(mesh, elements)


def source_layout(channels: int = CHANNELS, gaps=GAPS) -> dict:
    """Say which port is which source, by channel, end and kind."""
    strips = len(gaps) + 1
    per_end = strips + len(gaps)
    per_channel = 2 * per_end
    layout = {"per_channel": per_channel, "per_end": per_end, "strips": strips}
    layout["channel_of"] = [
        port // per_channel for port in range(channels * per_channel)
    ]
    kinds = ["strip"] * strips + ["gap"] * len(gaps)
    layout["kind_of"] = [
        kinds[(port % per_channel) % per_end] for port in range(channels * per_channel)
    ]
    layout["end_of"] = [
        (port % per_channel) // per_end for port in range(channels * per_channel)
    ]
    return layout


def default_weights(channels: int = CHANNELS, gaps=GAPS) -> torch.Tensor:
    """Give each channel's sources the currents the strips carry in parallel.

    Both sources of a strip drive one ampere the same way along it, so the
    strip carries that current from end to end, and no current crosses the
    gaps. The pattern of a channel follows; its scale, and the share between
    the strips, wait on a fit to measured maps.

    Returns
    -------
    torch.Tensor
        Shape ``(channels, 2 * (2 * len(gaps) + 1))``, complex: the current of
        every source of every channel, the ports in the order
        :func:`build_mesh` tags them.
    """
    layout = source_layout(channels, gaps)
    per_channel = layout["per_channel"]
    weights = torch.zeros((channels, per_channel), dtype=torch.complex128)
    for port in range(per_channel):
        if layout["kind_of"][port] == "strip":
            weights[:, port] = 1.0
    return weights


def channel_currents(
    coil: SurfaceCoil, medium: Medium, weights: torch.Tensor, *, system=None
):
    """Solve the coil in free space and combine its sources into channel currents.

    Parameters
    ----------
    coil
        The array, one port per current source.
    medium
        The frequency.
    weights
        Shape ``(n_channels, n_sources_per_channel)``, the current each source
        carries.
    system
        The coil's assembled system, when it is already to hand.

    Returns
    -------
    currents : torch.Tensor
        Shape ``(n_channels, n_dof)``, the surface current of each channel.
    system : mariepy.sie.CoilSystem
        The system the solve used.
    impedance : torch.Tensor
        The port impedance matrix of the array in free space.
    """
    from mariepy import network, sie

    system = sie.assemble(coil, medium) if system is None else system
    driven = torch.linalg.solve(
        system.impedance, system.excitation.transpose(0, 1)
    ).transpose(0, 1)
    admittance = network.symmetrise(network.port_admittance(system.excitation, driven))
    impedance = network.y_to_z(admittance)

    # A unit current into one port with the others open is the voltage drive
    # its column of the port impedance gives.
    per_source = impedance.transpose(0, 1).to(driven.dtype) @ driven
    channels, per_channel = weights.shape
    currents = torch.stack(
        [
            weights[channel]
            @ per_source[channel * per_channel : (channel + 1) * per_channel]
            for channel in range(channels)
        ]
    )
    return currents, system, impedance


def solve_body(
    body,
    coil: SurfaceCoil,
    medium: Medium,
    currents: torch.Tensor,
    *,
    system,
    coupling=None,
    tol: float = 1e-5,
    precision: str = "mixed",
    linear: bool = False,
    **orders,
):
    """Drive a body with prescribed coil currents and give the fields they induce.

    The sources set the coil's current whatever the body does, so the body
    sees the coil's field as an incident field and never reacts back on it:
    the model Ozkara et al. fit, and what a current source means.

    Parameters
    ----------
    body
        The body and its grid.
    coil
        The array.
    medium
        The frequency.
    currents
        Shape ``(n_channels, n_dof)``, the coil's surface current per channel.
    system
        The coil's assembled system.
    coupling
        A coupling assembled over a region holding this body, from
        :func:`mariepy.pfft.assemble`; built for this body when absent.
    tol
        Target for the relative residual of each channel's solve.
    precision
        ``"mixed"`` takes the body's products in complex64.
    linear
        Give the body the piecewise-linear basis.
    **orders
        Quadrature orders, as :func:`mariepy.pfft.assemble` takes them.

    Returns
    -------
    fields : mariepy.fields.Fields
        The total electric and magnetic fields of every channel.
    operator : mariepy.system.CoupledOperator
        The operator they were solved with.
    """
    from mariepy import fields as field_module
    from mariepy import pfft
    from mariepy.gmres import gmres, refine
    from mariepy.preconditioner import body_diagonal
    from mariepy.system import CoupledOperator

    if coupling is None:
        coupling = pfft.assemble(
            body, coil, system.impedance, medium, linear=linear, **orders
        )
    else:
        coupling = pfft.restrict(coupling, body.mask)
    operator = CoupledOperator(
        body=body, coil=coil, medium=medium, system=system, coupling=coupling
    )
    diagonal = body_diagonal(body, medium, linear=coupling.linear)
    krylov = refine if precision == "mixed" else gmres
    solved = []
    for channel in range(currents.shape[0]):
        solution = krylov(
            operator.body_block,
            operator.couple(currents[channel]),
            preconditioner=lambda vector: diagonal.to(vector.dtype) * vector,
            tol=tol,
        )
        solved.append(solution.x)
    return field_module.compute(operator, currents, torch.stack(solved)), operator


def _report(coil: SurfaceCoil, mesh: SurfaceMesh, impedance: torch.Tensor) -> None:
    lengths = mesh.edge_lengths()
    print(
        f"coil: {coil.n_dof} unknowns, {coil.n_driven} sources, "
        f"{mesh.n_triangles} triangles, longest edge {float(lengths.max()) * 1e3:.1f} mm"
    )
    diagonal = torch.diagonal(impedance)
    print(
        "free-space port impedance: |Z| median "
        f"{float(diagonal.abs().median()):.1f} ohm, "
        f"R median {float(diagonal.real.median()):.2f} ohm"
    )


def main() -> None:
    """Run one of the three subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=("geometry", "phantom", "bodies"))
    parser.add_argument("--out", default="nova", help="folder for the outputs")
    parser.add_argument(
        "--resolution", type=float, default=4.0, help="voxel size in mm"
    )
    parser.add_argument("--along", type=int, default=18)
    parser.add_argument("--across", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", help="a .pt file of fitted source currents")
    parser.add_argument("--subjects", type=int, nargs="*", default=[4, 5, 6])
    arguments = parser.parse_args()

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    medium = Medium(FIELD_STRENGTH)
    mesh = build_mesh(
        along=arguments.along, across=arguments.across, device=arguments.device
    )
    layout = source_layout()
    coil = build_coil(mesh, sources=CHANNELS * layout["per_channel"])
    weights = (
        torch.load(arguments.weights) if arguments.weights else default_weights()
    ).to(torch.complex128)

    print(f"{medium.frequency * 1e-6:.2f} MHz")
    currents, system, impedance = channel_currents(coil, medium, weights)
    _report(coil, mesh, impedance)
    torch.save(
        {"currents": currents, "impedance": impedance, "weights": weights},
        out / "coil.pt",
    )
    if arguments.what == "geometry":
        print(f"wrote {out / 'coil.pt'}")
        return
    if arguments.what == "phantom":
        run_phantom(coil, medium, currents, system, arguments, out)
        return
    raise SystemExit(f"{arguments.what} is not built yet")


def run_phantom(coil, medium, currents, system, arguments, out: Path) -> None:
    """Solve the sphere the model was fitted in, and draw each channel's B1+."""
    import time

    from mariepy import fields as field_module
    from mariepy.body import VoxelBody

    resolution = arguments.resolution * 1e-3
    body = VoxelBody.sphere(
        PHANTOM_RADIUS,
        resolution,
        PHANTOM_PERMITTIVITY,
        PHANTOM_CONDUCTIVITY,
        padding=2,
        device=arguments.device,
    )
    print(
        f"phantom: {body.n_voxels} voxels of {arguments.resolution:.0f} mm, "
        f"grid {tuple(body.shape)}",
        flush=True,
    )
    start = time.time()
    solved, operator = solve_body(
        body, coil, medium, currents, system=system, precision="mixed"
    )
    print(f"solved eight channels in {time.time() - start:.0f}s", flush=True)
    plus, minus = field_module.circular_components(operator, solved)
    del minus
    torch.save({"plus": plus, "mask": body.mask}, out / "phantom_b1.pt")
    draw_channels(
        plus * 1e6,
        body.mask,
        out / "phantom_b1",
        title="B1+ [uT per unit drive], channel",
    )
    peak = plus.abs().amax(dim=(1, 2, 3)) * 1e6
    print("channel peak |B1+| [uT per unit drive]:", [round(float(v), 3) for v in peak])


def draw_channels(volumes, mask, stem: Path, *, title: str) -> None:
    """Write one image per channel, three orthogonal cuts, phase in the hue."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    from mariepy import plot

    ceiling = float(volumes.abs().max())
    for channel, volume in enumerate(volumes):
        figure = plot.complex_slices(
            volume,
            mask=mask,
            title=f"{title} {channel + 1}",
            ceiling=ceiling,
        )
        path = stem.with_name(f"{stem.name}_ch{channel + 1:02d}.png")
        figure.savefig(path, dpi=110, facecolor="white")
        pyplot.close(figure)
    print(f"wrote {stem.parent}/{stem.name}_ch*.png")


if __name__ == "__main__":
    main()
