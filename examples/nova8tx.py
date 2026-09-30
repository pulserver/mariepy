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
import dataclasses
import math
from pathlib import Path

import torch

from mariepy.body import VoxelBody
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

# BrainWeb's models all cover 181 mm across the head, the classic phantom in
# 181 voxels and each subject in 362.
FIELD_OF_VIEW_MM = 181.0

COIL_NAME = "nova8tx"
DRIVE_UNIT = "one ampere in each source of the channel, as --weights sets them"
FRAME = (
    "scanner frame of a head-first supine subject: x to the subject's left, "
    "y posterior, z superior, the coil's centre at the origin"
)
LICENCE = "BrainWeb models, for research use; see brainweb.bic.mni.mcgill.ca"

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
    tol: float = 1e-4,
    precision: str = "single",
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
        ``"single"`` solves in complex64, which costs about a thousandth of
        the field and runs two to three times faster; ``"mixed"`` keeps the
        Krylov basis in complex128 and takes the products in complex64;
        ``"double"`` stays in complex128.
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
    working = torch.complex64 if precision == "single" else torch.complex128
    solved = []
    for channel in range(currents.shape[0]):
        solution = krylov(
            operator.body_block,
            operator.couple(currents[channel].to(working)),
            preconditioner=lambda vector: diagonal.to(vector.dtype) * vector,
            tol=tol,
        )
        solved.append(solution.x.to(torch.complex128))
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
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="VOP overestimation, as a fraction of the worst case",
    )
    parser.add_argument(
        "--positions",
        type=float,
        nargs="*",
        default=[-10.0, 0.0, 10.0],
        help="where the head sits along z, in millimetres",
    )
    parser.add_argument(
        "--images", action="store_true", help="write a PNG per channel and data set"
    )
    parser.add_argument(
        "--precision", default="single", choices=("single", "mixed", "double")
    )
    parser.add_argument(
        "--tol", type=float, default=1e-4, help="relative residual of each solve"
    )
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
    run_bodies(coil, medium, currents, system, arguments, out)


def brainweb_head(subject: int, resolution: float, medium, device):
    """Build one BrainWeb subject's head, centred where the coil holds it.

    The head is placed with the centre of its box at the coil's own centre,
    which is what a subject in a head coil sits at. The subjects carry the
    twelve classes of BrainWeb's anatomical models, which
    ``brainweb_subject_tissues.csv`` names; the classic phantom's ten are a
    different set, and ``brainweb_tissues.csv`` names those.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent))
    import brainweb_dl
    import numpy as np
    from scipy import ndimage

    from mariepy import tissue

    # A subject's fractions fill 5 GB in double precision, and a fraction
    # carries far fewer digits than that.
    fine = np.asarray(brainweb_dl.get_mri(subject, "fuzzy"), dtype=np.float32)
    labelled, _ = ndimage.label(fine[..., 0] < 0.5)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    stray = (labelled != 0) & (labelled != sizes.argmax())
    del labelled, sizes
    fine[stray] = 0.0
    fine[stray, 0] = 1.0
    del stray
    shape = fine.shape
    scanner = np.ascontiguousarray(fine.transpose(2, 1, 0, 3)[::-1, ::-1])
    del fine
    # Every model covers the same field of view, and the files carry no
    # spacing: the classic phantom samples it at 1 mm and the subjects at
    # 0.5 mm, so the shape gives the voxel.
    native = FIELD_OF_VIEW_MM / shape[2]
    step = round(resolution * 1e3 / native)
    if not math.isclose(step * native, resolution * 1e3, rel_tol=1e-9):
        raise ValueError(
            f"a {resolution * 1e3:.1f} mm voxel is not a whole number of the "
            f"model's own {native:.2f} mm"
        )
    coarse = tissue.coarsen(torch.from_numpy(scanner).to(torch.float64), step)
    del scanner
    table = tissue.read_table(_Path(__file__).with_name("brainweb_subject_tissues.csv"))
    built = tissue.mix(
        coarse.to(device), table, medium, resolution, origin=(0.0, 0.0, 0.0)
    )

    body = built.body
    points = body.coordinates()[:, body.mask]
    middle = 0.5 * (points.amax(dim=1) + points.amin(dim=1))
    origin = tuple(float(o - c) for o, c in zip(body.origin, middle, strict=True))
    return dataclasses.replace(built, body=dataclasses.replace(body, origin=origin))


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
        body, coil, medium, currents, system=system, precision=arguments.precision
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


def place(built, shape, origin, shift):
    """Put a head on a common grid, moved by ``shift`` metres along the axes."""
    body = built.body
    permittivity = torch.ones(shape, dtype=body.permittivity.dtype, device=body.device)
    conductivity = torch.zeros_like(permittivity)
    density = torch.zeros_like(permittivity)
    mask = torch.zeros(shape, dtype=torch.bool, device=body.device)
    corner = [
        round((body.origin[axis] + shift[axis] - origin[axis]) / body.resolution)
        for axis in range(3)
    ]
    window = tuple(
        slice(corner[axis], corner[axis] + body.shape[axis]) for axis in range(3)
    )
    for target, source in (
        (permittivity, body.permittivity),
        (conductivity, body.conductivity),
        (density, built.density),
        (mask, body.mask),
    ):
        target[window] = source
    return (
        VoxelBody(
            permittivity=permittivity,
            conductivity=conductivity,
            mask=mask,
            resolution=body.resolution,
            origin=origin,
        ),
        density,
    )


def common_grid(heads, shifts, resolution: float, margin: int = 2):
    """Give the grid that holds every head at every position, centred on the coil."""
    reach = [0.0, 0.0, 0.0]
    for built in heads:
        body = built.body
        points = body.coordinates()[:, body.mask]
        for axis in range(3):
            half = 0.5 * float(points[axis].max() - points[axis].min())
            moved = max(abs(shift[axis]) for shift in shifts)
            reach[axis] = max(reach[axis], half + moved)
    shape = tuple(
        2 * (math.ceil(reach[axis] / resolution) + margin) + 1 for axis in range(3)
    )
    origin = tuple(-0.5 * (shape[axis] - 1) * resolution for axis in range(3))
    return shape, origin


def run_bodies(coil, medium, currents, system, arguments, out: Path) -> None:
    """Solve every head at every position, and compress the population's VOPs.

    The coil never changes, so the coupling is assembled once over the region
    every head reaches and restricted to each one, and a head enters the
    population once per position: a body model sits where it happens to sit,
    and the points have to cover that.
    """
    import time

    from mariepy import averaging, maps, pfft, sar, vop
    from mariepy import fields as field_module

    resolution = arguments.resolution * 1e-3
    channels = [f"ch{index + 1}" for index in range(CHANNELS)]
    shifts = [(0.0, 0.0, 1e-3 * offset) for offset in arguments.positions]

    start = time.time()
    heads = [
        brainweb_head(subject, resolution, medium, arguments.device)
        for subject in arguments.subjects
    ]
    shape, origin = common_grid(heads, shifts, resolution)
    print(
        f"{len(heads)} heads at {len(shifts)} positions on a {shape} grid of "
        f"{arguments.resolution:.0f} mm, built in {time.time() - start:.0f}s",
        flush=True,
    )

    placed = [place(built, shape, origin, shift) for built in heads for shift in shifts]
    names = [
        f"BrainWeb subject {subject:02d} at z {offset:+.0f} mm"
        for subject in arguments.subjects
        for offset in arguments.positions
    ]
    region = placed[0][0].mask.clone()
    for body, _ in placed[1:]:
        region |= body.mask
    print(
        f"region: {int(region.sum())} voxels of the {region.numel()} the grid holds",
        flush=True,
    )

    start = time.time()
    whole_region = dataclasses.replace(placed[0][0], mask=region)
    coupling = pfft.assemble(whole_region, coil, system.impedance, medium, linear=False)
    print(f"coupling assembled once in {time.time() - start:.0f}s", flush=True)

    averaged_all, whole_all = [], []
    for (body, density), name in zip(placed, names, strict=True):
        start = time.time()
        solved, operator = solve_body(
            body,
            coil,
            medium,
            currents,
            system=system,
            coupling=coupling,
            precision=arguments.precision,
            tol=arguments.tol,
        )
        solved_at = time.time()
        plus, minus = field_module.circular_components(operator, solved)
        stem = name.replace("BrainWeb subject ", "subject").replace(" at z ", "_z")
        stem = stem.replace(" mm", "").replace("+", "p").replace("-", "m")
        maps.write(
            out / f"{stem}.npz",
            plus,
            minus,
            body.mask,
            coil=COIL_NAME,
            channels=channels,
            frequency_hz=medium.frequency,
            drive_unit=DRIVE_UNIT,
            origin=body.origin,
            resolution=body.resolution,
            frame=FRAME,
            bodies=[name],
            data_licence=LICENCE,
        )
        if arguments.images:
            draw_channels(
                plus * 1e6,
                body.mask,
                out / f"{stem}_b1",
                title=f"{name}: B1+ [uT per unit drive], channel",
            )
            electric = field_module.at_centres(solved.electric)
            draw_channels(
                torch.linalg.vector_norm(electric, dim=1).to(torch.complex128),
                body.mask,
                out / f"{stem}_e",
                title=f"{name}: |E| [V/m per unit drive], channel",
            )
        drawn_at = time.time()

        mass = torch.where(body.mask, density * body.resolution**3, 0.0)
        local = sar.local_matrices(
            solved.electric, body.conductivity, density, body.mask
        )
        pool = averaging.cube_pool(mass, body.mask, 10e-3)
        averaged_all.append(averaging.averaged_matrices(pool, local, mass, body.mask))
        whole_all.append(
            sar.average(local, sar.voxel_mass(density, body.resolution, body.mask))
        )
        print(
            f"{name}: {body.n_voxels} voxels, peak 10 g eigenvalue "
            f"{float(torch.linalg.eigvalsh(averaged_all[-1]).amax()):.4g} W/kg; "
            f"eight channels {solved_at - start:.0f}s "
            f"({(solved_at - start) / CHANNELS:.0f}s a channel), maps "
            f"{drawn_at - solved_at:.0f}s, SAR {time.time() - drawn_at:.0f}s",
            flush=True,
        )

    start = time.time()
    points, _ = vop.compress(torch.cat(averaged_all), arguments.margin)
    vop.write(
        out / "nova8tx_vops.npz",
        points,
        torch.stack(whole_all),
        coil=COIL_NAME,
        frequency_hz=medium.frequency,
        drive_unit=DRIVE_UNIT,
        channels=channels,
        averaging="10 g, IEC/IEEE 62704-1",
        bodies=names,
        compression_margin=arguments.margin,
        data_licence=LICENCE,
    )
    print(
        f"wrote {out / 'nova8tx_vops.npz'}: {points.shape[0]} points at a margin of "
        f"{arguments.margin:.0%} from {sum(a.shape[0] for a in averaged_all)} cubes "
        f"over {len(names)} data sets, in {time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
