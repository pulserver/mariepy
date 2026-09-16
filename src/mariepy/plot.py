"""Figures of a model and its results, with matplotlib.

Ported in intent from MARIE 3.0's ``src_visualizer``: ``visualize_geometry.m``,
``visualize_coil_currents.m``, ``visualize_s_parameters.m``,
``visualize_z_parameters.m``, ``visualize_ZPm_SPm_freq_sweep.m`` and
``visualize_ideal_current_patterns.m``, with
``slices`` for any voxel map (SNR, transmit efficiency, ``B1``, SAR). Each
function draws on an axes it is given, or on a new figure, and returns the
figure. matplotlib is an optional dependency: ``pip install "mariepy[plot]"``.
"""

from __future__ import annotations

import cmath

import torch

from mariepy.body import VoxelBody
from mariepy.coil import SurfaceCoil

__all__ = [
    "coil_currents",
    "current_density",
    "geometry",
    "ideal_current_patterns",
    "impedance",
    "scattering",
    "slices",
    "sweep",
]


def _pyplot():
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "figures need matplotlib: pip install 'mariepy[plot]'"
        ) from error
    return pyplot


def _axes(ax, projection=None):
    pyplot = _pyplot()
    if ax is not None:
        return ax.figure, ax
    figure = pyplot.figure()
    return figure, figure.add_subplot(projection=projection)


def _numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def _limits(ax, points) -> None:
    """Fit a 3-D axes to some points, padding any flat direction."""
    low, high = points.min(axis=0), points.max(axis=0)
    pad = 0.05 * max(float((high - low).max()), 1e-3)
    for setter, a, b in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim), low, high, strict=True
    ):
        setter(a - pad, b + pad)


def _surfaces_and_wires(coil):
    from mariepy.wire import CombinedCoil, WireCoil

    if isinstance(coil, CombinedCoil):
        return [coil.surface], [coil.wire]
    if isinstance(coil, WireCoil):
        return [], [coil]
    return [coil], []


def geometry(body: VoxelBody, coil=None, shield: SurfaceCoil | None = None, ax=None):
    """Draw the body's voxels and the coil and shield around them.

    Parameters
    ----------
    body
        Its tissue voxels are drawn as points.
    coil
        A surface coil, a wire coil, or both together, or None.
    shield
        A shield, drawn translucent, or None.
    ax
        A 3-D axes to draw on; a new figure by default.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure, ax = _axes(ax, projection="3d")
    centres = body.coordinates().reshape(3, -1)[:, body.mask.reshape(-1)]
    x, y, z = _numpy(centres)
    ax.scatter(x, y, z, s=2, color=(254 / 255, 227 / 255, 212 / 255), label="body")
    surfaces, wires = ([], []) if coil is None else _surfaces_and_wires(coil)
    for surface in surfaces:
        ax.add_collection3d(
            Poly3DCollection(
                _numpy(surface.mesh.vertices()),
                facecolor=(0.72, 0.45, 0.2),
                edgecolor="k",
                linewidth=0.2,
            )
        )
    for wire in wires:
        for start, stop in wire.loops:
            nodes = _numpy(
                torch.cat([wire.centre[start:stop], wire.centre[start : start + 1]])
            )
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=(0.72, 0.45, 0.2))
    if shield is not None:
        ax.add_collection3d(
            Poly3DCollection(
                _numpy(shield.mesh.vertices()), facecolor=(0.6, 0.6, 0.6), alpha=0.15
            )
        )
    points = [centres.transpose(0, 1)]
    points += [s.mesh.nodes for s in surfaces] + [w.centre for w in wires]
    if shield is not None:
        points.append(shield.mesh.nodes)
    _limits(ax, _numpy(torch.cat([p.cpu() for p in points])))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    return figure


def current_density(coil: SurfaceCoil, current: torch.Tensor) -> torch.Tensor:
    """Give the surface current density at each triangle's centroid.

    Parameters
    ----------
    coil
        The surface coil.
    current
        Its basis coefficients, shape ``(n_dof,)``.

    Returns
    -------
    torch.Tensor
        Shape ``(n_triangles, 3)``, complex, in amperes per metre.
    """
    vertices = coil.mesh.vertices()
    centroid = vertices.mean(dim=1)
    arms = centroid[:, None, :] - vertices  # (t, 3, 3), from each free vertex
    dof = coil.dof_of_triangle()
    scale = coil.mesh.edge_lengths() * coil.signs / (2 * coil.mesh.areas()[:, None])
    weights = torch.where(dof >= 0, current[dof.clamp(min=0)], 0) * scale.to(
        current.dtype
    )
    return torch.einsum("ta,tac->tc", weights, arms.to(current.dtype))


def coil_currents(coil, current: torch.Tensor, ax=None):
    """Colour the coil by the magnitude of its current, as ``visualize_coil_currents.m``.

    Parameters
    ----------
    coil
        A surface coil, a wire coil, or both together.
    current
        Its coefficients for one port, shape ``(n_dof,)``, wire first.
    ax
        A 3-D axes to draw on; a new figure by default.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.
    """
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    figure, ax = _axes(ax, projection="3d")
    surfaces, wires = _surfaces_and_wires(coil)
    offset = 0
    pieces = []
    for wire in wires:
        pieces.append(("wire", wire, current[offset : offset + wire.n_dof]))
        offset += wire.n_dof
    for surface in surfaces:
        pieces.append(("surface", surface, current[offset : offset + surface.n_dof]))
        offset += surface.n_dof

    magnitudes = []
    for kind, part, values in pieces:
        if kind == "surface":
            magnitudes.append(
                torch.linalg.vector_norm(current_density(part, values), dim=-1)
            )
        else:
            magnitudes.append(values.abs())
    top = float(max(float(m.max()) for m in magnitudes)) if magnitudes else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=top or 1.0)
    for (kind, part, _), magnitude in zip(pieces, magnitudes, strict=True):
        shade = cm.viridis(norm(_numpy(magnitude)))
        if kind == "surface":
            ax.add_collection3d(
                Poly3DCollection(_numpy(part.mesh.vertices()), facecolors=shade)
            )
        else:
            segments = _numpy(torch.stack([part.centre, part.last], dim=1))
            ax.add_collection3d(Line3DCollection(segments, colors=shade, linewidths=2))
    _limits(
        ax,
        _numpy(
            torch.cat(
                [p.mesh.nodes if k == "surface" else p.centre for k, p, _ in pieces]
            )
        ),
    )
    figure.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="|J|")
    return figure


def ideal_current_patterns(
    support: SurfaceCoil,
    current: torch.Tensor,
    body: VoxelBody | None = None,
    target=None,
    *,
    phases=(0.0, cmath.pi / 2),
):
    """Draw a surface current at two phases of the carrier.

    The figure of ``visualize_ideal_current_patterns.m``: the support's
    triangles carry an arrow of the instantaneous current at their centroid, one
    panel per phase, over the body and the target point the pattern was formed
    for.

    Parameters
    ----------
    support
        The surface the current lives on, the basis's own support.
    current
        Its basis coefficients, shape ``(n_dof,)``, as
        :func:`mariepy.basis.ideal_currents` gives them.
    body
        Drawn as points behind the support, or None.
    target
        The point the pattern was formed for, shape ``(3,)``, marked in red, or
        None.
    phases
        The carrier phases in radians, one panel each.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per phase.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pyplot = _pyplot()
    density = current_density(support, current)
    centroids = support.mesh.vertices().mean(dim=1)
    nodes = support.mesh.nodes
    span = float((nodes.max(dim=0).values - nodes.min(dim=0).values).max())
    peak = float(torch.linalg.vector_norm(density, dim=-1).max())
    reach = 0.12 * span / peak if peak else 0.0

    figure = pyplot.figure(figsize=(5.5 * len(phases), 5.0))
    for panel, phase in enumerate(phases):
        ax = figure.add_subplot(1, len(phases), panel + 1, projection="3d")
        if body is not None:
            centres = body.coordinates().reshape(3, -1)[:, body.mask.reshape(-1)]
            x, y, z = _numpy(centres)
            ax.scatter(x, y, z, s=2, color=(224 / 255, 177 / 255, 164 / 255))
        ax.add_collection3d(
            Poly3DCollection(
                _numpy(support.mesh.vertices()),
                facecolor=(255 / 255, 238 / 255, 117 / 255),
                alpha=0.3,
            )
        )
        if target is not None:
            point = _numpy(torch.as_tensor(target, dtype=torch.float64).reshape(3))
            ax.scatter(*point, s=60, marker="s", color=(1.0, 44 / 255, 44 / 255))
        arrows = (density * cmath.exp(1j * phase)).real * reach
        base, tip = _numpy(centroids), _numpy(arrows)
        ax.quiver(
            base[:, 0],
            base[:, 1],
            base[:, 2],
            tip[:, 0],
            tip[:, 1],
            tip[:, 2],
            color="k",
            linewidth=0.6,
        )
        ax.set_title(f"$\\omega t = {phase:.2f}$")
        ax.set_axis_off()
        _limits(ax, _numpy(nodes))
    return figure


def scattering(matrix: torch.Tensor, ax=None, *, floor: float = -40.0):
    """Show a scattering matrix in decibels, as ``visualize_s_parameters.m``.

    Parameters
    ----------
    matrix
        Square, complex.
    ax
        An axes to draw on; a new figure by default.
    floor
        Lowest decibel value on the colour scale.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn on.
    """
    figure, ax = _axes(ax)
    decibels = _numpy(20 * torch.log10(matrix.abs().clamp(min=1e-30)))
    image = ax.imshow(decibels, vmin=floor, vmax=0.0, cmap="Reds_r")
    for i in range(decibels.shape[0]):
        for j in range(decibels.shape[1]):
            ax.text(j, i, f"{decibels[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ports = range(1, decibels.shape[0] + 1)
    ax.set_xticks(range(decibels.shape[1]), ports)
    ax.set_yticks(range(decibels.shape[0]), ports)
    ax.set_xlabel("port")
    ax.set_ylabel("port")
    figure.colorbar(image, ax=ax, label="|S| (dB)")
    return figure


def impedance(matrix: torch.Tensor):
    """Show an impedance matrix's resistance and reactance, as ``visualize_z_parameters.m``.

    Parameters
    ----------
    matrix
        Square, complex, in ohms.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with one panel for each part.
    """
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(9, 4))
    for ax, part, name in (
        (axes[0], matrix.real, "Re Z (ohm)"),
        (axes[1], matrix.imag, "Im Z (ohm)"),
    ):
        values = _numpy(part)
        image = ax.imshow(values, cmap="coolwarm")
        ports = range(1, values.shape[0] + 1)
        ax.set_xticks(range(values.shape[1]), ports)
        ax.set_yticks(range(values.shape[0]), ports)
        ax.set_title(name)
        figure.colorbar(image, ax=ax)
    return figure


def sweep(result):
    """Plot the matched ports across a band, as ``visualize_ZPm_SPm_freq_sweep.m``.

    Parameters
    ----------
    result
        A :class:`mariepy.cosim.Sweep`.

    Returns
    -------
    matplotlib.figure.Figure
        Reflection in decibels above, input impedance below, one line per
        port, transmit side solid and receive side dashed.
    """
    pyplot = _pyplot()
    figure, (top, bottom) = pyplot.subplots(2, 1, sharex=True, figsize=(7, 6))
    frequency = _numpy(result.frequency) / 1e6
    sides = []
    if result.transmit_scattering is not None:
        sides.append(
            (
                "Tx",
                "-",
                torch.diagonal(result.transmit_scattering, dim1=-2, dim2=-1),
                torch.diagonal(result.transmit_impedance, dim1=-2, dim2=-1),
            )
        )
    if result.receive_scattering is not None:
        sides.append(("Rx", "--", result.receive_scattering, result.receive_impedance))
    for name, style, reflection, input_impedance in sides:
        for port in range(reflection.shape[-1]):
            label = f"{name} {port + 1}"
            top.plot(
                frequency,
                _numpy(20 * torch.log10(reflection[:, port].abs())),
                style,
                label=label,
            )
            bottom.plot(frequency, _numpy(input_impedance[:, port].real), style)
            bottom.plot(
                frequency, _numpy(input_impedance[:, port].imag), style, alpha=0.5
            )
    working = frequency[result.index]
    for ax in (top, bottom):
        ax.axvline(working, color="k", linewidth=0.5)
    top.set_ylabel("|S| (dB)")
    top.legend(fontsize=7)
    bottom.set_ylabel("Z (ohm): Re, then Im faded")
    bottom.set_xlabel("frequency (MHz)")
    return figure


def slices(
    volume: torch.Tensor, index=None, *, mask=None, title: str = "", cmap="inferno"
):
    """Show three orthogonal slices of a voxel map.

    Parameters
    ----------
    volume
        Shape ``(n1, n2, n3)``, real.
    index
        The slice along each axis; the middle by default.
    mask
        Where to show values; elsewhere is blank.
    title
        Figure title.
    cmap
        Colour map.

    Returns
    -------
    matplotlib.figure.Figure
        Three panels: across the first axis, the second, the third.
    """
    pyplot = _pyplot()
    values = volume.detach().to(torch.float64).cpu()
    if mask is not None:
        values = torch.where(mask.cpu(), values, torch.nan)
    if index is None:
        index = tuple(n // 2 for n in values.shape)
    top = float(torch.nan_to_num(values, nan=float("-inf")).max())
    figure, axes = pyplot.subplots(1, 3, figsize=(11, 4))
    cuts = (values[index[0]], values[:, index[1]], values[:, :, index[2]])
    names = ("x", "y", "z")
    for ax, cut, name, position in zip(axes, cuts, names, index, strict=True):
        image = ax.imshow(cut.T.numpy(), origin="lower", cmap=cmap, vmin=0, vmax=top)
        ax.set_title(f"{name} = {position}")
        ax.set_axis_off()
    figure.colorbar(image, ax=list(axes), shrink=0.8)
    if title:
        figure.suptitle(title)
    return figure
