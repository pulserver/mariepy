"""Figures draw what they are given."""

import numpy as np
import pytest
import torch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from mariepy import cosim, plot  # noqa: E402
from mariepy.body import VoxelBody  # noqa: E402
from mariepy.coil import Port, SurfaceCoil  # noqa: E402
from mariepy.mesh import SurfaceMesh  # noqa: E402
from mariepy.wire import CombinedCoil, WireCoil  # noqa: E402


def _port(tag):
    return Port(tag=tag, kind="port", load="none", value=0.0, quality=1.0, voltage=1.0)


def _surface():
    return SurfaceCoil.build(
        SurfaceMesh.loop(radius=0.05, width=0.01, n_around=8, n_across=1),
        (_port(1),),
    )


@pytest.fixture(autouse=True)
def _close():
    import matplotlib.pyplot as pyplot

    yield
    pyplot.close("all")


def test_the_current_density_of_one_basis_function_is_its_rwg_definition():
    coil = _surface()
    current = torch.zeros(coil.n_dof, dtype=torch.complex128)
    current[3] = 1.0
    density = plot.current_density(coil, current)
    triangle, slot = (coil.dof_of_triangle() == 3).nonzero()[0].tolist()
    vertices = coil.mesh.vertices()[triangle]
    length = coil.mesh.edge_lengths()[triangle, slot]
    sign = coil.signs[triangle, slot]
    area = coil.mesh.areas()[triangle]
    expected = sign * length / (2 * area) * (vertices.mean(0) - vertices[slot])
    torch.testing.assert_close(density[triangle].real, expected)
    carried = (coil.dof_of_triangle() == 3).any(dim=1)
    assert bool((density[~carried] == 0).all())


def test_the_scattering_figure_shows_the_matrix_in_decibels():
    matrix = torch.tensor([[0.1, 0.01j], [0.01j, 0.5]], dtype=torch.complex128)
    figure = plot.scattering(matrix)
    shown = figure.axes[0].images[0].get_array()
    np.testing.assert_allclose(shown, 20 * np.log10(np.abs(matrix.numpy())))


def test_the_slices_cut_the_map_through_the_given_voxel_and_blank_the_rest():
    volume = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    mask = volume > 5
    figure = plot.slices(volume, (1, 2, 3), mask=mask, title="map")
    first = figure.axes[0].images[0].get_array()
    np.testing.assert_array_equal(first, volume[1].T.numpy())
    third = figure.axes[2].images[0].get_array()
    assert np.isnan(np.ma.filled(third, np.nan)[0, 0])
    assert figure._suptitle.get_text() == "map"


def test_the_model_figures_draw_every_kind_of_coil():
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    surface = _surface()
    wire = WireCoil.loop(0.06, 12, (_port(1),))
    both = CombinedCoil(wire=wire, surface=surface)
    shield = SurfaceCoil.build(SurfaceMesh.sphere(radius=0.1, subdivisions=1))
    for coil in (surface, wire, both):
        assert plot.geometry(body, coil, shield).axes
        current = torch.rand(coil.n_dof, dtype=torch.float64).to(torch.complex128)
        assert plot.coil_currents(coil, current).axes
    impedance = torch.tensor([[50 + 5j, 1j], [1j, 48 - 2j]], dtype=torch.complex128)
    assert len(plot.impedance(impedance).axes) >= 2


def test_the_sweep_figure_marks_the_working_frequency():
    band = torch.linspace(120e6, 136e6, 11, dtype=torch.float64)
    reflection = torch.full((11, 2), 0.1 + 0j, dtype=torch.complex128)
    result = cosim.Sweep(
        frequency=band,
        index=5,
        transmit_impedance=torch.eye(2, dtype=torch.complex128).expand(11, 2, 2) * 50,
        transmit_scattering=torch.diag_embed(reflection),
        receive_impedance=torch.full((11, 2), 50 + 0j, dtype=torch.complex128),
        receive_scattering=reflection,
    )
    figure = plot.sweep(result)
    vertical = [
        line.get_xdata() for line in figure.axes[0].lines if len(line.get_xdata()) == 2
    ]
    assert any(np.allclose(x, 128.0) for x in vertical)
    assert len(figure.axes[0].get_legend().get_texts()) == 4


def test_the_ideal_pattern_figure_draws_one_panel_per_phase():
    body = VoxelBody.sphere(0.02, 0.01, 52.0, 0.55, padding=1)
    coil = _surface()
    generator = torch.Generator().manual_seed(1)
    current = torch.randn(coil.n_dof, dtype=torch.complex128, generator=generator)
    phases = (0.0, np.pi / 2, np.pi)
    figure = plot.ideal_current_patterns(
        coil, current, body, target=torch.zeros(3), phases=phases
    )
    assert len(figure.axes) == len(phases)
    for ax, phase in zip(figure.axes, phases, strict=True):
        assert f"{phase:.2f}" in ax.get_title()
        assert ax.collections
