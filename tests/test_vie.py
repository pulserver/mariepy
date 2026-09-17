"""The body kernel integrates what it should, and the FFT path applies it."""

import itertools

import pytest
import torch

from mariepy import tucker, vie

WAVENUMBER = 30.0
RESOLUTION = 0.01


def _offsets(triples, resolution=RESOLUTION, device=None):
    return resolution * torch.tensor(triples, dtype=torch.float64, device=device)


def test_the_dyadic_kernel_is_symmetric_under_exchanging_two_axes(device):
    """Swapping x and y must swap the xx and yy components and fix xy."""
    separation = _offsets([[1.0, 2.0, 3.0]], device=device)
    swapped = separation[:, [1, 0, 2]]
    here = vie.green_n(separation, WAVENUMBER)
    there = vie.green_n(swapped, WAVENUMBER)
    assert torch.allclose(here[:, 0], there[:, 3])
    assert torch.allclose(here[:, 1], there[:, 1])
    assert torch.allclose(here[:, 5], there[:, 5])


@pytest.mark.parametrize(
    ("component", "parity"),
    [
        (0, (+1, +1, +1)),
        (1, (-1, -1, +1)),
        (2, (-1, +1, -1)),
        (3, (+1, +1, +1)),
        (4, (+1, -1, -1)),
        (5, (+1, +1, +1)),
    ],
)
def test_the_dyadic_kernel_has_the_parity_the_circulant_embedding_assumes(
    component, parity, device
):
    """The mirror signs in tucker.circulant_tucker describe this kernel."""
    separation = _offsets([[1.0, 2.0, 3.0]], device=device)
    here = vie.green_n(separation, WAVENUMBER)[0, component]
    for axis in range(3):
        flipped = separation.clone()
        flipped[:, axis] *= -1.0
        there = vie.green_n(flipped, WAVENUMBER)[0, component]
        assert torch.allclose(there, parity[axis] * here)


@pytest.mark.parametrize(("component", "axis"), [(0, 0), (1, 1), (2, 2)])
def test_the_curl_kernel_is_odd_in_its_own_axis_and_even_in_the_others(
    component, axis, device
):
    separation = _offsets([[1.0, 2.0, 3.0]], device=device)
    here = vie.green_k(separation, WAVENUMBER)[0, component]
    for flip in range(3):
        flipped = separation.clone()
        flipped[:, flip] *= -1.0
        there = vie.green_k(flipped, WAVENUMBER)[0, component]
        expected = -here if flip == axis else here
        assert torch.allclose(there, expected)


def test_the_dyadic_kernel_is_traceless_away_from_the_source(device):
    """The double curl of the free-space Green function has zero trace off-source.

    Its trace is ``-2 (Gxx + Gyy + Gzz)``, and the scalar Green function obeys
    the Helmholtz equation, so the trace reduces to ``2 k**2 G``.
    """
    separation = _offsets([[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]], device=device)
    kernel = vie.green_n(separation, WAVENUMBER)
    trace = kernel[:, 0] + kernel[:, 3] + kernel[:, 5]

    r = torch.linalg.vector_norm(separation, dim=-1)
    scalar = torch.exp(-1j * WAVENUMBER * r) / (4.0 * torch.pi * r)
    assert torch.allclose(trace, 2.0 * WAVENUMBER**2 * scalar, rtol=1e-12)


@pytest.mark.parametrize("kernel", ["n", "k"])
def test_the_volume_rule_converges_as_its_order_rises(kernel, device):
    integrate = vie.volume_volume_n if kernel == "n" else vie.volume_volume_k
    offsets = _offsets([[2.0, 0.0, 0.0], [3.0, 1.0, 2.0]], device=device)
    reference = integrate(offsets, RESOLUTION, WAVENUMBER, order=10)

    coarse = torch.linalg.vector_norm(
        integrate(offsets, RESOLUTION, WAVENUMBER, order=2) - reference
    )
    fine = torch.linalg.vector_norm(
        integrate(offsets, RESOLUTION, WAVENUMBER, order=6) - reference
    )
    assert fine < coarse / 100.0


@pytest.mark.parametrize("kernel", ["n", "k"])
def test_a_distant_voxel_pair_integrates_to_the_kernel_times_the_volume(kernel, device):
    """Far apart, the kernel is nearly constant over the pair of voxels."""
    integrate = vie.volume_volume_n if kernel == "n" else vie.volume_volume_k
    green = vie.green_n if kernel == "n" else vie.green_k
    offsets = _offsets([[60.0, 0.0, 0.0]], device=device)

    got = integrate(offsets, RESOLUTION, WAVENUMBER, order=4)
    midpoint = green(offsets, WAVENUMBER) * RESOLUTION**6
    assert torch.allclose(got, midpoint, rtol=2e-3)


@pytest.mark.parametrize("kernel", ["n", "k"])
@pytest.mark.parametrize("linear", [False, True])
def test_the_reduced_rule_is_closer_to_the_converged_integral_than_marie_s(
    kernel, linear
):
    """The overlap rule reaches the product rule's integral, with fewer points."""
    green = vie.green_n if kernel == "n" else vie.green_k
    n_components = 6 if kernel == "n" else 3
    offsets = _offsets([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    reference = vie._volume_volume_product(
        offsets, RESOLUTION, WAVENUMBER, 9, green, n_components, linear
    )
    for order in (2, 4):
        product = vie._volume_volume_product(
            offsets, RESOLUTION, WAVENUMBER, order, green, n_components, linear
        )
        reduced = vie._volume_volume(
            offsets, RESOLUTION, WAVENUMBER, order, green, n_components, linear
        )
        product_error = torch.linalg.vector_norm(product - reference)
        assert torch.linalg.vector_norm(reduced - reference) <= product_error


@pytest.mark.parametrize("kernel", ["n", "k"])
def test_the_compiled_volume_rule_matches_the_torch_one(kernel):
    green = vie.green_n if kernel == "n" else vie.green_k
    n_components = 6 if kernel == "n" else 3
    offsets = _offsets([[1.0, 1.0, 0.0], [2.0, -1.0, 3.0], [0.0, 0.0, 5.0]])
    compiled = vie._volume_volume(
        offsets, RESOLUTION, WAVENUMBER, 3, green, n_components, linear=True
    )

    nodes, weights = vie._overlap_rule(3, offsets.device, offsets.dtype)
    grid = RESOLUTION * torch.cartesian_prod(nodes, nodes, nodes)
    pair_weight = torch.stack(
        [
            (
                weights[(test == 1) + 2 * (basis == 1)][:, None, None]
                * weights[(test == 2) + 2 * (basis == 2)][None, :, None]
                * weights[(test == 3) + 2 * (basis == 3)][None, None, :]
            ).reshape(-1)
            for test, basis in vie.PAIRS
        ]
    )
    reference = vie._volume_volume_torch(
        offsets,
        grid,
        pair_weight,
        RESOLUTION**6,
        WAVENUMBER,
        green,
        n_components,
        True,
    )
    scale = reference.abs().max()
    assert (compiled - reference).abs().max() <= 1e-12 * scale


def test_the_volume_rule_refuses_an_offset_of_zero(device):
    with pytest.raises(ValueError, match="diverges at a zero offset"):
        vie.volume_volume_n(
            _offsets([[0.0, 0.0, 0.0]], device=device), RESOLUTION, WAVENUMBER
        )


def test_the_volume_rule_refuses_offsets_of_the_wrong_shape(device):
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        vie.volume_volume_n(
            torch.ones(4, dtype=torch.float64, device=device), RESOLUTION, WAVENUMBER
        )


def _dense_operator(kernel, shape, index, sign):
    """Build the dense operator a stored kernel stands for, three by three."""
    cells = list(itertools.product(*(range(n) for n in shape)))
    size = len(cells)
    dense = torch.zeros((3 * size, 3 * size), device=kernel.device, dtype=kernel.dtype)
    parity = tucker._PARITY_XX_TO_ZZ if kernel.shape[-1] == 6 else tucker._PARITY_X_TO_Z

    for row_cell, observer in enumerate(cells):
        for column_cell, source in enumerate(cells):
            offset = [a - b for a, b in zip(observer, source, strict=True)]
            for row in range(3):
                for column in range(3):
                    which = index[row][column]
                    if which is None:
                        continue
                    value = kernel[
                        abs(offset[0]), abs(offset[1]), abs(offset[2]), which
                    ]
                    for axis in range(3):
                        if offset[axis] < 0:
                            value = value * parity[axis][which]
                    scale = 1.0 if sign is None else sign[row][column]
                    dense[3 * row_cell + row, 3 * column_cell + column] = scale * value
    return dense


def _stored_kernel(shape, n_components, device, seed=0):
    """Return a smooth, non-singular kernel of the right parity per offset.

    The physical kernel diverges at a zero offset, which the surface-surface
    treatment handles; the FFT path is a property of the storage, so it is
    checked here on a kernel that is finite everywhere.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    grids = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij"
    )
    offsets = torch.stack(grids, dim=-1)
    parity = tucker._PARITY_XX_TO_ZZ if n_components == 6 else tucker._PARITY_X_TO_Z

    components = []
    for component in range(n_components):
        weights = torch.randn(3, generator=generator, dtype=torch.float64)
        decay = torch.exp(-(offsets**2).sum(dim=-1) / 8.0)
        smooth = decay * (offsets * weights).sum(dim=-1)
        # Give the component a factor that is odd along each axis it must be odd
        # along, so the stored half determines the whole operator.
        for axis in range(3):
            if parity[axis][component] < 0:
                smooth = smooth * offsets[..., axis]
        components.append(torch.complex(smooth, 0.5 * smooth))
    return torch.stack(components, dim=-1).to(device)


@pytest.mark.parametrize(
    ("n_components", "index", "sign"),
    [(6, vie._DYADIC_INDEX, None), (3, vie._CURL_INDEX, vie._CURL_SIGN)],
    ids=["n", "k"],
)
def test_the_fft_path_reproduces_the_dense_operator(n_components, index, sign, device):
    shape = (3, 4, 2)
    kernel = _stored_kernel(shape, n_components, device)
    symbols = tucker.circulant_tucker(kernel, tol=None)

    generator = torch.Generator(device="cpu").manual_seed(5)
    current = torch.complex(
        torch.randn((3, *shape), generator=generator, dtype=torch.float64),
        torch.randn((3, *shape), generator=generator, dtype=torch.float64),
    ).to(device)

    apply = vie.apply_n if n_components == 6 else vie.apply_k
    got = apply(symbols, current)

    dense = _dense_operator(kernel, shape, index, sign)
    # The dense operator orders its unknowns cell by cell, three components at a
    # time; the field orders them component by component over the grid.
    flat = current.reshape(3, -1).transpose(0, 1).reshape(-1)
    expected = (dense @ flat).reshape(-1, 3).transpose(0, 1).reshape(3, *shape)
    assert torch.allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize(
    ("n_components", "index", "sign"),
    [(6, vie._DYADIC_INDEX, None), (3, vie._CURL_INDEX, vie._CURL_SIGN)],
    ids=["n", "k"],
)
def test_the_compiled_product_reproduces_the_dense_operator(
    n_components, index, sign, monkeypatch
):
    monkeypatch.setattr(vie, "_COMPILED_MIN_CELLS", 1)
    shape = (3, 4, 2)
    kernel = _stored_kernel(shape, n_components, "cpu")
    symbols = tucker.circulant_tucker(kernel, tol=None)
    generator = torch.Generator(device="cpu").manual_seed(5)
    current = torch.complex(
        torch.randn((3, *shape), generator=generator, dtype=torch.float64),
        torch.randn((3, *shape), generator=generator, dtype=torch.float64),
    )
    assert vie._compiled(current, symbols[0].shape)

    apply = vie.apply_n if n_components == 6 else vie.apply_k
    got = apply(symbols, current)

    dense = _dense_operator(kernel, shape, index, sign)
    flat = current.reshape(3, -1).transpose(0, 1).reshape(-1)
    expected = (dense @ flat).reshape(-1, 3).transpose(0, 1).reshape(3, *shape)
    assert torch.allclose(got, expected, atol=1e-10)


def _random_symbols(padded, per_set, n_sets, seed, device="cpu"):
    """Tucker symbols of distinct ranks, one tuple per set, or one tuple if ``n_sets`` is 0."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def draw(*shape):
        return torch.complex(
            torch.randn(shape, generator=generator, dtype=torch.float64),
            torch.randn(shape, generator=generator, dtype=torch.float64),
        ).to(device)

    def one(rank):
        return tucker.CirculantSymbol(
            core=draw(*rank),
            factors=tuple(draw(r, n) for r, n in zip(rank, padded, strict=True)),
        )

    ranks = [(2, 3, 4), (3, 1, 2), (4, 2, 3)]
    if n_sets == 0:
        return tuple(one(ranks[i % 3]) for i in range(per_set))
    return tuple(
        tuple(one(ranks[(i + j) % 3]) for j in range(per_set)) for i in range(n_sets)
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.complex128, 1e-12), (torch.complex64, 1e-5)],
    ids=["double", "single"],
)
@pytest.mark.parametrize("linear", [False, True], ids=["constant", "linear"])
@pytest.mark.parametrize("curl", [False, True], ids=["n", "k"])
def test_the_compiled_product_matches_the_torch_one_across_a_port_axis(
    linear, curl, dtype, tolerance, monkeypatch
):
    shape = (4, 3, 5)
    padded = tuple(tucker.transform_length(n) for n in shape)
    symbols = _random_symbols(
        padded, 3 if curl else 6, len(vie.PAIRS) if linear else 0, seed=3
    )
    generator = torch.Generator(device="cpu").manual_seed(9)
    n_components = 12 if linear else 3
    current = torch.complex(
        torch.randn(
            (2, n_components, *shape), generator=generator, dtype=torch.float64
        ),
        torch.randn(
            (2, n_components, *shape), generator=generator, dtype=torch.float64
        ),
    )
    apply = vie.apply_k if curl else vie.apply_n

    monkeypatch.setattr(vie, "_COMPILED_MIN_CELLS", 1)
    compiled = apply(symbols, current)
    monkeypatch.setattr(vie, "_COMPILED_MIN_CELLS", 10**12)
    reference = apply(symbols, current)
    torch.testing.assert_close(compiled, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("linear", [False, True], ids=["constant", "linear"])
@pytest.mark.parametrize("curl", [False, True], ids=["n", "k"])
def test_a_unit_response_is_the_product_applied_to_a_unit_current(linear, curl, device):
    shape = (5, 5, 5)
    padded = tuple(tucker.transform_length(n) for n in shape)
    symbols = _random_symbols(
        padded, 3 if curl else 6, len(vie.PAIRS) if linear else 0, seed=4, device=device
    )
    offsets = torch.tensor([[1, 2, 3], [2, 2, 2], [4, 0, 1]], device=device)
    n_components = 12 if linear else 3
    n_offsets = offsets.shape[0]

    sources = torch.zeros(
        (n_components, n_offsets, n_components, *shape),
        dtype=torch.complex128,
        device=device,
    )
    for component in range(n_components):
        for place, (i, j, k) in enumerate(offsets.tolist()):
            sources[component, place, component, i, j, k] = 1.0
    sources = sources.reshape(n_components * n_offsets, n_components, *shape)
    apply = vie.apply_k if curl else vie.apply_n
    expected = apply(symbols, sources).reshape(
        n_components * n_offsets, n_components, -1
    )

    got = vie.unit_responses(symbols, offsets, shape, curl=curl)
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_the_fft_path_carries_a_leading_port_axis(device):
    shape = (3, 4, 2)
    symbols = tucker.circulant_tucker(_stored_kernel(shape, 6, device), tol=None)
    generator = torch.Generator(device="cpu").manual_seed(7)
    currents = torch.complex(
        torch.randn((2, 3, *shape), generator=generator, dtype=torch.float64),
        torch.randn((2, 3, *shape), generator=generator, dtype=torch.float64),
    ).to(device)

    together = vie.apply_n(symbols, currents)
    assert together.shape == (2, 3, *shape)
    assert torch.allclose(together[1], vie.apply_n(symbols, currents[1]))


def test_compressing_the_kernel_changes_the_operator_within_the_tolerance(device):
    """PLAN.md's criterion for the compressed body operator."""
    shape = (4, 4, 4)
    kernel = _stored_kernel(shape, 6, device)
    generator = torch.Generator(device="cpu").manual_seed(11)
    current = torch.complex(
        torch.randn((3, *shape), generator=generator, dtype=torch.float64),
        torch.randn((3, *shape), generator=generator, dtype=torch.float64),
    ).to(device)

    exact = vie.apply_n(tucker.circulant_tucker(kernel, tol=None), current)
    tol = 1e-7
    compressed = vie.apply_n(tucker.circulant_tucker(kernel, tol=tol), current)
    error = torch.linalg.vector_norm(compressed - exact)
    assert error <= tol * torch.linalg.vector_norm(exact)


def test_the_mass_matrix_and_its_inverse_undo_each_other(device):
    current = torch.ones((3, 2, 2, 2), dtype=torch.complex128, device=device)
    assert torch.allclose(
        vie.apply_inverse_g(vie.apply_g(current, RESOLUTION), RESOLUTION), current
    )
    assert torch.allclose(vie.apply_g(current, RESOLUTION), RESOLUTION**3 * current)


def test_the_operators_refuse_the_wrong_number_of_symbols(device):
    symbols = tucker.circulant_tucker(_stored_kernel((2, 2, 2), 6, device), tol=None)
    with pytest.raises(ValueError, match="three symbols"):
        vie.apply_k(symbols, torch.zeros((3, 2, 2, 2), dtype=torch.complex128))
    with pytest.raises(ValueError, match="six symbols"):
        vie.apply_n(symbols[:3], torch.zeros((3, 2, 2, 2), dtype=torch.complex128))
