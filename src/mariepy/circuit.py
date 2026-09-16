"""Lumped circuits connected to a multiport: tuning, matching and calibration.

Ported from MARIE 3.0's ``src_physics/src_electronics/src_electronics/
co_simulation`` (the copy that adds each matching stage only to the ports that
carry it), ``co_sim_cost_functions/{tuning,match_optim}.m``,
``co_sim_calibration/{tune,match,calibration_matching}.m``, and the network
transforms under ``src_network_parameters/src_network_parameter_transforms``.

A coil solved with every tunable element opened into a port gives a multiport
admittance ``Y``. Its ports split into the coil's own ports, where matching
networks sit, and element ports, where tuning elements are placed. Placing the
tuning elements adds their admittances to the element ports, and eliminating
those ports leaves the admittance the coil ports see. The matching network is
then added stage by stage: a parallel stage adds to the admittance, a series
stage to the impedance.

Every function here works on small dense matrices. The placing functions also
take a leading batch of them, one per candidate or per frequency, with element
values and the angular frequency broadcast against that batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "MutualPair",
    "Stage",
    "abcd_to_s",
    "coil_voltage",
    "decoupling_weights",
    "element_admittance",
    "matching_calibration",
    "place_matching",
    "place_tuning",
    "reduce",
    "s_to_abcd",
    "toeplitz",
    "tuning_calibration",
    "z_to_s",
]

PARALLEL_LOADS = ("capacitorParallel", "inductorParallel", "resistorParallel")
SERIES_LOADS = ("capacitorSeries", "inductorSeries", "resistorSeries")


def z_to_s(impedance: torch.Tensor, reference: float) -> torch.Tensor:
    """Solve ``(Z + z0 I) S = Z - z0 I``, as ``np_z2s.m`` does.

    Parameters
    ----------
    impedance
        Square, complex.
    reference
        Reference impedance in ohms.

    Returns
    -------
    torch.Tensor
        The scattering matrix.
    """
    eye = torch.eye(impedance.shape[-1], dtype=impedance.dtype, device=impedance.device)
    return torch.linalg.solve(impedance + reference * eye, impedance - reference * eye)


def s_to_abcd(scattering: torch.Tensor, reference: float) -> torch.Tensor:
    """Convert a ``2m``-port scattering matrix to its ``m``-by-``m`` block ABCD form.

    Ported from ``np_s2abcd.m``, whose single-pair branch is the ``m = 1`` case
    of the block formula.

    Parameters
    ----------
    scattering
        Shape ``(2m, 2m)``.
    reference
        Reference impedance in ohms.

    Returns
    -------
    torch.Tensor
        Shape ``(2m, 2m)``: ``[[A, B], [C, D]]``.
    """
    m = scattering.shape[0] // 2
    eye = torch.eye(m, dtype=scattering.dtype, device=scattering.device)
    s11, s12 = scattering[:m, :m], scattering[:m, m:]
    s21, s22 = scattering[m:, :m], scattering[m:, m:]

    def through(left, right):
        return (
            torch.linalg.solve(s21.transpose(0, 1), left.transpose(0, 1)).transpose(
                0, 1
            )
            @ right
        )

    a = 0.5 * (through(eye + s11, eye - s22) + s12)
    b = reference * 0.5 * (through(eye + s11, eye + s22) - s12)
    c = 0.5 / reference * (through(eye - s11, eye - s22) - s12)
    d = 0.5 * (through(eye - s11, eye + s22) + s12)
    return torch.cat([torch.cat([a, b], dim=1), torch.cat([c, d], dim=1)], dim=0)


def abcd_to_s(abcd: torch.Tensor, reference: float) -> torch.Tensor:
    """Convert a block ABCD matrix back to scattering parameters.

    Ported from ``np_abcd2s.m``.

    Parameters
    ----------
    abcd
        Shape ``(2m, 2m)``.
    reference
        Reference impedance in ohms.

    Returns
    -------
    torch.Tensor
        Shape ``(2m, 2m)``.
    """
    m = abcd.shape[0] // 2
    eye = torch.eye(m, dtype=abcd.dtype, device=abcd.device)
    a, b = abcd[:m, :m], abcd[:m, m:] / reference
    c, d = abcd[m:, :m] * reference, abcd[m:, m:]
    denominator = a + b + c + d

    def right_divide(left):
        return torch.linalg.solve(
            denominator.transpose(0, 1), left.transpose(0, 1)
        ).transpose(0, 1)

    s11 = right_divide(a + b - c - d)
    s12 = ((a - b - c + d) - right_divide(a + b - c - d) @ (a - b + c - d)) / 2
    s21 = right_divide(2 * eye)
    s22 = torch.linalg.solve(denominator, -a + b - c + d)
    return torch.cat(
        [torch.cat([s11, s12], dim=1), torch.cat([s21, s22], dim=1)], dim=0
    )


def element_admittance(
    load: str,
    value: torch.Tensor,
    omega: float,
    quality: torch.Tensor | None = None,
) -> torch.Tensor:
    """Give the admittance a tuning element puts across its port.

    Ported from ``tuning.m`` (without loss) and ``tune.m`` (with loss). A
    mutual inductor contributes its self inductance here; its coupling to its
    partner is placed by :func:`place_tuning`.

    Parameters
    ----------
    load
        ``"capacitor"``, ``"inductor"``, ``"resistor"`` or ``"mutual_inductor"``.
    value
        Capacitance, inductance or resistance.
    omega
        Angular frequency in rad/s.
    quality
        Quality factor, or None for a lossless element.

    Returns
    -------
    torch.Tensor
        The admittance in siemens.

    Raises
    ------
    ValueError
        If the load is none of the four.
    """
    value = torch.as_tensor(value, dtype=torch.complex128)
    if load == "resistor":
        return 1.0 / value
    if load == "capacitor":
        reactance = 1.0 / (1j * omega * value)
        loss = 0.0 if quality is None else 1.0 / (omega * value * quality)
        return 1.0 / (reactance + loss)
    if load in ("inductor", "mutual_inductor"):
        reactance = 1j * omega * value
        loss = 0.0 if quality is None else omega * value / quality
        return 1.0 / (reactance + loss)
    raise ValueError(f"unknown tuning load {load!r}")


@dataclass(frozen=True)
class MutualPair:
    """Two coupled inductors among the element ports.

    Attributes
    ----------
    first, second
        Their element-port indices.
    first_inductance, second_inductance
        Their self inductances in henries.
    coefficient
        The coupling coefficient ``k``: the mutual inductance over the root of
        the product of the two self inductances.
    """

    first: int
    second: int
    first_inductance: torch.Tensor
    second_inductance: torch.Tensor
    coefficient: torch.Tensor


def place_tuning(
    admittance: torch.Tensor,
    ports: list[int],
    loads: list[str],
    values: torch.Tensor,
    omega: float,
    *,
    quality: torch.Tensor | None = None,
    pairs: tuple[MutualPair, ...] = (),
    without_resistors: bool = False,
) -> torch.Tensor:
    """Place the tuning elements across their element ports.

    Ported from ``tuning.m`` and ``tune.m``. Each element's admittance joins
    its own diagonal entry; each mutual pair adds ``1 / (j omega k sqrt(L1 L2))``
    to the two entries that join the pair, as MARIE writes it.

    Parameters
    ----------
    admittance
        The multiport admittance, shape ``(..., n, n)``.
    ports
        The element port of each element.
    loads, values
        Each element's load type, and its value, shape ``(..., n_elements)``.
    omega
        Angular frequency in rad/s, a float or a tensor broadcast against the
        batch.
    quality
        Each element's quality factor, or None for lossless elements.
    pairs
        The mutual couplings among the elements.
    without_resistors
        Leave the resistors out, as ``tune.m`` does for the lossless copy it
        measures the elements' own dissipation against.

    Returns
    -------
    torch.Tensor
        A new admittance with the elements placed.
    """
    placed = admittance.clone()
    for index, (port, load) in enumerate(zip(ports, loads, strict=True)):
        if without_resistors and load == "resistor":
            continue
        q = None if quality is None else quality[index]
        placed[..., port, port] = placed[..., port, port] + element_admittance(
            load, values[..., index], omega, q
        )
    for pair in pairs:
        coupling = 1.0 / (
            1j
            * omega
            * pair.coefficient
            * torch.sqrt(pair.first_inductance * pair.second_inductance)
        )
        placed[..., pair.first, pair.second] += coupling
        placed[..., pair.second, pair.first] += coupling
    return placed


def reduce(admittance: torch.Tensor, keep: list[int], drop: list[int]) -> torch.Tensor:
    """Eliminate ports carrying no current: the Schur complement onto the kept ports.

    Parameters
    ----------
    admittance
        Shape ``(..., n, n)``.
    keep, drop
        Port indices.

    Returns
    -------
    torch.Tensor
        ``Y_kk - Y_kd Y_dd^-1 Y_dk``.
    """
    keep = torch.as_tensor(keep, dtype=torch.long, device=admittance.device)
    drop = torch.as_tensor(drop, dtype=torch.long, device=admittance.device)
    rows_kept = admittance.index_select(-2, keep)
    y_kk = rows_kept.index_select(-1, keep)
    if drop.numel() == 0:
        return y_kk
    rows_dropped = admittance.index_select(-2, drop)
    y_kd = rows_kept.index_select(-1, drop)
    y_dk = rows_dropped.index_select(-1, keep)
    y_dd = rows_dropped.index_select(-1, drop)
    return y_kk - y_kd @ torch.linalg.solve(y_dd, y_dk)


@dataclass(frozen=True)
class Stage:
    """One stage of the matching networks, across every coil port.

    Attributes
    ----------
    loads
        Each port's element in this stage, one of :data:`PARALLEL_LOADS` or
        :data:`SERIES_LOADS`, or ``""`` where the port has no element here.
    values
        Each port's element value, zero where it has none, shape
        ``(..., n_ports)``.
    quality
        Each port's quality factor, or None for lossless elements.
    """

    loads: tuple[str, ...]
    values: torch.Tensor
    quality: torch.Tensor | None = None


def place_matching(
    admittance: torch.Tensor,
    stages: tuple[Stage, ...],
    omega: float,
    *,
    lossless: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add the matching networks, stage by stage, and return the port impedance.

    Ported from ``match_optim.m`` (without loss) and ``match.m`` (with loss),
    in the copy that adds each stage's elements only to the ports that carry
    them. Within a stage the parallel elements join the admittance first, then
    the series elements join the impedance.

    Parameters
    ----------
    admittance
        The coil ports' admittance, shape ``(..., n_ports, n_ports)``.
    stages
        The stages, from the coil outwards.
    omega
        Angular frequency in rad/s, a float or a tensor broadcast against the
        batch.
    lossless
        A second admittance carried alongside with lossless elements, as
        ``match.m`` carries ``YPm_loss``, or None.

    Returns
    -------
    impedance : torch.Tensor
        The impedance at the network inputs.
    lossless_impedance : torch.Tensor or None
        The same for the lossless copy.
    """
    y = admittance.clone()
    y_lossless = None if lossless is None else lossless.clone()
    for stage in stages:
        loads = stage.loads
        values = stage.values.to(torch.complex128)
        q = stage.quality
        parallel = [port for port, load in enumerate(loads) if load in PARALLEL_LOADS]
        series = [port for port, load in enumerate(loads) if load in SERIES_LOADS]
        for port in parallel:
            element, element_lossless = _parallel(
                loads[port], values[..., port], omega, q, port
            )
            y[..., port, port] += element
            if y_lossless is not None and element_lossless is not None:
                y_lossless[..., port, port] += element_lossless
        if series:
            z = torch.linalg.inv(y)
            z_lossless = None if y_lossless is None else torch.linalg.inv(y_lossless)
            for port in series:
                element, element_lossless = _series(
                    loads[port], values[..., port], omega, q, port
                )
                z[..., port, port] += element
                if z_lossless is not None and element_lossless is not None:
                    z_lossless[..., port, port] += element_lossless
            y = torch.linalg.inv(z)
            if z_lossless is not None:
                y_lossless = torch.linalg.inv(z_lossless)
    return (
        torch.linalg.inv(y),
        None if y_lossless is None else torch.linalg.inv(y_lossless),
    )


def _parallel(load, value, omega, quality, port):
    """Return a parallel element's admittance, with its loss and without."""
    if load == "resistorParallel":
        return 1.0 / value, None
    if load == "capacitorParallel":
        reactance = 1.0 / (1j * omega * value)
        if quality is None:
            return 1.0 / reactance, None
        loss = 1.0 / (omega * value * quality[port])
        return 1.0 / (reactance + loss), 1.0 / reactance
    reactance = 1j * omega * value
    if quality is None:
        return 1.0 / reactance, None
    loss = omega * value / quality[port]
    return 1.0 / (reactance + loss), 1.0 / reactance


def _series(load, value, omega, quality, port):
    """Return a series element's impedance, with its loss and without."""
    if load == "resistorSeries":
        return value, None
    if load == "capacitorSeries":
        reactance = 1.0 / (1j * omega * value)
        if quality is None:
            return reactance, None
        loss = 1.0 / (omega * value * quality[port])
        return reactance + loss, reactance
    reactance = 1j * omega * value
    if quality is None:
        return reactance, None
    loss = omega * value / quality[port]
    return reactance + loss, reactance


def coil_voltage(
    admittance: torch.Tensor,
    stages: tuple[Stage, ...],
    omega,
    reference: float,
) -> torch.Tensor:
    """Map the waves incident on the matching networks' inputs to the coil voltages.

    Each port's network is a chain of two-ports, a shunt admittance ``y`` with
    ABCD matrix ``[[1, 0], [y, 1]]`` and a series impedance ``z`` with
    ``[[1, z], [0, 1]]``, so the input side is ``A V + B I`` and ``C V + D I``
    of the coil side. A source of the line impedance sending the wave ``a``
    sets ``V_in + z0 I_in = 2 sqrt(z0) a``, and the coil draws
    ``I = Y V``, whence ``V = 2 sqrt(z0) [(A + z0 C) + (B + z0 D) Y]^-1 a``
    with the diagonal chain matrices.

    Parameters
    ----------
    admittance
        The coil ports' admittance, shape ``(..., n, n)``.
    stages
        The matching stages, from the coil outwards, as for
        :func:`place_matching`; their quality factors, if given, set the loss.
    omega
        Angular frequency in rad/s, a float or a tensor broadcast against the
        batch.
    reference
        Line impedance in ohms.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n, n)``: column ``p`` holds the coil voltages when a unit
        wave arrives at port ``p`` and none at the others.
    """
    n = admittance.shape[-1]
    batch = admittance.shape[:-2]
    one = torch.ones((*batch, n), dtype=admittance.dtype, device=admittance.device)
    a, b = one.clone(), torch.zeros_like(one)
    c, d = torch.zeros_like(one), one.clone()
    for stage in stages:
        values = stage.values.to(torch.complex128)
        q = stage.quality
        for port, load in enumerate(stage.loads):
            if load in PARALLEL_LOADS:
                y, _ = _parallel(load, values[..., port], omega, q, port)
                # [[1, 0], [y, 1]] @ [[a, b], [c, d]]
                c[..., port] = c[..., port] + y * a[..., port]
                d[..., port] = d[..., port] + y * b[..., port]
            elif load in SERIES_LOADS:
                z, _ = _series(load, values[..., port], omega, q, port)
                # [[1, z], [0, 1]] @ [[a, b], [c, d]]
                a[..., port] = a[..., port] + z * c[..., port]
                b[..., port] = b[..., port] + z * d[..., port]
    system = (
        torch.diag_embed(a + reference * c)
        + (b + reference * d)[..., :, None] * admittance
    )
    eye = torch.eye(n, dtype=admittance.dtype, device=admittance.device)
    return 2.0 * reference**0.5 * torch.linalg.solve(system, eye.expand_as(system))


def tuning_calibration(
    admittance: torch.Tensor, coil_ports: list[int], element_ports: list[int]
) -> torch.Tensor:
    """Map the coil ports' voltages to every port's voltage, elements placed.

    Ported from ``tune.m``: with the element ports carrying no external
    current, the voltage on every port is ``Z[:, M] Z[M, M]^-1`` times the coil
    ports' voltages.

    Parameters
    ----------
    admittance
        The multiport admittance with the tuning elements placed.
    coil_ports, element_ports
        The two kinds of port.

    Returns
    -------
    torch.Tensor
        Shape ``(n_ports, n_coil_ports)``, rows in the multiport's own order.
    """
    impedance = torch.linalg.inv(admittance)
    n = admittance.shape[0]
    mapping = torch.zeros(
        (n, len(coil_ports)), dtype=admittance.dtype, device=admittance.device
    )
    columns = impedance[:, coil_ports]
    mapping[coil_ports] = columns[coil_ports]
    mapping[element_ports] = columns[element_ports]
    square = impedance[coil_ports][:, coil_ports]
    return torch.linalg.solve(
        square.transpose(0, 1), mapping.transpose(0, 1)
    ).transpose(0, 1)


def _matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    """Return the principal square root of a Hermitian matrix.

    MARIE takes it through a Schur decomposition, which for a Hermitian matrix
    is its eigendecomposition.
    """
    values, vectors = torch.linalg.eigh(matrix)
    roots = torch.sqrt(values.to(matrix.dtype))
    return (vectors * roots[None, :]) @ vectors.mH


def matching_calibration(
    unmatched: torch.Tensor, matched: torch.Tensor, reference: float
) -> torch.Tensor:
    """Map the waves incident at the matching networks' inputs through to the coil.

    Ported from ``calibration_matching.m``. The networks are recovered as a
    two-port ``Sb`` from the matched scattering matrix, cascaded with a
    two-port ``Sa`` built from the unmatched one, and the transmission of the
    cascade into the coil, ``(I - Sc22 Ss)^-1 Sc21``, is returned.

    Parameters
    ----------
    unmatched
        The coil ports' scattering matrix with the tuning elements placed,
        MARIE's ``SPs``.
    matched
        The same with the matching networks added, MARIE's ``SPm``.
    reference
        Reference impedance in ohms.

    Returns
    -------
    torch.Tensor
        Square, of the coil ports' size.
    """
    n = unmatched.shape[0]
    eye = torch.eye(n, dtype=unmatched.dtype, device=unmatched.device)

    sa22 = unmatched.mH
    sa21 = _matrix_sqrt(eye - unmatched.mH @ unmatched)
    sa12 = sa21.transpose(0, 1)
    sa11 = -torch.linalg.solve(sa12.mH, unmatched) @ sa21
    sa = torch.cat([torch.cat([sa11, sa12], 1), torch.cat([sa21, sa22], 1)], 0)

    sb11 = matched
    sb21 = _matrix_sqrt(eye - matched.mH @ matched)
    sb12 = sb21.transpose(0, 1)
    sb22 = (
        torch.linalg.solve(sb21.transpose(0, 1), (sb12.mH @ sb11).transpose(0, 1))
        .transpose(0, 1)
        .mH
    )
    sb = torch.cat([torch.cat([sb11, sb12], 1), torch.cat([sb21, sb22], 1)], 0)

    cascade = abcd_to_s(s_to_abcd(sb, reference) @ s_to_abcd(sa, reference), reference)
    sc21 = cascade[n:, :n]
    sc22 = cascade[n:, n:]
    return torch.linalg.solve(eye - sc22 @ unmatched, sc21)


def toeplitz(first_row, first_column) -> torch.Tensor:
    """Build the Toeplitz matrix with the given first row and first column.

    MARIE's decoupling cost calls ``toeplitz``, which ``marie-tools`` does not
    define. This is the matrix it names: entry ``(i, j)`` is
    ``first_row[j - i]`` above the diagonal and ``first_column[i - j]`` below.

    Parameters
    ----------
    first_row, first_column
        Sequences of equal length, sharing their first entry.

    Returns
    -------
    torch.Tensor
        Square, float64.
    """
    row = torch.as_tensor(first_row, dtype=torch.float64)
    column = torch.as_tensor(first_column, dtype=torch.float64)
    n = row.numel()
    index = torch.arange(n)
    offset = index[None, :] - index[:, None]
    return torch.where(
        offset >= 0, row[offset.clamp(min=0)], column[(-offset).clamp(min=0)]
    )


def decoupling_weights(n_ports: int) -> torch.Tensor:
    """Return the weights MARIE's decoupling cost puts on the port impedance.

    Ported from ``matching_and_tuning_and_decoupling.m``: nearest and
    next-nearest neighbours, closed around the ring of ports, below the
    diagonal. With MARIE's weights, one for every pair, this is every entry
    strictly below the diagonal.

    Parameters
    ----------
    n_ports
        Number of coil ports.

    Returns
    -------
    torch.Tensor
        Shape ``(n_ports, n_ports)``, float64.
    """
    w1, w2, w3, w0 = 0.0, 1.0, 1.0, 1.0
    first = [w1, w2, w3] + [w0] * max(n_ports - 3, 0)
    weights = toeplitz(first, first)[:n_ports, :n_ports].clone()
    if n_ports > 2:
        weights[-1, 0] = w2
    if n_ports > 3:
        weights[-2, 0] = w3
    return torch.tril(weights, diagonal=-1)
