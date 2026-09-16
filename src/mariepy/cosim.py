"""Co-simulation: tune, match and decouple a coil's lumped elements, then calibrate.

Ported from MARIE 3.0's ``src_physics/src_electronics/src_electronics/
co_simulation``, for coils whose ports all share one role: transmit (``Tx``),
receive (``Rx``) or both (``TxRx``). Coils that mix roles are not supported yet.

A coil solved with MARIE's ``TMD`` flag set opens every optimisable element
into a port, so the port admittance the solver returns spans the coil's own
ports and those element ports. Here that admittance is closed again with
lumped values found in three searches, each MARIE's:

1. tuning, per coil entity: the element values that leave the entity's ports
   without reactance (``runner_T.m``);
2. matching and tuning, per entity: the matching networks as well, bringing
   the port resistance to the line impedance (``runner_M_T.m``);
3. all entities together, adding each side's terms: port-to-port coupling on
   the transmit side, the impedance each port shows its preamplifier with the
   others preamplifier-decoupled on the receive side (``runner_M_T_D.m``,
   ``runner_M_T_PD.m``, ``runner_M_T_D_PD.m``). With mixed roles, each role's
   rows are searched on their own first (``runner_M_T_D_PD_split.m``), then all
   together with the other side detuned (``runner_*_M_T_PD_DT_*_M_T_D.m``).

The searches use scipy's differential evolution where MARIE uses
``particleswarm``, with its restarts, growing population and narrowing
bounds. The values found then give the calibration: the voltage every
solver port carries per unit wave incident on each matched port, which
multiplies the solver's per-port currents and fields.

MARIE's masks become a :class:`Network` of :class:`Terminal` rows read once
from the element file, and a layout per search that says which values each
candidate vector supplies. The numerics are :mod:`mariepy.circuit`'s.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mariepy import circuit

__all__ = [
    "DETUNING_RESISTANCE",
    "PREAMPLIFIER_RESISTANCE",
    "CoSimulation",
    "Component",
    "Coupling",
    "Network",
    "Search",
    "Sweep",
    "Terminal",
    "calibrate",
    "co_simulate",
    "read_network",
    "sweep",
]

PREAMPLIFIER_RESISTANCE = 1500.0
"""MARIE's ``emc.Preamp_res``, in ohms."""

DETUNING_RESISTANCE = 1e5
"""MARIE's ``emc.Detune_res``, in ohms: what a port of the other role presents."""

ROLES = ("Tx", "Rx", "TxRx")


@dataclass(frozen=True)
class Component:
    """One lumped value: a tuning element, or one stage of a matching network.

    Attributes
    ----------
    load
        ``"capacitor"``, ``"inductor"``, ``"resistor"`` or
        ``"mutual_inductor"`` for a tuning element; one of
        :data:`mariepy.circuit.PARALLEL_LOADS` or
        :data:`mariepy.circuit.SERIES_LOADS` for a matching stage.
    value
        The file's value: the start of a search, or the value itself when the
        component is not searched.
    quality
        Quality factor, which sets the loss at calibration.
    minimum, maximum
        The search bounds the file gives.
    group
        The search variable the component takes, shared by components the
        file declares symmetric; None when the component is not searched.
    """

    load: str
    value: float
    quality: float
    minimum: float
    maximum: float
    group: int | None = None


@dataclass(frozen=True)
class Terminal:
    """One port of the solved coil: a driven port or an opened element.

    Attributes
    ----------
    number
        The element's number in the file.
    kind
        ``"port"`` for a coil port, which carries a matching network;
        ``"element"`` for a tuning element opened into a port.
    role
        ``"Tx"``, ``"Rx"`` or ``"TxRx"``.
    entities
        The coil entities it belongs to, numbered from 1 as the file numbers
        them.
    optimise
        Whether its values are searched.
    components
        A port's matching network, from the coil outwards, or an element's
        single value.
    """

    number: int
    kind: str
    role: str
    entities: tuple[int, ...]
    optimise: bool
    components: tuple[Component, ...]


@dataclass(frozen=True)
class Coupling:
    """A mutual coupling between two opened mutual inductors.

    Attributes
    ----------
    first, second
        Their rows in :attr:`Network.terminals`.
    mutual
        The file's mutual inductance in henries.
    minimum, maximum
        Search bounds of the coupling coefficient.
    group
        Its search variable.
    """

    first: int
    second: int
    mutual: float
    minimum: float
    maximum: float
    group: int


@dataclass(frozen=True)
class Network:
    """The ports of a solved coil and the lumped values that close them.

    Attributes
    ----------
    terminals
        One per row of the solver's port admittance, in the same order.
    couplings
        The mutual couplings among the terminals.
    tmd
        Whether the coil was solved with its optimisable elements opened,
        MARIE's ``TMD`` flag, and so whether the values are searched.
    """

    terminals: tuple[Terminal, ...]
    couplings: tuple[Coupling, ...] = ()
    tmd: bool = False

    @property
    def ports(self) -> list[int]:
        """Rows that are coil ports."""
        return [i for i, t in enumerate(self.terminals) if t.kind == "port"]

    @property
    def elements(self) -> list[int]:
        """Rows that are opened elements."""
        return [i for i, t in enumerate(self.terminals) if t.kind == "element"]

    @property
    def roles(self) -> set[str]:
        """Return the roles the ports take."""
        return {self.terminals[i].role for i in self.ports}

    @property
    def entities(self) -> list[int]:
        """Every entity number, in increasing order."""
        return sorted({e for t in self.terminals for e in t.entities})

    def rows_of(self, entity: int) -> list[int]:
        """Rows that belong to an entity."""
        return [i for i, t in enumerate(self.terminals) if entity in t.entities]

    def start(self) -> dict[int, float]:
        """Return the file's value of every search variable.

        A coupling coefficient starts at the file's mutual inductance over the
        root of the two file inductances.
        """
        values: dict[int, float] = {}
        for terminal in self.terminals:
            for component in terminal.components:
                if component.group is not None:
                    values.setdefault(component.group, component.value)
        for coupling in self.couplings:
            first = self.terminals[coupling.first].components[0].value
            second = self.terminals[coupling.second].components[0].value
            values.setdefault(
                coupling.group, _clamp(coupling.mutual / math.sqrt(first * second))
            )
        return values

    def values_of(self, result: CoSimulation) -> dict[int, float]:
        """Return every search variable's value in a result."""
        values = self.start()
        for terminal, row in zip(self.terminals, result.values, strict=True):
            for component, value in zip(terminal.components, row, strict=True):
                if component.group is not None:
                    values[component.group] = value
        values.update(result.couplings)
        return values


def read_network(path: str | Path, *, tmd: bool) -> Network:
    """Read the co-simulation settings from MARIE's JSON element file.

    Ported from ``geo_scoil_lumped_elements.m`` and the settings MARIE derives
    from it (``co_simulation_tuning_settings.m``,
    ``co_simulation_matching_tuning_settings.m``). The rows follow the solver's
    ports: every ``port``, and with ``tmd`` every element whose
    ``optim.boolean`` is set, in file order, as
    :func:`mariepy.coil.read_lumped_elements` opens them.

    Search variables are shared as MARIE shares them. A matching stage takes
    ``symmetry`` plus the count of matching stages before it; a tuning element
    takes ``symmetry`` plus the index of the last matching stage before it; a
    mutual coupling takes a variable of its own. Every other mutual inductor
    in file order heads a coupling with the partner its ``cross_talk`` names,
    and the coefficient's bounds follow from the mutual inductance and the two
    inductors' bounds. MARIE numbers the coupling variables from the count of
    searched values, which can meet a symmetry-offset number; here they are
    numbered past the largest other variable.

    Parameters
    ----------
    path
        The element file.
    tmd
        MARIE's ``TMD`` flag, as the coil was solved with.

    Returns
    -------
    Network
        The rows and their values.

    Raises
    ------
    ValueError
        If a role is not one of ``Tx``, ``Rx``, ``TxRx``.
    """
    elements = json.loads(Path(path).read_text())["coil_configuration"]["elements"]

    def optimised(element):
        return bool(tmd and (element.get("optim") or {}).get("boolean"))

    rows = [e for e in elements if e["type"] == "port" or optimised(e)]
    row_of = {int(e["number"]): i for i, e in enumerate(rows)}

    counter = 0
    last = 0
    terminals = []
    for element in rows:
        optim = element.get("optim") or {}
        excitation = element.get("excitation") or {}
        role = excitation.get("TxRx", "TxRx")
        if role not in ROLES:
            raise ValueError(f"element {element['number']} has role {role!r}")
        entities = tuple(int(e) for e in np.atleast_1d(excitation.get("entity", 1)))
        search = optimised(element)
        symmetry = int(optim.get("symmetry", 0) or 0)
        if element["type"] == "port":
            loads = _split(element["load"])
            values = _listed(element["value"], len(loads))
            qualities = _listed(element["Q"], len(loads))
            minima = _listed(optim.get("minim", 0.0), len(loads))
            maxima = _listed(optim.get("maxim", 0.0), len(loads))
            components = tuple(
                Component(
                    load=loads[i],
                    value=values[i],
                    quality=qualities[i],
                    minimum=minima[i],
                    maximum=maxima[i],
                    group=symmetry + counter + i if search else None,
                )
                for i in range(len(loads))
            )
            counter += len(loads)
            last = counter - 1
        else:
            components = (
                Component(
                    load=element["load"],
                    value=float(np.atleast_1d(element["value"])[0]),
                    quality=float(np.atleast_1d(element["Q"])[0]),
                    minimum=float(optim["minim"]),
                    maximum=float(optim["maxim"]),
                    group=symmetry + last,
                ),
            )
            counter += 1
        terminals.append(
            Terminal(
                number=int(element["number"]),
                kind=element["type"] if element["type"] == "port" else "element",
                role=role,
                entities=entities,
                optimise=search,
                components=components,
            )
        )

    largest = max(
        (c.group for t in terminals for c in t.components if c.group is not None),
        default=0,
    )
    couplings = []
    heads = 0
    mutual_seen = 0
    for row, element in enumerate(rows):
        if element["type"] != "port" and element["load"] == "mutual_inductor":
            mutual_seen += 1
            if mutual_seen % 2 == 0:
                continue
            heads += 1
            partner_number, mutual = _partner(element)
            partner = row_of[partner_number]
            own, other = terminals[row].components[0], terminals[partner].components[0]
            k1 = _clamp(mutual / math.sqrt(own.maximum * other.maximum))
            k2 = _clamp(mutual / math.sqrt(own.minimum * other.minimum))
            couplings.append(
                Coupling(
                    first=row,
                    second=partner,
                    mutual=mutual,
                    minimum=min(k1, k2),
                    maximum=max(k1, k2),
                    group=heads + largest,
                )
            )
    return Network(terminals=tuple(terminals), couplings=tuple(couplings), tmd=tmd)


def _split(load) -> list[str]:
    """Split a matching network's load string into its stages."""
    if isinstance(load, list):
        return [str(item) for item in load]
    return str(load).split("_")


def _listed(value, n: int) -> list[float]:
    """Return a per-stage list of floats from a scalar, a list or strings."""
    items = np.atleast_1d(np.asarray(value, dtype=object)).tolist()
    if len(items) == 1 and n > 1:
        items = items * n
    return [float(item) for item in items[:n]]


def _partner(element: dict) -> tuple[int, float]:
    """Return a mutual inductor's partner number and mutual inductance."""
    cross_talk = element.get("cross_talk")
    if isinstance(cross_talk, dict) and "coupled_port" in cross_talk:
        return int(cross_talk["coupled_port"]), float(cross_talk["coupled_value"])
    if isinstance(cross_talk, list) and len(cross_talk) == 2:
        return int(cross_talk[0]), float(cross_talk[1])
    raise ValueError(f"mutual inductor {element['number']} names no partner")


def _clamp(value: float) -> float:
    return max(min(value, 1.0), -1.0)


def _around(value: float, low: float, high: float) -> tuple[float, float]:
    """MARIE's narrowed bounds: the value scaled both ways, clamped to [-1, 1].

    The clamp is MARIE's, written for coupling coefficients and applied to
    every variable; it binds only for values beyond one, such as a resistance.
    """
    a, b = low * value, high * value
    return _clamp(min(a, b)), _clamp(max(a, b))


# --------------------------------------------------------------------------
# Layout: which values a candidate vector supplies
# --------------------------------------------------------------------------


@dataclass
class _Layout:
    """The rows one search closes and where each value comes from.

    Values are gathered from a candidate batch ``x``, shape ``(batch,
    n_variables)``: a slot of ``-1`` takes the fixed value instead.
    """

    admittance: torch.Tensor
    ports: list[int]
    elements: list[int]
    optimised: list[int]
    roles: list[str]
    element_loads: list[str]
    element_slots: torch.Tensor
    element_fixed: torch.Tensor
    element_quality: torch.Tensor
    stage_loads: list[tuple[str, ...]]
    stage_slots: list[torch.Tensor]
    stage_fixed: list[torch.Tensor]
    stage_quality: list[torch.Tensor]
    pairs: list[tuple[int, int, int, int, int]]
    groups: list[int]
    lower: np.ndarray
    upper: np.ndarray

    @property
    def n_variables(self) -> int:
        return len(self.groups)

    @classmethod
    def build(
        cls,
        network: Network,
        admittance: torch.Tensor,
        rows: list[int],
        *,
        matching: bool,
    ) -> _Layout:
        rows = sorted(rows)
        dropped = [i for i in range(len(network.terminals)) if i not in rows]
        reduced = circuit.reduce(admittance.to(torch.complex128).cpu(), rows, dropped)
        local = {row: i for i, row in enumerate(rows)}

        groups: list[int] = []
        bounds: list[tuple[float, float]] = []

        def slot(group, minimum, maximum):
            if group is None:
                return -1
            if group not in groups:
                groups.append(group)
                bounds.append((minimum, maximum))
            return groups.index(group)

        ports = [local[r] for r in rows if network.terminals[r].kind == "port"]
        elements = [local[r] for r in rows if network.terminals[r].kind == "element"]
        port_rows = [r for r in rows if network.terminals[r].kind == "port"]
        element_rows = [r for r in rows if network.terminals[r].kind == "element"]

        depth = max(
            (len(network.terminals[r].components) for r in port_rows), default=0
        )
        stage_loads, stage_slots, stage_fixed, stage_quality = [], [], [], []
        if matching:
            for s in range(depth):
                loads, slots, fixed, quality = [], [], [], []
                for r in port_rows:
                    chain = network.terminals[r].components
                    if s >= len(chain):
                        loads.append("")
                        slots.append(-1)
                        fixed.append(0.0)
                        quality.append(1.0)
                        continue
                    c = chain[s]
                    loads.append(c.load)
                    slots.append(slot(c.group, c.minimum, c.maximum))
                    fixed.append(c.value)
                    quality.append(c.quality)
                stage_loads.append(tuple(loads))
                stage_slots.append(torch.tensor(slots, dtype=torch.long))
                stage_fixed.append(torch.tensor(fixed, dtype=torch.float64))
                stage_quality.append(torch.tensor(quality, dtype=torch.float64))

        element_slots, element_fixed, element_quality, element_loads = [], [], [], []
        for r in element_rows:
            c = network.terminals[r].components[0]
            element_loads.append(c.load)
            element_slots.append(slot(c.group, c.minimum, c.maximum))
            element_fixed.append(c.value)
            element_quality.append(c.quality)

        pairs = []
        for coupling in network.couplings:
            if coupling.first in local and coupling.second in local:
                pairs.append(
                    (
                        local[coupling.first],
                        local[coupling.second],
                        element_rows.index(coupling.first),
                        element_rows.index(coupling.second),
                        slot(coupling.group, coupling.minimum, coupling.maximum),
                    )
                )

        optimised = [
            i for i, r in enumerate(port_rows) if network.terminals[r].optimise
        ]
        return cls(
            admittance=reduced,
            ports=ports,
            elements=elements,
            optimised=optimised,
            roles=[network.terminals[r].role for r in port_rows],
            element_loads=element_loads,
            element_slots=torch.tensor(element_slots, dtype=torch.long),
            element_fixed=torch.tensor(element_fixed, dtype=torch.float64),
            element_quality=torch.tensor(element_quality, dtype=torch.float64),
            stage_loads=stage_loads,
            stage_slots=stage_slots,
            stage_fixed=stage_fixed,
            stage_quality=stage_quality,
            pairs=pairs,
            groups=groups,
            lower=np.array([b[0] for b in bounds], dtype=np.float64),
            upper=np.array([b[1] for b in bounds], dtype=np.float64),
        )

    def vector(self, values: dict[int, float]) -> torch.Tensor:
        """Return one candidate holding the given value of each variable."""
        return torch.tensor([[values[g] for g in self.groups]], dtype=torch.float64)

    # -- closing the network ------------------------------------------------

    def tuned(
        self, x: torch.Tensor, omega, *, lossy: bool = False, lossless: bool = False
    ) -> torch.Tensor:
        """Return the full admittance with the tuning elements placed.

        With ``lossy`` the elements carry their quality factors; with
        ``lossless`` they carry none and the resistors are left out, as
        ``tune.m`` builds its two copies. Neither gives the search's copy.
        """
        batch = x.shape[0]
        values = _take(x, self.element_slots, self.element_fixed)
        pairs = tuple(
            circuit.MutualPair(
                first=a,
                second=b,
                first_inductance=values[:, ea],
                second_inductance=values[:, eb],
                coefficient=x[:, s],
            )
            for a, b, ea, eb, s in self.pairs
        )
        return circuit.place_tuning(
            self.admittance.expand(batch, -1, -1),
            self.elements,
            self.element_loads,
            values,
            _batched(omega),
            quality=self.element_quality if lossy else None,
            pairs=pairs,
            without_resistors=lossless,
        )

    def ports_admittance(self, placed: torch.Tensor) -> torch.Tensor:
        """Eliminate the element ports."""
        return circuit.reduce(placed, self.ports, self.elements)

    def stages(self, x: torch.Tensor, *, lossy: bool = False) -> tuple:
        """Return the matching stages with this candidate's values."""
        return tuple(
            circuit.Stage(
                loads=loads,
                values=_take(x, slots, fixed),
                quality=quality if lossy else None,
            )
            for loads, slots, fixed, quality in zip(
                self.stage_loads,
                self.stage_slots,
                self.stage_fixed,
                self.stage_quality,
                strict=True,
            )
        )


def _take(x: torch.Tensor, slots: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
    """Gather values from a candidate batch, with fixed values where a slot is -1."""
    if slots.numel() == 0:
        return torch.zeros((x.shape[0], 0), dtype=torch.float64)
    picked = x[:, slots.clamp(min=0)] if x.shape[1] else fixed.expand(x.shape[0], -1)
    return torch.where(slots >= 0, picked, fixed)


def _batched(omega):
    """Give a per-candidate angular frequency the batch's trailing shape."""
    if isinstance(omega, torch.Tensor) and omega.ndim:
        return omega
    return float(omega)


def _single_port_stages(stages: tuple, port: int) -> tuple:
    """One port's matching chain, as stages of a one-port network."""
    return tuple(
        circuit.Stage(
            loads=(stage.loads[port],),
            values=stage.values[..., port : port + 1],
            quality=None if stage.quality is None else stage.quality[port : port + 1],
        )
        for stage in stages
    )


# --------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------


def _diagonal(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1)


def _norm(values: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(values, dim=-1)


def _matching_cost(impedance, optimised, reference):
    """``matching_and_tuning.m``: port resistance at the line, no reactance."""
    diagonal = _diagonal(impedance)
    return _norm(diagonal[..., optimised].real - reference) + _norm(diagonal.imag)


def _decoupling_cost(impedance):
    """Weigh the port coupling as ``matching_and_tuning_and_decoupling.m`` does."""
    weights = circuit.decoupling_weights(impedance.shape[-1]).to(impedance.dtype)
    return torch.linalg.matrix_norm(weights * impedance)


def _preamplifier_loaded(admittance, port, resistance):
    """Terminate every other port in the preamplifier resistance."""
    n = admittance.shape[-1]
    rest = torch.tensor([i for i in range(n) if i != port], dtype=torch.long)
    loaded = admittance.clone()
    loaded[..., rest, rest] += 1.0 / resistance
    return loaded, rest.tolist()


def _receive_impedance(admittance, stages, omega, resistance):
    """``preamplifier_match_optim.m``: each port's input impedance, others decoupled."""
    impedances = []
    for port in range(admittance.shape[-1]):
        loaded, rest = _preamplifier_loaded(admittance, port, resistance)
        single = circuit.reduce(loaded, [port], rest)
        impedance, _ = circuit.place_matching(
            single, _single_port_stages(stages, port), omega
        )
        impedances.append(impedance[..., 0, 0])
    return torch.stack(impedances, dim=-1)


def _receive_cost(impedance, optimised, reference):
    """``matching_and_tuning_and_preamplifierdecoupling.m``."""
    return _norm(impedance[..., optimised].real - reference) + _norm(impedance.imag)


@dataclass(frozen=True)
class _Sides:
    """Say which of a layout's ports transmit, which receive, and which are detuned.

    Positions are among the layout's ports. ``decouple`` is off only where
    MARIE's own cost for transmit-only and transmit-and-receive ports together
    weighs no coupling.
    """

    transmit: list[int]
    receive: list[int]
    transmit_only: list[int]
    receive_only: list[int]
    decouple: bool

    @classmethod
    def of(cls, roles: list[str]) -> _Sides:
        present = set(roles)
        return cls(
            transmit=[i for i, r in enumerate(roles) if r in ("Tx", "TxRx")],
            receive=[i for i, r in enumerate(roles) if r in ("Rx", "TxRx")],
            transmit_only=[i for i, r in enumerate(roles) if r == "Tx"],
            receive_only=[i for i, r in enumerate(roles) if r == "Rx"],
            decouple=present != {"Tx", "TxRx"},
        )


def _detune(admittance, keep, detuned, resistance):
    """Terminate the detuned ports in the detuning resistance and eliminate them.

    Returns the admittance the kept ports see, and the whole terminated one.
    """
    loaded = admittance.clone()
    if detuned:
        index = torch.tensor(detuned, dtype=torch.long)
        loaded[..., index, index] += 1.0 / resistance
    return circuit.reduce(loaded, keep, detuned), loaded


def _subset(stages: tuple, ports: list[int]) -> tuple:
    """Return the matching stages of some ports, as stages of a smaller network."""
    index = torch.tensor(ports, dtype=torch.long)
    return tuple(
        circuit.Stage(
            loads=tuple(stage.loads[i] for i in ports),
            values=stage.values.index_select(-1, index),
            quality=None if stage.quality is None else stage.quality[index],
        )
        for stage in stages
    )


def _positions(side: list[int], optimised: list[int]) -> list[int]:
    """Return where the optimised ports fall within a side."""
    chosen = set(optimised)
    return [k for k, port in enumerate(side) if port in chosen]


def _side_cost(layout, x, omega, reference, preamplifier, detuning):
    """Weigh a layout's candidates as MARIE's ``matching_and_tuning_and_*`` do.

    The transmit side weighs its ports' match and, where MARIE does, their
    coupling; the receive side weighs the impedance each port shows its
    preamplifier. Each enters when the side has ports.
    """
    sides = _Sides.of(layout.roles)
    coil = layout.ports_admittance(layout.tuned(x, omega))
    stages = layout.stages(x)
    total = torch.zeros(x.shape[0], dtype=torch.float64)
    if sides.transmit:
        seen, _ = _detune(coil, sides.transmit, sides.receive_only, detuning)
        impedance, _ = circuit.place_matching(
            seen, _subset(stages, sides.transmit), omega
        )
        total = total + _matching_cost(
            impedance, _positions(sides.transmit, layout.optimised), reference
        )
        if sides.decouple:
            total = total + _decoupling_cost(impedance)
    if sides.receive:
        seen, _ = _detune(coil, sides.receive, sides.transmit_only, detuning)
        received = _receive_impedance(
            seen, _subset(stages, sides.receive), omega, preamplifier
        )
        total = total + _receive_cost(
            received, _positions(sides.receive, layout.optimised), reference
        )
    return total


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Search:
    """How one of MARIE's ``particleswarm`` searches is run.

    ``particleswarm``'s swarm size becomes the differential evolution's total
    population, its iteration limit and function tolerance carry over, and its
    objective limit stops the search early. Each restart that improves grows
    the population and the iteration limit as MARIE does; its inertia and
    adjustment weights have no counterpart and are dropped.

    Attributes
    ----------
    population, iterations, tolerance, target
        ``SwarmSize``, ``MaxIterations``, ``FunctionTolerance`` and
        ``ObjectiveLimit``.
    restarts
        How many searches to run at most; a search stops the restarts once its
        cost is below one.
    growth
        Iterations added after each improving restart.
    """

    population: int = 300
    iterations: int = 500
    tolerance: float = 1e-5
    target: float = 1e-8
    restarts: int = 10
    growth: int = 100


TUNING = Search(restarts=2)
MATCHING = Search()
DECOUPLING = Search(iterations=400, tolerance=1e-4)
RECEIVE_DECOUPLING = Search(iterations=400, tolerance=1e-4, restarts=5, growth=10)

_NARROWING = {"Tx": (0.95, 1.05), "Rx": (0.5, 2.05), "TxRx": (0.8, 1.2)}


def _inside(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Clip to the bounds, a hair inside them where they differ."""
    span = upper - lower
    fraction = np.divide(x - lower, span, out=np.zeros_like(x), where=span > 0)
    return lower + span * np.clip(fraction, 1e-9, 1 - 1e-9)


def _stop_below(target: float):
    """Return a callback that ends a differential evolution below ``target``.

    scipy hands the running result only to a callback whose one parameter is
    named ``intermediate_result``.
    """

    def callback(intermediate_result):
        return bool(intermediate_result.fun <= target)

    return callback


def _minimise(
    cost, lower, upper, search: Search, seed: int, narrowing=None, start=None
):
    """Run the restarts of one search and return the best vector and its cost.

    ``start``, where given, joins the first population, and each restart's
    population holds the best vector so far, so the search ends no worse than
    the values it was handed; MARIE's swarms start at random.
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "co-simulation searches need scipy: pip install 'mariepy[cosim]'"
        ) from error

    lower = np.array(lower, dtype=np.float64)
    upper = np.array(upper, dtype=np.float64)
    best, best_x = math.inf, 0.5 * (lower + upper)
    population, iterations = search.population, search.iterations
    for restart in range(search.restarts):
        if best < 1:
            break
        free = upper > lower
        pinned = lower.copy()

        def batch(x, free=free, pinned=pinned):
            x = np.atleast_2d(x.T) if x.ndim == 2 else x[None]
            full = np.repeat(pinned[None], x.shape[0], axis=0)
            full[:, free] = x
            return cost(torch.from_numpy(full)).numpy()

        if not free.any():
            value = float(batch(np.empty((0, 1)))[0])
            x = pinned
        else:
            callback = _stop_below(search.target)
            seeded = None
            if start is not None:
                seeded = _inside(np.asarray(start, dtype=np.float64), lower, upper)[
                    free
                ]
            result = differential_evolution(
                batch,
                list(zip(lower[free], upper[free], strict=True)),
                popsize=max(1, math.ceil(population / int(free.sum()))),
                maxiter=iterations,
                tol=search.tolerance,
                seed=seed + restart + 1,
                vectorized=True,
                updating="deferred",
                polish=False,
                init="latinhypercube",
                callback=callback,
                x0=seeded,
            )
            x = pinned.copy()
            x[free] = result.x
            value = float(result.fun)
        if value < best:
            best, best_x, start = value, x, x
            population = min(population + 100, 1000)
            iterations = min(iterations + search.growth, 1500)
            if narrowing is not None:
                bounds = [_around(v, *narrowing) for v in x]
                lower = np.array([b[0] for b in bounds])
                upper = np.array([b[1] for b in bounds])
    return best_x, best


# --------------------------------------------------------------------------
# The three searches
# --------------------------------------------------------------------------


def _tune(network, admittance, omega, seed, search, start):
    """``runner_T.m``: per entity, tune the elements so its ports show no reactance."""
    tuned, costs = {}, {}
    if not network.elements:
        return tuned, costs
    for entity in network.entities:
        layout = _Layout.build(
            network, admittance, network.rows_of(entity), matching=False
        )
        if not layout.n_variables or not layout.ports:
            continue

        def cost(x, layout=layout):
            impedance = torch.linalg.inv(
                layout.ports_admittance(layout.tuned(x, omega))
            )
            return _norm(_diagonal(impedance).imag)

        x, best = _minimise(
            cost,
            layout.lower,
            layout.upper,
            search,
            seed,
            start=[start[g] for g in layout.groups],
        )
        tuned.update(zip(layout.groups, x.tolist(), strict=True))
        costs[entity] = best
    return tuned, costs


def _match(
    network, admittance, omega, reference, seed, search, start, tuned, tuning_cost
):
    """``runner_M_T.m``: per entity, match and tune together."""
    first_entity = {}
    for terminal in network.terminals:
        if terminal.kind == "element":
            first_entity.setdefault(terminal.components[0].group, terminal.entities[0])
    coupling_groups = {c.group for c in network.couplings}

    matched, costs, narrowed = {}, {}, {}
    for entity in network.entities:
        layout = _Layout.build(
            network, admittance, network.rows_of(entity), matching=True
        )
        if not layout.n_variables:
            continue
        lower, upper = layout.lower.copy(), layout.upper.copy()
        for i, group in enumerate(layout.groups):
            if group not in tuned:
                continue
            if group in coupling_groups:
                lower[i], upper[i] = _around(tuned[group], 0.8, 1.2)
            elif tuning_cost.get(first_entity.get(group), math.inf) <= 1:
                lower[i], upper[i] = 0.8 * tuned[group], 1.2 * tuned[group]

        def cost(x, layout=layout):
            coil = layout.ports_admittance(layout.tuned(x, omega))
            impedance, _ = circuit.place_matching(coil, layout.stages(x), omega)
            return _matching_cost(impedance, layout.optimised, reference)

        x, best = _minimise(
            cost, lower, upper, search, seed, start=[start[g] for g in layout.groups]
        )
        factors = (0.5, 1.5) if best > 1 else (0.8, 1.2)
        for group, value in zip(layout.groups, x.tolist(), strict=True):
            matched[group] = value
            narrowed[group] = _around(value, *factors)
        costs[entity] = best
    return matched, costs, narrowed


def _bounds(layout, around, narrowed, factors, use_narrowed):
    """MARIE's bounds for a later search: its predecessor's, or around its values."""
    if use_narrowed:
        pairs = [
            narrowed.get(g, (layout.lower[i], layout.upper[i]))
            for i, g in enumerate(layout.groups)
        ]
    else:
        pairs = [_around(around[g], *factors) for g in layout.groups]
    return np.array([b[0] for b in pairs]), np.array([b[1] for b in pairs])


def _decouple(
    network, admittance, omega, options, seed, search, values, costs, narrowed
):
    """``runner_M_T_D.m`` and its receive variants: every entity at once, one role."""
    (role,) = network.roles
    layout = _Layout.build(
        network, admittance, list(range(len(network.terminals))), matching=True
    )
    if len(layout.ports) <= 1 or not layout.n_variables:
        return {}, None
    failed = sum(costs.values()) > 1
    factors = (0.5, 1.5) if failed else _NARROWING[role]
    lower, upper = _bounds(layout, values, narrowed, factors, failed)
    x, best = _minimise(
        lambda x: _side_cost(layout, x, omega, *options),
        lower,
        upper,
        search,
        seed,
        narrowing=factors,
        start=[values[g] for g in layout.groups],
    )
    return dict(zip(layout.groups, x.tolist(), strict=True)), best


def _search_roles(
    network, admittance, omega, options, seed, search, values, costs, narrowed
):
    """``runner_M_T_D_PD_split.m``: each role's rows on their own, the rest open."""
    port_entities = {
        role: {
            e
            for i in network.ports
            if network.terminals[i].role == role
            for e in network.terminals[i].entities
        }
        for role in ROLES
    }
    found, scores = {}, []
    for role in ("Rx", "Tx", "TxRx"):
        rows = [i for i, t in enumerate(network.terminals) if t.role == role]
        ports = [i for i in rows if network.terminals[i].kind == "port"]
        if not ports:
            continue
        matched = sum(costs.get(e, 0.0) for e in port_entities[role])
        layout = _Layout.build(network, admittance, rows, matching=True)
        if not layout.n_variables:
            scores.append(matched)
            continue
        factors = (0.5, 1.5) if matched > 1 else (0.8, 1.2)
        lower, upper = _bounds(layout, values, narrowed, factors, matched > 50)
        x, best = _minimise(
            lambda x, layout=layout: _side_cost(layout, x, omega, *options),
            lower,
            upper,
            search,
            seed,
            narrowing=factors,
            start=[values[g] for g in layout.groups],
        )
        found.update(zip(layout.groups, x.tolist(), strict=True))
        scores.append(best if len(ports) > 1 else matched)
    return found, sum(scores) / max(1, len(scores))


def _search_joint(
    network, admittance, omega, options, seed, search, values, score, narrowed
):
    """``runner_*_M_T_PD_DT_*_M_T_D.m``: every role together, the other side detuned."""
    layout = _Layout.build(
        network, admittance, list(range(len(network.terminals))), matching=True
    )
    if len(layout.ports) <= 1 or not layout.n_variables:
        return {}, None
    factors = (0.5, 1.5) if score > 50 else (0.95, 1.05)
    lower, upper = _bounds(layout, values, narrowed, factors, score > 100)
    x, best = _minimise(
        lambda x: _side_cost(layout, x, omega, *options),
        lower,
        upper,
        search,
        seed,
        narrowing=factors,
        start=[values[g] for g in layout.groups],
    )
    return dict(zip(layout.groups, x.tolist(), strict=True)), best


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CoSimulation:
    """A coil closed with its lumped values, and what that makes of its ports.

    Attributes
    ----------
    values
        Each terminal's values, in :attr:`Network.terminals` order.
    couplings
        Each coupling's coefficient, by its search variable.
    transmit_ports, receive_ports
        The rows of the ports on each side.
    transmit
        The voltage on every solver port per unit wave incident on each
        transmitting port's network, the receive-only ports detuned, shape
        ``(n_rows, n_transmit)``. MARIE's ``M_cal`` (``M_cal.tx`` where there
        are two sides) holds the same map to the wave rather than the voltage.
        None without transmitting ports.
    receive
        The same for each receiving port, the transmit-only ports detuned and
        the other receiving ports terminated in the preamplifier resistance,
        as MARIE's ``M_cal.rx`` builds it. None without receiving ports.
    admittance
        The coil ports' admittance with the tuning elements placed, lossy,
        MARIE's returned ``YPm``.
    impedance, scattering
        The transmitting ports' matched impedance and scattering parameters,
        lossy, or every port's without transmitting ports.
    receive_scattering
        Each receiving port's reflection with the others
        preamplifier-decoupled, MARIE's ``SP_check``, or None.
    transmit_dissipation, receive_dissipation
        Per port, half the resistance the matching elements' losses add, over
        the squared magnitude of the port impedance: MARIE's
        ``phi_lumped_elements``, which stores the transmit one on a diagonal.
        The receive one is None without receiving ports.
    costs
        The cost each search reached: ``"tuning"`` and ``"matching"`` per
        entity, ``"split"`` per role group's mean where roles mix, ``"final"``
        for the joint search (None when it did not run). Empty without a
        search.
    """

    values: tuple[tuple[float, ...], ...]
    couplings: dict[int, float]
    transmit_ports: list[int]
    receive_ports: list[int]
    transmit: torch.Tensor | None
    receive: torch.Tensor | None
    admittance: torch.Tensor
    impedance: torch.Tensor
    scattering: torch.Tensor
    receive_scattering: torch.Tensor | None
    transmit_dissipation: torch.Tensor
    receive_dissipation: torch.Tensor | None
    costs: dict


def _first(stages: tuple) -> tuple:
    """Drop the batch of a single candidate's stages."""
    return tuple(
        circuit.Stage(stage.loads, stage.values[0], stage.quality) for stage in stages
    )


def _dissipation(lossy: torch.Tensor, lossless: torch.Tensor) -> torch.Tensor:
    """``phi_lumped_elements``: ``|Re(Z_lossless - Z)| / (2 |Z|^2)``."""
    return 0.5 * (lossless - lossy).real.abs() / lossy.abs() ** 2


def _detuning_map(loaded, side):
    """Map a side's voltages to every port's, as ``M_detune_*`` does."""
    full = torch.linalg.inv(loaded)
    columns = full[:, side]
    return torch.linalg.solve(
        columns[side].transpose(0, 1), columns.transpose(0, 1)
    ).transpose(0, 1)


def _calibrate(network, admittance, omega, values, reference, preamplifier, detuning):
    """``calibration_tune_match_*.m`` and ``calibration_match_*.m``.

    Where MARIE maps the incident wave to the coil through
    ``calibration_matching.m``, which rebuilds a lossless network from the
    scattering parameters alone, the map here is the exact one through the
    matching elements, :func:`mariepy.circuit.coil_voltage`.
    """
    layout = _Layout.build(
        network, admittance, list(range(len(network.terminals))), matching=True
    )
    sides = _Sides.of(layout.roles)
    x = layout.vector(values)
    placed = layout.tuned(x, omega, lossy=True)[0]
    placed_lossless = layout.tuned(x, omega, lossless=True)[0]
    coil = layout.ports_admittance(placed)
    coil_lossless = layout.ports_admittance(placed_lossless)
    if layout.elements:
        tuning = circuit.tuning_calibration(placed, layout.ports, layout.elements)
    else:
        tuning = torch.eye(len(layout.ports), dtype=torch.complex128)
    stages = _first(layout.stages(x, lossy=True))

    shown = sides.transmit or list(range(len(layout.ports)))
    seen, loaded = _detune(coil, shown, sides.receive_only, detuning)
    seen_lossless, _ = _detune(coil_lossless, shown, sides.receive_only, detuning)
    chain = _subset(stages, shown)
    impedance, impedance_lossless = circuit.place_matching(
        seen, chain, omega, lossless=seen_lossless
    )
    transmit = None
    if sides.transmit:
        transmit = (
            tuning
            @ _detuning_map(loaded, shown)
            @ circuit.coil_voltage(seen, chain, omega, reference)
        )

    receive = receive_scattering = receive_dissipation = None
    if sides.receive:
        side, loaded = _detune(coil, sides.receive, sides.transmit_only, detuning)
        side_lossless, _ = _detune(
            coil_lossless, sides.receive, sides.transmit_only, detuning
        )
        chain = _subset(stages, sides.receive)
        n = len(sides.receive)
        decoupling = torch.zeros((n, n), dtype=torch.complex128)
        drive = torch.zeros((n, n), dtype=torch.complex128)
        received = torch.zeros(n, dtype=torch.complex128)
        received_lossless = torch.zeros(n, dtype=torch.complex128)
        for port in range(n):
            terminated, rest = _preamplifier_loaded(side, port, preamplifier)
            terminated_lossless, _ = _preamplifier_loaded(
                side_lossless, port, preamplifier
            )
            full = torch.linalg.inv(terminated)
            decoupling[:, port] = full[:, port] / full[port, port]
            single = circuit.reduce(terminated, [port], rest)
            own = _single_port_stages(chain, port)
            z, z_lossless = circuit.place_matching(
                single,
                own,
                omega,
                lossless=circuit.reduce(terminated_lossless, [port], rest),
            )
            drive[port, port] = circuit.coil_voltage(single, own, omega, reference)[
                0, 0
            ]
            received[port] = z[0, 0]
            received_lossless[port] = z_lossless[0, 0]
        receive = tuning @ _detuning_map(loaded, sides.receive) @ decoupling @ drive
        receive_scattering = (received - reference) / (received + reference)
        receive_dissipation = _dissipation(received, received_lossless)

    per_row = tuple(
        tuple(
            values[c.group] if c.group is not None else c.value
            for c in terminal.components
        )
        for terminal in network.terminals
    )
    rows = network.ports
    return {
        "values": per_row,
        "couplings": {c.group: values[c.group] for c in network.couplings},
        "transmit_ports": [rows[i] for i in sides.transmit],
        "receive_ports": [rows[i] for i in sides.receive],
        "transmit": transmit,
        "receive": receive,
        "admittance": coil,
        "impedance": impedance,
        "scattering": circuit.z_to_s(impedance, reference),
        "receive_scattering": receive_scattering,
        "transmit_dissipation": _dissipation(
            _diagonal(impedance), _diagonal(impedance_lossless)
        ),
        "receive_dissipation": receive_dissipation,
    }


def co_simulate(
    network: Network,
    admittance: torch.Tensor,
    omega: float,
    *,
    reference: float = 50.0,
    preamplifier_resistance: float = PREAMPLIFIER_RESISTANCE,
    detuning_resistance: float = DETUNING_RESISTANCE,
    seed: int = 0,
    tuning: Search = TUNING,
    matching: Search = MATCHING,
    decoupling: Search | None = None,
) -> CoSimulation:
    """Close a solved coil with its lumped values and calibrate its ports.

    Ported from ``co_simulation.m`` and its masters. With :attr:`Network.tmd`
    set, the values are searched; otherwise the file's values are used, and
    only the matching networks are placed.

    Parameters
    ----------
    network
        The rows of ``admittance`` and their values, from :func:`read_network`.
    admittance
        The solver's port admittance, :attr:`mariepy.solver.Result.admittance`.
    omega
        Angular frequency in rad/s.
    reference
        Line impedance in ohms, MARIE's ``emc.Z0``.
    preamplifier_resistance
        The resistance a preamplifier presents, MARIE's ``emc.Preamp_res``.
    detuning_resistance
        The resistance a detuned port presents, MARIE's ``emc.Detune_res``.
    seed
        Seeds the searches; each restart adds its own count, as MARIE's
        ``rng(counter_repeat)`` does.
    tuning, matching
        How the two per-entity searches run.
    decoupling
        How the later searches run; by default as MARIE runs them.

    Returns
    -------
    CoSimulation
        The values, the calibrations and the matched port parameters.

    Raises
    ------
    ValueError
        If the admittance does not have one row per terminal.
    """
    admittance = admittance.detach().to(torch.complex128).cpu()
    if admittance.shape[-1] != len(network.terminals):
        raise ValueError(
            f"the admittance has {admittance.shape[-1]} ports, the network "
            f"{len(network.terminals)} terminals: solve with the same TMD flag"
        )
    roles = network.roles
    options = (reference, preamplifier_resistance, detuning_resistance)
    values = network.start()
    costs: dict = {}
    if network.tmd:
        tuned, tuning_cost = _tune(network, admittance, omega, seed, tuning, values)
        values.update(tuned)
        matched, matching_cost, narrowed = _match(
            network,
            admittance,
            omega,
            reference,
            seed,
            matching,
            values,
            tuned,
            tuning_cost,
        )
        values.update(matched)
        costs = {"tuning": tuning_cost, "matching": matching_cost}
        if len(roles) == 1:
            search = decoupling or (
                RECEIVE_DECOUPLING if roles == {"Rx"} else DECOUPLING
            )
            final, final_cost = _decouple(
                network,
                admittance,
                omega,
                options,
                seed,
                search,
                values,
                matching_cost,
                narrowed,
            )
        else:
            search = decoupling or DECOUPLING
            split, score = _search_roles(
                network,
                admittance,
                omega,
                options,
                seed,
                search,
                values,
                matching_cost,
                narrowed,
            )
            values.update(split)
            costs["split"] = score
            final, final_cost = _search_joint(
                network,
                admittance,
                omega,
                options,
                seed,
                search,
                values,
                score,
                narrowed,
            )
        values.update(final)
        costs["final"] = final_cost
    calibrated = _calibrate(network, admittance, omega, values, *options)
    return CoSimulation(costs=costs, **calibrated)


def calibrate(per_port: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Combine the solver's per-port quantities by a calibration.

    MARIE's ``co_simulation_fields_calibration.m``: ``Jcb * M_cal``.

    Parameters
    ----------
    per_port
        One quantity per solver port along the first axis, each for a unit
        voltage on that port: currents, fields.
    mapping
        :attr:`CoSimulation.transmit` or :attr:`CoSimulation.receive`.

    Returns
    -------
    torch.Tensor
        One quantity per matched port along the first axis, each for a unit
        wave incident on that port.
    """
    mapping = mapping.to(device=per_port.device, dtype=torch.complex128)
    return torch.einsum("rp,r...->p...", mapping, per_port.to(torch.complex128))


@dataclass(frozen=True)
class Sweep:
    """The matched ports across a band, the solved coupling held fixed.

    Attributes
    ----------
    frequency
        Shape ``(n_frequencies,)``, in Hz.
    index
        The entry that is the working frequency.
    transmit_impedance, transmit_scattering
        Shape ``(n_frequencies, n_transmit, n_transmit)``, or None without
        transmitting ports.
    receive_impedance, receive_scattering
        Shape ``(n_frequencies, n_receive)``, or None without receiving ports.
    """

    frequency: torch.Tensor
    index: int
    transmit_impedance: torch.Tensor | None
    transmit_scattering: torch.Tensor | None
    receive_impedance: torch.Tensor | None
    receive_scattering: torch.Tensor | None


def sweep(
    network: Network,
    admittance: torch.Tensor,
    omega: float,
    result: CoSimulation,
    *,
    span: float = 0.2,
    points: int = 5000,
    reference: float = 50.0,
    preamplifier_resistance: float = PREAMPLIFIER_RESISTANCE,
    detuning_resistance: float = DETUNING_RESISTANCE,
) -> Sweep:
    """Sweep the lumped elements' frequency around the working one.

    Ported from ``get_ZPm_*_freq.m``. As there, only the lumped elements see
    the frequency change: the coil's admittance stays the one solved at
    ``omega``.

    Parameters
    ----------
    network, admittance, omega
        As given to :func:`co_simulate`.
    result
        Its return.
    span
        Half-width of the band, relative to the working frequency.
    points
        Number of frequencies; the one nearest the working frequency is
        replaced by it.
    reference, preamplifier_resistance, detuning_resistance
        As given to :func:`co_simulate`.

    Returns
    -------
    Sweep
        The matched ports' parameters across the band, lossy.
    """
    admittance = admittance.detach().to(torch.complex128).cpu()
    centre = omega / (2 * math.pi)
    frequency = torch.linspace(
        centre * (1 - span), centre * (1 + span), points, dtype=torch.float64
    )
    index = int((frequency - centre).abs().argmin())
    frequency[index] = centre
    omegas = 2 * math.pi * frequency

    layout = _Layout.build(
        network, admittance, list(range(len(network.terminals))), matching=True
    )
    sides = _Sides.of(layout.roles)
    x = layout.vector(network.values_of(result)).expand(points, -1)
    coil = layout.ports_admittance(layout.tuned(x, omegas, lossy=True))
    stages = layout.stages(x, lossy=True)

    transmit_impedance = transmit_scattering = None
    receive_impedance = receive_scattering = None
    if sides.transmit:
        seen, _ = _detune(coil, sides.transmit, sides.receive_only, detuning_resistance)
        transmit_impedance, _ = circuit.place_matching(
            seen, _subset(stages, sides.transmit), omegas
        )
        transmit_scattering = circuit.z_to_s(transmit_impedance, reference)
    if sides.receive:
        seen, _ = _detune(coil, sides.receive, sides.transmit_only, detuning_resistance)
        receive_impedance = _receive_impedance(
            seen, _subset(stages, sides.receive), omegas, preamplifier_resistance
        )
        receive_scattering = (receive_impedance - reference) / (
            receive_impedance + reference
        )
    return Sweep(
        frequency=frequency,
        index=index,
        transmit_impedance=transmit_impedance,
        transmit_scattering=transmit_scattering,
        receive_impedance=receive_impedance,
        receive_scattering=receive_scattering,
    )
