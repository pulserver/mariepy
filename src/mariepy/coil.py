"""The RWG basis on a surface coil: edges, degrees of freedom, ports and loads.

Ported from MARIE 3.0's ``src_geometry/scoil_geometry/geo_scoil.m``,
``mesh_geo/Mesh_PreProc.m``, ``rwg_geo/get_rwg_vertices.m`` and
``ports_geo/geo_scoil_lumped_elements.m``.

Every mesh edge shared by two triangles carries one Rao-Wilton-Glisson basis
function; edges on a rim carry none, so no current leaves the sheet. The
function on edge ``e`` is positive on the triangle whose local edge points along
the stored edge direction and negative on the other, which is what ``signs``
records.

Degrees of freedom are numbered ports first, then lumped elements, then the
remaining interior edges, so that a port's rows and columns are a contiguous
block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from mariepy.mesh import _FIRST_NODE, _SECOND_NODE, SurfaceMesh

__all__ = ["Adjacency", "Port", "SurfaceCoil", "read_lumped_elements"]

_PAIR_CHUNK = 256


@dataclass(frozen=True)
class Adjacency:
    """Triangle pairs that touch, which the singular integrals need.

    Attributes
    ----------
    edge
        Pairs sharing two vertices, shape ``(n_edge_adjacent, 2)``, first index
        smaller.
    vertex
        Pairs sharing one vertex, same layout.
    """

    edge: torch.Tensor
    vertex: torch.Tensor


@dataclass(frozen=True)
class Port:
    """A port or a lumped element sitting on a set of coil edges.

    Attributes
    ----------
    tag
        Physical line tag in the mesh, which is the element's number.
    kind
        ``"port"`` for a driven port, ``"element"`` for a lumped load.
    load
        Load type: ``"resistor"``, ``"inductor"``, ``"capacitor"`` or
        ``"mutual_inductor"`` for an element. A driven port carries the string
        naming its matching network, which milestone 1 does not evaluate.
    value
        Resistance in ohms, inductance in henries or capacitance in farads.
    quality
        Quality factor that sets the element's loss.
    voltage
        Delta-gap drive, 1 for a driven port and 0 otherwise.
    coupled_tag
        Tag of the partner element of a mutual inductor, else None.
    coupled_value
        Mutual inductance in henries, else None.
    dofs
        Degrees of freedom the element spans, shape ``(n_terms,)``.
    """

    tag: int
    kind: str
    load: str
    value: float
    quality: float
    voltage: float
    coupled_tag: int | None = None
    coupled_value: float | None = None
    dofs: torch.Tensor | None = None

    def impedance(self, omega: float) -> tuple[complex, float]:
        """Series impedance of the load and the resistive part of it.

        Parameters
        ----------
        omega
            Angular frequency in rad/s.

        Returns
        -------
        tuple of complex and float
            The impedance in ohms, and its loss resistance.

        Raises
        ------
        ValueError
            If the load type is not one this milestone models.
        """
        if self.load == "resistor":
            return complex(self.value), float(self.value)
        if self.load in ("inductor", "mutual_inductor"):
            loss = omega * self.value / self.quality
            return 1j * omega * self.value + loss, loss
        if self.load == "capacitor":
            loss = 1.0 / (omega * self.value * self.quality)
            return 1.0 / (1j * omega * self.value) + loss, loss
        raise ValueError(f"unknown lumped load {self.load!r}")


@dataclass(frozen=True)
class SurfaceCoil:
    """A surface mesh with its RWG basis, its ports and its loads.

    Attributes
    ----------
    mesh
        The triangulated surface.
    edges
        Node pair of each edge, shape ``(n_edges, 2)``, in the direction the
        basis function on it takes as positive.
    edge_of_triangle
        Edge index of each triangle's three local edges, shape
        ``(n_triangles, 3)``.
    signs
        Whether a triangle's local edge runs along the stored edge direction,
        ``+1`` or ``-1``, same shape.
    dof_of_edge
        Degree of freedom carried by each edge, ``-1`` for a rim edge, shape
        ``(n_edges,)``.
    ports
        Ports and lumped elements, ports first.
    adjacency
        Triangle pairs that touch.
    """

    mesh: SurfaceMesh
    edges: torch.Tensor
    edge_of_triangle: torch.Tensor
    signs: torch.Tensor
    dof_of_edge: torch.Tensor
    ports: tuple[Port, ...]
    adjacency: Adjacency

    @property
    def n_edges(self) -> int:
        """Number of mesh edges, rims included."""
        return int(self.edges.shape[0])

    @property
    def n_dof(self) -> int:
        """Number of basis functions."""
        return int((self.dof_of_edge >= 0).sum())

    @property
    def n_driven(self) -> int:
        """Number of driven ports."""
        return sum(1 for port in self.ports if port.kind == "port")

    @property
    def edge_of_dof(self) -> torch.Tensor:
        """Edge index of each degree of freedom, shape ``(n_dof,)``."""
        order = torch.argsort(self.dof_of_edge)
        return order[self.n_edges - self.n_dof :]

    def dof_lengths(self) -> torch.Tensor:
        """Length of the edge each basis function sits on, shape ``(n_dof,)``."""
        pairs = self.edges[self.edge_of_dof]
        nodes = self.mesh.nodes
        return torch.linalg.vector_norm(nodes[pairs[:, 1]] - nodes[pairs[:, 0]], dim=-1)

    def dof_of_triangle(self) -> torch.Tensor:
        """Degree of freedom of each local edge, ``-1`` where there is none.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3)``.
        """
        return self.dof_of_edge[self.edge_of_triangle]

    def rwg_vertices(self) -> torch.Tensor:
        """Give the four vertices that define each basis function.

        Returns
        -------
        torch.Tensor
            Shape ``(n_dof, 4, 3)``: the free vertex of the positive triangle,
            the free vertex of the negative triangle, then the two vertices of
            the shared edge in the positive direction.
        """
        dof = self.dof_of_triangle()
        n_dof = self.n_dof
        positive = torch.zeros(n_dof, dtype=torch.int64, device=dof.device)
        negative = torch.zeros_like(positive)
        positive_local = torch.zeros_like(positive)
        negative_local = torch.zeros_like(positive)
        triangle, local = torch.nonzero(dof >= 0, as_tuple=True)
        which = dof[triangle, local]
        forward = self.signs[triangle, local] > 0
        positive[which[forward]] = triangle[forward]
        positive_local[which[forward]] = local[forward]
        negative[which[~forward]] = triangle[~forward]
        negative_local[which[~forward]] = local[~forward]

        nodes = self.mesh.nodes
        triangles = self.mesh.triangles
        free_positive = nodes[triangles[positive, positive_local]]
        free_negative = nodes[triangles[negative, negative_local]]
        shared = nodes[self.edges[self.edge_of_dof]]
        return torch.stack(
            [free_positive, free_negative, shared[:, 0], shared[:, 1]], dim=1
        )

    def shared_vertices(self, rows: torch.Tensor) -> torch.Tensor:
        """Count vertices each triangle shares with the given triangles.

        Parameters
        ----------
        rows
            Triangle indices, shape ``(k,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(k, n_triangles)``, entries 0 to 3.
        """
        return _shared_vertices(self.mesh.triangles, rows)

    @classmethod
    def build(cls, mesh: SurfaceMesh, elements: tuple[Port, ...] = ()) -> SurfaceCoil:
        """Build the RWG basis of a mesh and attach its ports.

        Parameters
        ----------
        mesh
            The coil surface. Its triangle order fixes the sign of each basis
            function, so pass the mesh through ``align_to_lines`` first if the
            port polarity matters.
        elements
            Port and lumped-element definitions, matched to the mesh by tag.

        Returns
        -------
        SurfaceCoil
            The coil with its basis, degree-of-freedom numbering and ports.

        Raises
        ------
        ValueError
            If an edge is shared by more than two triangles, or an element's tag
            names no interior edge of the mesh.
        """
        edges, edge_of_triangle, signs = _build_edges(mesh)
        n_edges = int(edges.shape[0])
        occurrences = torch.bincount(edge_of_triangle.reshape(-1), minlength=n_edges)
        if int(occurrences.max()) > 2:
            raise ValueError("an edge is shared by more than two triangles")
        interior = occurrences == 2

        tag_of_edge = torch.zeros(n_edges, dtype=torch.int64, device=edges.device)
        if mesh.lines.numel():
            tag_of_edge[_edge_of_pairs(edges, mesh.lines)] = mesh.line_tags

        dof_of_edge = torch.full((n_edges,), -1, dtype=torch.int64, device=edges.device)
        ordered = sorted(elements, key=lambda port: (port.kind != "port", port.tag))
        placed: list[Port] = []
        assigned = 0
        for port in ordered:
            on_port = torch.nonzero(
                interior & (tag_of_edge == port.tag), as_tuple=False
            ).flatten()
            if on_port.numel() == 0:
                raise ValueError(
                    f"element {port.tag} names no interior edge of the mesh"
                )
            dofs = torch.arange(
                assigned, assigned + on_port.numel(), device=edges.device
            )
            dof_of_edge[on_port] = dofs
            assigned += int(on_port.numel())
            placed.append(
                Port(
                    tag=port.tag,
                    kind=port.kind,
                    load=port.load,
                    value=port.value,
                    quality=port.quality,
                    voltage=port.voltage,
                    coupled_tag=port.coupled_tag,
                    coupled_value=port.coupled_value,
                    dofs=dofs,
                )
            )
        rest = torch.nonzero(interior & (dof_of_edge < 0), as_tuple=False).flatten()
        dof_of_edge[rest] = torch.arange(
            assigned, assigned + rest.numel(), device=edges.device
        )

        return cls(
            mesh=mesh,
            edges=edges,
            edge_of_triangle=edge_of_triangle,
            signs=signs,
            dof_of_edge=dof_of_edge,
            ports=tuple(placed),
            adjacency=_classify_pairs(mesh.triangles),
        )


def read_lumped_elements(path: str | Path) -> tuple[Port, ...]:
    """Read MARIE's JSON lumped-element file.

    Each element's ``number`` is the physical line tag it sits on. A driven port
    takes a unit delta-gap drive; a lumped element takes none.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    tuple of Port
        The elements the file defines, in file order.

    Raises
    ------
    ValueError
        If an element declares a type other than ``port`` or ``element``.
    """
    data = json.loads(Path(path).read_text())
    elements = data["coil_configuration"]["elements"]
    ports: list[Port] = []
    for element in elements:
        kind = element["type"]
        if kind not in ("port", "element"):
            raise ValueError(f"element {element['number']} has unknown type {kind!r}")
        cross_talk = element.get("cross_talk") or {}
        value = element["value"]
        quality = element["Q"]
        ports.append(
            Port(
                tag=int(element["number"]),
                kind=kind,
                load=element["load"],
                value=float(value[0]) if isinstance(value, list) else float(value),
                quality=float(quality[0])
                if isinstance(quality, list)
                else float(quality),
                voltage=1.0 if kind == "port" else 0.0,
                coupled_tag=(
                    int(cross_talk["coupled_port"])
                    if "coupled_port" in cross_talk
                    else None
                ),
                coupled_value=(
                    float(cross_talk["coupled_value"])
                    if "coupled_value" in cross_talk
                    else None
                ),
            )
        )
    return tuple(ports)


def _build_edges(mesh: SurfaceMesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give each mesh edge the number at which the triangle sweep first meets it."""
    triangles = mesh.triangles
    directed = torch.stack(
        [triangles[:, _FIRST_NODE], triangles[:, _SECOND_NODE]], dim=-1
    ).reshape(-1, 2)
    low = directed.min(dim=1).values
    high = directed.max(dim=1).values
    key = low * mesh.n_nodes + high

    _, inverse = torch.unique(key, return_inverse=True)
    n_edges = int(inverse.max()) + 1 if inverse.numel() else 0
    appearance = torch.full(
        (n_edges,), inverse.numel(), dtype=torch.int64, device=key.device
    )
    appearance.scatter_reduce_(
        0, inverse, torch.arange(inverse.numel(), device=key.device), reduce="amin"
    )
    identifier = torch.empty(n_edges, dtype=torch.int64, device=key.device)
    identifier[torch.argsort(appearance)] = torch.arange(n_edges, device=key.device)

    edges = torch.empty((n_edges, 2), dtype=torch.int64, device=key.device)
    edges[identifier] = directed[appearance]
    edge_of_triangle = identifier[inverse].reshape(-1, 3)
    along = directed[:, 0] == edges[identifier[inverse], 0]
    signs = torch.where(along, 1, -1).reshape(-1, 3)
    return edges, edge_of_triangle, signs


def _edge_of_pairs(edges: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Find the edge index of each unordered node pair.

    Raises
    ------
    ValueError
        If a pair is not an edge of the mesh.
    """
    span = int(edges.max()) + 1
    keys = edges.min(dim=1).values * span + edges.max(dim=1).values
    order = torch.argsort(keys)
    sought = pairs.min(dim=1).values * span + pairs.max(dim=1).values
    at = torch.searchsorted(keys[order], sought)
    at = at.clamp(max=keys.numel() - 1)
    found = order[at]
    hit = keys[found] == sought
    if not bool(hit.all()):
        missing = pairs[~hit][0].tolist()
        raise ValueError(f"node pair {missing} is not an edge of the mesh")
    return found


def _shared_vertices(triangles: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Count the vertices each triangle shares with each of ``rows``."""
    here = triangles[rows][:, None, :, None]
    return (here == triangles[None, :, None, :]).any(dim=-1).sum(dim=-1)


def _classify_pairs(triangles: torch.Tensor) -> Adjacency:
    """Split the triangle pairs that touch into edge-adjacent and vertex-adjacent."""
    n_triangles = int(triangles.shape[0])
    device = triangles.device
    edge_pairs: list[torch.Tensor] = []
    vertex_pairs: list[torch.Tensor] = []
    for start in range(0, n_triangles, _PAIR_CHUNK):
        rows = torch.arange(start, min(start + _PAIR_CHUNK, n_triangles), device=device)
        shared = _shared_vertices(triangles, rows)
        below = torch.arange(n_triangles, device=device)[None, :] < rows[:, None]
        for count, sink in ((2, edge_pairs), (1, vertex_pairs)):
            here, there = torch.nonzero((shared == count) & below, as_tuple=True)
            sink.append(torch.stack([there, rows[here]], dim=1))
    return Adjacency(
        edge=torch.cat(edge_pairs)
        if edge_pairs
        else torch.empty((0, 2), dtype=torch.int64),
        vertex=torch.cat(vertex_pairs)
        if vertex_pairs
        else torch.empty((0, 2), dtype=torch.int64),
    )
