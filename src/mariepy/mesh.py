"""A triangular surface mesh, its element geometry, and the GMSH 2.2 reader.

Ported from MARIE 3.0's ``src_geometry/scoil_geometry/mesh_geo/Mesh_Parse.m``,
``Mesh_Permute.m``, ``Mesh_CLP.m`` and ``rwg_geo/Triangle_area.m``.

A coil is a sheet of triangles carrying a surface current. The mesh holds the
node coordinates, the triangles, and the line elements that tag where the ports
and the lumped elements sit. Local edge ``n`` of a triangle is the edge opposite
vertex ``n``, and every per-edge quantity below follows that numbering.

Arrays are C-ordered with the element index first: ``(n_triangles, 3, 3)`` is
three vectors of three components for each triangle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = ["SurfaceMesh"]

# Local edge n runs from vertex _FIRST_NODE[n] to vertex _SECOND_NODE[n], so
# that edge n is the one opposite vertex n and l_0 = r_1 - r_2.
_FIRST_NODE = (2, 0, 1)
_SECOND_NODE = (1, 2, 0)

_GMSH_LINE = 1
_GMSH_TRIANGLE = 2

_GOLDEN = (1.0 + 5.0**0.5) / 2.0
_ICOSAHEDRON_FACES = (
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
)


@dataclass(frozen=True)
class SurfaceMesh:
    """A triangulated surface with tagged line elements.

    Attributes
    ----------
    nodes
        Node coordinates in metres, shape ``(n_nodes, 3)``, real.
    triangles
        Node indices of each triangle, shape ``(n_triangles, 3)``, integer.
    triangle_tags
        Physical surface each triangle belongs to, shape ``(n_triangles,)``.
    lines
        Node indices of each line element, shape ``(n_lines, 2)``, integer.
    line_tags
        Physical line each line element belongs to, shape ``(n_lines,)``. A
        positive tag names a port or a lumped element; mesh edges that carry no
        line element are ordinary interior edges.
    """

    nodes: torch.Tensor
    triangles: torch.Tensor
    triangle_tags: torch.Tensor
    lines: torch.Tensor
    line_tags: torch.Tensor

    def __post_init__(self) -> None:
        """Check the shapes agree and every index names a node."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(
                f"nodes must have shape (n_nodes, 3); got {tuple(self.nodes.shape)}"
            )
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles must have shape (n_triangles, 3); got {tuple(self.triangles.shape)}"
            )
        if self.lines.ndim != 2 or self.lines.shape[1] != 2:
            raise ValueError(
                f"lines must have shape (n_lines, 2); got {tuple(self.lines.shape)}"
            )
        if self.triangle_tags.shape != self.triangles.shape[:1]:
            raise ValueError("triangle_tags must carry one tag per triangle")
        if self.line_tags.shape != self.lines.shape[:1]:
            raise ValueError("line_tags must carry one tag per line element")
        for name, index in (("triangles", self.triangles), ("lines", self.lines)):
            if index.numel() and (
                int(index.min()) < 0 or int(index.max()) >= self.n_nodes
            ):
                raise ValueError(f"{name} index a node outside 0..{self.n_nodes - 1}")

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.nodes.shape[0])

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        return int(self.triangles.shape[0])

    @property
    def device(self) -> torch.device:
        """Device the mesh lives on."""
        return self.nodes.device

    def to(self, device: torch.device | str) -> SurfaceMesh:
        """Return the mesh on another device.

        Parameters
        ----------
        device
            Target device.

        Returns
        -------
        SurfaceMesh
            The same mesh with every array moved.
        """
        return SurfaceMesh(
            nodes=self.nodes.to(device),
            triangles=self.triangles.to(device),
            triangle_tags=self.triangle_tags.to(device),
            lines=self.lines.to(device),
            line_tags=self.line_tags.to(device),
        )

    def vertices(self) -> torch.Tensor:
        """Coordinates of each triangle's three vertices.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3, 3)``, vertex then component.
        """
        return self.nodes[self.triangles]

    def edge_vectors(self) -> torch.Tensor:
        """Edge vectors ``l_n``, from vertex ``n + 1`` to vertex ``n + 2`` cyclically.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3, 3)``, local edge then component.
        """
        vertices = self.vertices()
        return vertices[:, _SECOND_NODE, :] - vertices[:, _FIRST_NODE, :]

    def edge_lengths(self) -> torch.Tensor:
        """Length of each local edge.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3)``.
        """
        return torch.linalg.vector_norm(self.edge_vectors(), dim=-1)

    def centroids(self) -> torch.Tensor:
        """Barycentre of each triangle.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3)``.
        """
        return self.vertices().mean(dim=1)

    def rho(self) -> torch.Tensor:
        """Vector from each vertex to the triangle's barycentre.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3, 3)``, vertex then component.
        """
        return self.centroids().unsqueeze(1) - self.vertices()

    def normals(self) -> torch.Tensor:
        """Give the unit normal of each triangle, right-handed about its vertex order.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles, 3)``.
        """
        vertices = self.vertices()
        cross = torch.linalg.cross(
            vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1
        )
        return cross / torch.linalg.vector_norm(cross, dim=-1, keepdim=True)

    def areas(self) -> torch.Tensor:
        """Area of each triangle.

        Returns
        -------
        torch.Tensor
            Shape ``(n_triangles,)``.
        """
        vertices = self.vertices()
        cross = torch.linalg.cross(
            vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], dim=-1
        )
        return 0.5 * torch.linalg.vector_norm(cross, dim=-1)

    def align_to_lines(self) -> SurfaceMesh:
        """Order the triangle pair of each line element by the line's direction.

        The two triangles that share a tagged line are swapped, when needed, so
        that the one in which the line runs in the triangle's own cyclic order
        comes first. The degree of freedom on that edge then takes its sign from
        the line's direction, which is what fixes the polarity of the delta-gap
        drive.

        Returns
        -------
        SurfaceMesh
            The same mesh with the triangle order adjusted.

        Raises
        ------
        ValueError
            If a line element is not shared by exactly two triangles.
        """
        triangles = self.triangles.clone()
        for line in self.lines:
            first, second = int(line[0]), int(line[1])
            holds_both = ((triangles == first).any(dim=1)) & (
                (triangles == second).any(dim=1)
            )
            found = torch.nonzero(holds_both, as_tuple=False).flatten()
            if found.numel() != 2:
                raise ValueError(
                    f"line element ({first}, {second}) is shared by {found.numel()} triangles,"
                    " not two"
                )
            left, right = int(found[0]), int(found[1])
            corner = triangles[left]
            at_first = int(torch.nonzero(corner == first, as_tuple=False)[0])
            at_second = int(torch.nonzero(corner == second, as_tuple=False)[0])
            if (at_second - at_first) % 3 != 1:
                swapped = triangles[left].clone()
                triangles[left] = triangles[right]
                triangles[right] = swapped
        return SurfaceMesh(
            nodes=self.nodes,
            triangles=triangles,
            triangle_tags=self.triangle_tags,
            lines=self.lines,
            line_tags=self.line_tags,
        )

    @classmethod
    def read_gmsh22(
        cls, path: str | Path, *, device: torch.device | str = "cpu"
    ) -> SurfaceMesh:
        """Read a GMSH 2.2 ASCII mesh.

        Node numbers in the file need be neither dense nor ordered; they are
        renumbered to the position the node takes in ``nodes``. Element types
        other than lines and triangles are skipped.

        Parameters
        ----------
        path
            File to read.
        device
            Device the arrays are built on.

        Returns
        -------
        SurfaceMesh
            The mesh the file describes.

        Raises
        ------
        ValueError
            If the file has no ``$Nodes`` or no ``$Elements`` section, or an
            element names a node number the file never defined.
        """
        text = Path(path).read_text()
        node_block = _section(text, "Nodes")
        element_block = _section(text, "Elements")
        if node_block is None or element_block is None:
            raise ValueError(f"{path} has no $Nodes or no $Elements section")

        coordinates: list[tuple[float, float, float]] = []
        position: dict[int, int] = {}
        for record in node_block[1:]:
            fields = record.split()
            position[int(fields[0])] = len(coordinates)
            coordinates.append((float(fields[1]), float(fields[2]), float(fields[3])))

        lines: list[tuple[int, int]] = []
        line_tags: list[int] = []
        triangles: list[tuple[int, int, int]] = []
        triangle_tags: list[int] = []
        for record in element_block[1:]:
            fields = [int(field) for field in record.split()]
            kind, n_tags = fields[1], fields[2]
            tag = fields[3] if n_tags else 0
            nodes = fields[3 + n_tags :]
            try:
                renumbered = [position[number] for number in nodes]
            except KeyError as missing:
                raise ValueError(
                    f"{path} uses undefined node number {missing.args[0]}"
                ) from None
            if kind == _GMSH_LINE:
                lines.append((renumbered[0], renumbered[1]))
                line_tags.append(tag)
            elif kind == _GMSH_TRIANGLE:
                triangles.append((renumbered[0], renumbered[1], renumbered[2]))
                triangle_tags.append(tag)

        return cls(
            nodes=torch.tensor(coordinates, dtype=torch.float64, device=device).reshape(
                -1, 3
            ),
            triangles=torch.tensor(triangles, dtype=torch.int64, device=device).reshape(
                -1, 3
            ),
            triangle_tags=torch.tensor(triangle_tags, dtype=torch.int64, device=device),
            lines=torch.tensor(lines, dtype=torch.int64, device=device).reshape(-1, 2),
            line_tags=torch.tensor(line_tags, dtype=torch.int64, device=device),
        )

    @classmethod
    def loop(
        cls,
        *,
        radius: float,
        width: float,
        n_around: int,
        n_across: int,
        ports: int = 1,
        device: torch.device | str = "cpu",
    ) -> SurfaceMesh:
        """Build a single-turn loop coil: a flat annular strip in the ``z = 0`` plane.

        The strip is closed in the azimuthal direction and open at both rims, so
        no current crosses them. Its ports sit on the radial edges at equally
        spaced azimuths, tagged 1 upwards, the first at azimuth zero.

        Parameters
        ----------
        radius
            Mean radius of the strip in metres.
        width
            Radial width of the strip in metres.
        n_around
            Number of divisions around the loop; at least three.
        n_across
            Number of divisions across the strip; at least one.
        ports
            Number of ports spaced around the loop; at least one, and no more
            than ``n_around``.
        device
            Device the arrays are built on.

        Returns
        -------
        SurfaceMesh
            The mesh, with its triangle order already aligned to the port line.

        Raises
        ------
        ValueError
            If the strip would be degenerate or too coarse to close.
        """
        if n_around < 3 or n_across < 1:
            raise ValueError(
                f"need n_around >= 3 and n_across >= 1; got {n_around}, {n_across}"
            )
        if not 1 <= ports <= n_around:
            raise ValueError(
                f"need 1 <= ports <= n_around; got {ports} against {n_around}"
            )
        if width <= 0.0 or radius <= 0.5 * width:
            raise ValueError(
                f"need 0 < width < 2 * radius; got width {width}, radius {radius}"
            )

        radii = torch.linspace(
            radius - 0.5 * width,
            radius + 0.5 * width,
            n_across + 1,
            dtype=torch.float64,
        )
        angles = torch.arange(n_around, dtype=torch.float64) * (
            2.0 * torch.pi / n_around
        )
        nodes = torch.stack(
            [
                (radii[:, None] * torch.cos(angles)[None, :]).reshape(-1),
                (radii[:, None] * torch.sin(angles)[None, :]).reshape(-1),
                torch.zeros((n_across + 1) * n_around, dtype=torch.float64),
            ],
            dim=1,
        )

        def number(across: int, around: int) -> int:
            return across * n_around + around % n_around

        triangles = []
        for across in range(n_across):
            for around in range(n_around):
                inner_here = number(across, around)
                outer_here = number(across + 1, around)
                outer_next = number(across + 1, around + 1)
                inner_next = number(across, around + 1)
                triangles.append((inner_here, outer_here, outer_next))
                triangles.append((inner_here, outer_next, inner_next))

        lines = []
        tags = []
        for port in range(ports):
            around = round(port * n_around / ports)
            lines += [
                (number(across, around), number(across + 1, around))
                for across in range(n_across)
            ]
            tags += [port + 1] * n_across

        mesh = cls(
            nodes=nodes.to(device),
            triangles=torch.tensor(triangles, dtype=torch.int64, device=device),
            triangle_tags=torch.ones(len(triangles), dtype=torch.int64, device=device),
            lines=torch.tensor(lines, dtype=torch.int64, device=device),
            line_tags=torch.tensor(tags, dtype=torch.int64, device=device),
        )
        return mesh.align_to_lines()

    @classmethod
    def sphere(
        cls,
        *,
        radius: float,
        subdivisions: int,
        device: torch.device | str = "cpu",
    ) -> SurfaceMesh:
        """Build a closed sphere by repeatedly halving the edges of an icosahedron.

        Each pass splits every triangle into four and pushes the new nodes out
        to the sphere, so the surface has no rim and every triangle is wound
        outwards. It carries no line elements: the sphere is a scatterer, not a
        driven coil.

        Parameters
        ----------
        radius
            Sphere radius in metres.
        subdivisions
            Number of halving passes; the mesh has ``20 * 4 ** subdivisions``
            triangles.
        device
            Device the arrays are built on.

        Returns
        -------
        SurfaceMesh
            The closed surface.

        Raises
        ------
        ValueError
            If the radius is not positive or the pass count is negative.
        """
        if radius <= 0.0:
            raise ValueError(f"need a positive radius; got {radius}")
        if subdivisions < 0:
            raise ValueError(
                f"need a non-negative number of subdivisions; got {subdivisions}"
            )

        rectangle = ((-1.0, _GOLDEN), (1.0, _GOLDEN), (-1.0, -_GOLDEN), (1.0, -_GOLDEN))
        nodes = [[short, long, 0.0] for short, long in rectangle]
        nodes += [[0.0, short, long] for short, long in rectangle]
        nodes += [[long, 0.0, short] for short, long in rectangle]
        faces = [list(face) for face in _ICOSAHEDRON_FACES]

        for _ in range(subdivisions):
            halves: dict[tuple[int, int], int] = {}
            split = []
            for first, second, third in faces:
                one = _halve(nodes, halves, first, second)
                two = _halve(nodes, halves, second, third)
                three = _halve(nodes, halves, third, first)
                split += [
                    [first, one, three],
                    [second, two, one],
                    [third, three, two],
                    [one, two, three],
                ]
            faces = split

        coordinates = torch.tensor(nodes, dtype=torch.float64, device=device)
        coordinates = (
            radius
            * coordinates
            / torch.linalg.vector_norm(coordinates, dim=1, keepdim=True)
        )
        return cls(
            nodes=coordinates,
            triangles=torch.tensor(faces, dtype=torch.int64, device=device),
            triangle_tags=torch.ones(len(faces), dtype=torch.int64, device=device),
            lines=torch.empty((0, 2), dtype=torch.int64, device=device),
            line_tags=torch.empty((0,), dtype=torch.int64, device=device),
        )


def _halve(
    nodes: list[list[float]], halves: dict[tuple[int, int], int], left: int, right: int
) -> int:
    """Return the index of the node halfway along an edge, making it if needed."""
    key = (min(left, right), max(left, right))
    if key not in halves:
        halves[key] = len(nodes)
        nodes.append(
            [(nodes[left][axis] + nodes[right][axis]) / 2 for axis in range(3)]
        )
    return halves[key]


def _section(text: str, name: str) -> list[str] | None:
    """Return the records between ``$name`` and ``$Endname``, or None if absent."""
    match = re.search(rf"^\${name}$(.*?)^\$End{name}$", text, re.DOTALL | re.MULTILINE)
    if match is None:
        return None
    return [record for record in match.group(1).splitlines() if record.strip()]
