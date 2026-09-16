"""Files in MARIE's own formats, written from arrays for the readers to read back.

MARIE's example bodies and coils are not carried, so the readers are checked
against files built here in the layouts MARIE writes.
"""

import numpy as np


def write_marie_body(path, permittivity, conductivity, *, pitch, origin, tissue=None):
    """Write a body as MARIE's ``RHBM`` structure, as ``ndgrid`` and MATLAB lay it out.

    ``tissue`` becomes ``idxS``, one-based and column-major; without it the
    file carries none, as some of MARIE's own do.
    """
    from scipy.io import savemat

    shape = permittivity.shape
    axes = [origin[axis] + pitch * np.arange(n) for axis, n in enumerate(shape)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    rhbm = {
        "name": "synthetic",
        "r": centres,
        "epsilon_r": permittivity,
        "sigma_e": conductivity,
        "rhos": np.ones(shape, dtype=np.uint8),
    }
    if tissue is not None:
        indices = np.flatnonzero(tissue.ravel(order="F")) + 1
        rhbm["idxS"] = indices.astype(np.int32)[:, None]
    savemat(path, {"RHBM": rhbm})


def write_gmsh22(path, mesh):
    """Write a surface mesh as GMSH 2.2 ASCII, each element tagged physical and elementary."""
    nodes = mesh.nodes.cpu().numpy()
    lines = mesh.lines.cpu().numpy()
    line_tags = mesh.line_tags.cpu().numpy()
    triangles = mesh.triangles.cpu().numpy()
    triangle_tags = mesh.triangle_tags.cpu().numpy()

    records = []
    for pair, tag in zip(lines, line_tags, strict=True):
        records.append(f"1 2 {tag} {tag} {pair[0] + 1} {pair[1] + 1}")
    for triple, tag in zip(triangles, triangle_tags, strict=True):
        records.append(f"2 2 {tag} {tag} " + " ".join(str(n + 1) for n in triple))

    text = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(nodes))]
    text += [f"{i + 1} {x:.17g} {y:.17g} {z:.17g}" for i, (x, y, z) in enumerate(nodes)]
    text += ["$EndNodes", "$Elements", str(len(records))]
    text += [f"{i + 1} {record}" for i, record in enumerate(records)]
    text += ["$EndElements", ""]
    path.write_text("\n".join(text))
