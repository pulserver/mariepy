"""MARIE's own coupling sources, compiled and called without MATLAB.

The 24 sources in ``tests/marie/`` are the reference `PLAN.md` sets for the
coupling kernels. They are MARIE 3.0's own files, kept here so that the
comparison runs wherever a C++ compiler does, and skipped only when none is on
the path. `THIRD_PARTY.md` records them and their licence.

They answer the transcription question -- did we copy MARIE faithfully -- and
not the physics question, which the Mie series, reciprocity and the power
balance answer. A port that reproduces MARIE reproduces its errors with it, so
both layers are kept.

Two things stand between the sources and a compiler. They include ``mex.h``,
which is absent, for the one type ``mxComplexDouble``; a four-line header
supplies it. And each defines the same three helpers at file scope, so linking
them together is a multiple definition; giving those three internal linkage in
each copy is what makes one library out of twenty-four translation units.
"""

import ctypes
import functools
import itertools
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SOURCES = Path(__file__).parent / "marie"

MEX_SHIM = """#ifndef MEX_H
#define MEX_H
typedef struct { double real; double imag; } mxComplexDouble;
#endif
"""

HELPERS = (
    ("\nvoid get_source_coords_mat", "\nstatic void get_source_coords_mat"),
    ("\ndouble vec_norm_l2", "\nstatic double vec_norm_l2"),
    ("\ndouble * compute_edge_length_v", "\nstatic double * compute_edge_length_v"),
)

SIGNATURE = (
    "(mxComplexDouble *, double *, double *, double *, double *, double *,"
    " double *, double *, double, const size_t, const size_t, double)"
)


def variants():
    """Yield ``(operator, component, basis_term)`` in the order the library indexes them."""
    return itertools.product("NK", range(3), range(4))


def _name(operator, component, term):
    return (
        f"Assemble_rwg_coupling_matrix_{operator}_{'xyz'[component]}"
        f"{['', '1', '2', '3'][term]}"
    )


def reason():
    """Say why the MARIE comparison cannot run, or None if it can."""
    if shutil.which("g++") is None:
        return "no C++ compiler on the path"
    return None


@functools.lru_cache(maxsize=1)
def build():
    """Compile the 24 sources into one shared library and return a caller.

    The library is built once per session, since all 24 variants live in it.

    Returns
    -------
    callable
        ``call(index, corners, points, triangle_rule, cell_rule, cell_size,
        wavenumber)`` returning the variant's ``(n_rwg, n_points)`` complex
        matrix, with ``index`` the position in :func:`variants`.
    """
    work = Path(tempfile.mkdtemp(prefix="mariepy-parity-"))
    (work / "mex.h").write_text(MEX_SHIM)

    names = [_name(*variant) for variant in variants()]
    for name in names:
        text = (SOURCES / f"{name}.cpp").read_text()
        for external, internal in HELPERS:
            text = text.replace(external, internal)
        (work / f"{name}.cpp").write_text(text)

    declarations = "\n".join(f"extern void {name}{SIGNATURE};" for name in names)
    branches = "\n".join(
        f"    if (which == {index}) {{ {name}(out, points, r_p, r_n, r_2, r_3,"
        " triangle, cell, size, n_points, n_rwg, wavenumber); return; }"
        for index, name in enumerate(names)
    )
    (work / "harness.cpp").write_text(
        '#include "mex.h"\n#include <cstddef>\n'
        f"{declarations}\n"
        'extern "C" void marie_coupling(int which, mxComplexDouble * out, double * points,'
        " double * r_p, double * r_n, double * r_2, double * r_3, double * triangle,"
        " double * cell, double size, size_t n_points, size_t n_rwg, double wavenumber)\n"
        f"{{\n{branches}\n}}\n"
    )

    library = work / "libmarie_coupling.so"
    subprocess.run(
        [
            "g++",
            "-O2",
            "-fPIC",
            "-shared",
            f"-I{work}",
            "-o",
            str(library),
            str(work / "harness.cpp"),
            *[str(work / f"{name}.cpp") for name in names],
        ],
        check=True,
    )

    handle = ctypes.CDLL(str(library))
    handle.marie_coupling.restype = None
    handle.marie_coupling.argtypes = (
        [ctypes.c_int]
        + [np.ctypeslib.ndpointer(dtype=np.float64)] * 8
        + [ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double]
    )

    def call(index, corners, points, triangle_rule, cell_rule, cell_size, wavenumber):
        n_rwg, n_points = corners.shape[0], points.shape[0]
        out = np.zeros(2 * n_points * n_rwg, dtype=np.float64)
        vertices = [
            np.ascontiguousarray(corners[:, slot], dtype=np.float64)
            for slot in range(4)
        ]
        handle.marie_coupling(
            index,
            out,
            np.ascontiguousarray(points, dtype=np.float64),
            *vertices,
            np.ascontiguousarray(triangle_rule, dtype=np.float64),
            np.ascontiguousarray(cell_rule, dtype=np.float64),
            cell_size,
            n_points,
            n_rwg,
            wavenumber,
        )
        return (out[0::2] + 1j * out[1::2]).reshape(n_rwg, n_points)

    return call


def packed_rules(triangle_weights, barycentric, cell_weights, cell_nodes):
    """Pack the two quadrature rules the way MARIE's sources index them.

    MARIE's ``dunavant_rule.m`` halves Burkardt's weights, which is what carries
    the RWG basis function's own ``1 / (2 A)``; the same halving is applied here
    so the comparison is against MARIE as it runs.
    """
    triangle = np.concatenate(
        [
            [len(triangle_weights)],
            0.5 * np.asarray(triangle_weights),
            np.asarray(barycentric)[:, 0],
            np.asarray(barycentric)[:, 1],
            np.asarray(barycentric)[:, 2],
        ]
    )
    cell = np.concatenate(
        [[len(cell_weights)], np.asarray(cell_weights), np.asarray(cell_nodes)]
    )
    return triangle, cell
