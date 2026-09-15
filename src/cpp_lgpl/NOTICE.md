# DIRECTFN — licence notice

Everything under this directory is the DIRECTFN singular-integral code as
distributed in MARIE 3.0 (<https://github.com/cloudmrhub/marie-tools>), together
with the wrappers and bindings that build it.

The vendored sources are unmodified. The repository stores text with LF line
endings, and the upstream files use CRLF, so that is the one difference from
them: `diff --strip-trailing-cr` against a MARIE checkout reports nothing.

Those sources carry the notice

> Licensing: This code is distributed under the GNU LGPL license.

and name **Athanasios Polimeridis** as their author. The distribution states no
version of the LGPL, and none is assumed here.

The code implements the direct evaluation method of:

- A. G. Polimeridis and T. V. Yioultsis, "On the direct evaluation of weakly
  singular integrals in Galerkin mixed potential integral equation
  formulations", IEEE Trans. Antennas Propag. 56 (2008) 3011–3019.
- A. G. Polimeridis and J. R. Mosig, "Complete semi-analytical treatment of
  weakly singular integrals on planar triangles via the direct evaluation
  method", Int. J. Numer. Methods Eng. 83 (2010) 1625–1650.

## Why this is a separate module

mariepy is MIT. These sources are not, so they build as `mariepy._directfn`,
their own extension module, and nothing here is compiled into `mariepy._ext`.
`module_directfn.cpp` and the `rwg_namespace_*.cpp` wrappers are derivatives of
the sources they bind and include, so they live here and carry the same licence.

## What is in each directory

| Path | Content |
|---|---|
| `directfn_vie/` | The voxel family: coincident, edge-adjacent and vertex-adjacent faces of cubic cells, used by the body kernel |
| `directfn_rwg/` | The triangle family: `direct_ws_st_rwg`, `direct_ws_ea_rwg` and `direct_ws_va_rwg`, used by the coil matrix |
| `rwg_namespace_*.cpp` | One wrapper per RWG source, giving it a namespace |
| `module_directfn.cpp` | The pybind11 bindings |

## Modifications

No line of the vendored sources is changed. `directfn_vie/src_cpp/main.cpp`, a
standalone driver, and the MATLAB `*_mex.cpp` gateways are not carried.

The three RWG sources each define `get_source_coords_mat` and the
`coefficients_*` family at file scope with external linkage, so they cannot be
linked into one module as they stand. Each is therefore included into a wrapper
that opens a namespace around it, with the system headers hoisted above the
namespace so the standard library keeps global linkage. The sources themselves
are untouched.

`GL_1D` is declared by the RWG headers and defined nowhere in that family;
MARIE computes its Gauss-Legendre nodes in MATLAB and passes them in. mariepy
passes them from `mariepy.quadrature.gauss_legendre_1d`.
