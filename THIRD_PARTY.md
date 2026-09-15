# Third-party sources

Code ported or kept from other projects, with its licence. An entry is added
when the first file from that source lands, and its copyright notice is
reproduced below the table at the same time.

| Source | Licence | Used as | Milestone |
|---|---|---|---|
| MARIE 3.0, <https://github.com/cloudmrhub/marie-tools> | MIT | Ported to Python and C++ | 1 |
| DIRECTFN singular integrals, as distributed in MARIE 3.0 | LGPL | Separate extension module, source kept with its notice | 1 |
| Dunavant triangle quadrature, as distributed in MARIE 3.0 | LGPL | Rules taken from Dunavant (1985); the files themselves are not carried | 1 |
| TT-Toolbox `dmrg_cross`, as distributed in MARIE 3.0 | MIT | Ported to Python | 2 |

## DIRECTFN singular integrals

Kept in `src/cpp_lgpl/`, byte-identical, with the bindings and namespace
wrappers that build them as `mariepy._directfn`. The sources carry the notice

> Licensing: This code is distributed under the GNU LGPL license.

and name Athanasios Polimeridis as their author. The distribution states no
version of the LGPL, and none is assumed. `src/cpp_lgpl/NOTICE.md` records what
is carried, what is not, and the two changes made to build it outside MATLAB.

