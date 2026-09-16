# Third-party sources

Code ported or kept from other projects, with its licence. An entry is added
when the first file from that source lands, and its copyright notice is
reproduced below the table at the same time.

| Source | Licence | Used as | Milestone |
|---|---|---|---|
| MARIE 3.0, <https://github.com/cloudmrhub/marie-tools> | MIT | Ported to Python and C++ | 1 |
| MARIE 3.0 coupling sources, `tests/marie/` | MIT | Kept verbatim as the test oracle for the coupling kernels; test-only, never in the wheel | 1 |
| MARIE 3.0 wire coupling sources, `tests/marie/wire/` | MIT | Kept verbatim as the test oracle for the wire coupling kernels, compiled with the falling-ramp change `PLAN.md` records; test-only, never in the wheel | 3 |
| DIRECTFN singular integrals, as distributed in MARIE 3.0 | LGPL | Separate extension module, source kept with its notice | 1 |
| Dunavant triangle quadrature, as distributed in MARIE 3.0 | LGPL | Rules taken from Dunavant (1985); the files themselves are not carried | 1 |
| TT-Toolbox `dmrg_cross` and the helpers it calls, as distributed in MARIE 3.0 | MIT | Ported to Python in `mariepy/tt.py` | 2 |

## DIRECTFN singular integrals

Kept in `src/cpp_lgpl/`, unmodified apart from the LF line endings this
repository stores text with, together with the bindings and namespace wrappers
that build them as `mariepy._directfn`. The sources carry the notice

> Licensing: This code is distributed under the GNU LGPL license.

and name Athanasios Polimeridis as their author. The distribution states no
version of the LGPL, and none is assumed. `src/cpp_lgpl/NOTICE.md` records what
is carried, what is not, and the two changes made to build it outside MATLAB.

## TT-Toolbox

`src/mariepy/tt.py` ports `cross/dmrg_cross_gpu.m`, `core/maxvol2.m`,
`core/my_chop2.m`, `core/reort.m` and `core/tt_ind2sub.m` from the copy of
TT-Toolbox MARIE 3.0 distributes under `TT_QTT/`, which carries this notice:

> Copyright (C) 2009-2012 Ivan Oseledets, Sergey Dolgov, Vladimir Kazeev,
> Thomas Mach, Olga Lebedeva, Dmitry Savostyanov, Pavel Zhlobich, Le Song
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in
> the Software without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
> of the Software, and to permit persons to whom the Software is furnished to do
> so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

