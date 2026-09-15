/**
 * @file rwg_namespace_st.cpp
 * @brief The st RWG singular integral, given a namespace of its own.
 *
 * The three RWG sources each define the same helper functions at file scope
 * with external linkage, so they cannot link together as they stand. Including
 * each into its own namespace resolves that at compile time and leaves the
 * vendored source byte-identical.
 *
 * The system headers the source includes are hoisted above the namespace, so
 * that the standard library keeps global linkage.
 *
 * This file is part of the LGPL extension: it is a derivative of the source it
 * includes, and carries that source's licence.
 */

#include <complex>
#include <math.h>

namespace marie_directfn_rwg_st {
#include "directfn_rwg/direct_ws_st_rwg.cpp"
}
