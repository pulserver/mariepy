/**
 * @file module.cpp
 * @brief The compiled kernels, bound as `mariepy._ext`.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_convolution(py::module_ &module);
void bind_coupling(py::module_ &module);
void bind_volume(py::module_ &module);

PYBIND11_MODULE(_ext, module)
{
    module.doc() = "Precompiled kernels for mariepy";
    bind_convolution(module);
    bind_coupling(module);
    bind_volume(module);
}
