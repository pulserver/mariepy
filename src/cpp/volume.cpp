/**
 * @file volume.cpp
 * @brief The volume-volume rule of the body kernels, N and K.
 *
 * The rule itself is built in `mariepy.vie._volume_volume`: the separation
 * nodes and one row of weights per basis-function pair. This evaluates the
 * kernel at every offset plus every node and contracts it with the weights.
 * Offsets are independent, so they partition across threads.
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <thread>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "green.hpp"

namespace py = pybind11;

namespace {

using mariepy::complex_t;

void volume_rows(std::size_t begin, std::size_t end, bool electric,
                 const double *offsets, const double *nodes, std::size_t n_nodes,
                 const double *weights, std::size_t n_pairs, double wavenumber,
                 complex_t *out)
{
    const std::size_t n_components = electric ? 6 : 3;
    std::vector<complex_t> value(n_components);
    for (std::size_t row = begin; row < end; ++row) {
        complex_t *target = out + row * n_pairs * n_components;
        for (std::size_t node = 0; node < n_nodes; ++node) {
            double separation[3];
            for (int axis = 0; axis < 3; ++axis) {
                separation[axis] = offsets[row * 3 + axis] + nodes[node * 3 + axis];
            }
            if (electric) {
                mariepy::green_n(separation, wavenumber, value.data());
            } else {
                mariepy::green_k(separation, wavenumber, value.data());
            }
            for (std::size_t pair = 0; pair < n_pairs; ++pair) {
                const double weight = weights[pair * n_nodes + node];
                if (weight == 0.0) {
                    continue;
                }
                for (std::size_t c = 0; c < n_components; ++c) {
                    target[pair * n_components + c] += weight * value[c];
                }
            }
        }
    }
}

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<complex_t> volume_volume(const Array &offsets, const Array &nodes,
                                     const Array &weights, double wavenumber,
                                     bool electric)
{
    const std::size_t rows = static_cast<std::size_t>(offsets.shape(0));
    const std::size_t n_nodes = static_cast<std::size_t>(nodes.shape(0));
    const std::size_t n_pairs = static_cast<std::size_t>(weights.shape(0));
    if (offsets.ndim() != 2 || offsets.shape(1) != 3 || nodes.ndim() != 2
        || nodes.shape(1) != 3 || weights.ndim() != 2
        || static_cast<std::size_t>(weights.shape(1)) != n_nodes) {
        throw std::invalid_argument(
            "offsets (n, 3), nodes (m, 3) and weights (p, m) are required");
    }
    const std::size_t n_components = electric ? 6 : 3;
    py::array_t<complex_t> result({static_cast<py::ssize_t>(rows),
                                   static_cast<py::ssize_t>(n_pairs),
                                   static_cast<py::ssize_t>(n_components)});
    complex_t *out = result.mutable_data();
    std::fill(out, out + rows * n_pairs * n_components, complex_t(0.0));
    const double *offset_data = offsets.data();
    const double *node_data = nodes.data();
    const double *weight_data = weights.data();
    {
        py::gil_scoped_release release;
        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware == 0) {
            hardware = 1;
        }
        const std::size_t workers = std::max<std::size_t>(
            1, std::min<std::size_t>(hardware, rows));
        const std::size_t share = (rows + workers - 1) / workers;
        std::vector<std::thread> threads;
        for (std::size_t worker = 1; worker < workers; ++worker) {
            const std::size_t begin = worker * share;
            const std::size_t end = std::min(begin + share, rows);
            if (begin >= end) {
                break;
            }
            threads.emplace_back(volume_rows, begin, end, electric, offset_data,
                                 node_data, n_nodes, weight_data, n_pairs, wavenumber,
                                 out);
        }
        volume_rows(0, std::min(share, rows), electric, offset_data, node_data,
                    n_nodes, weight_data, n_pairs, wavenumber, out);
        for (auto &thread : threads) {
            thread.join();
        }
    }
    return result;
}

} // namespace

void bind_volume(py::module_ &module)
{
    module.def("volume_volume", &volume_volume, py::arg("offsets"), py::arg("nodes"),
               py::arg("weights"), py::arg("wavenumber"), py::arg("electric"),
               R"doc(Contract the N or K kernel with a volume-volume rule at each offset.

``offsets`` is ``(n, 3)``, ``nodes`` is ``(m, 3)`` and ``weights`` is
``(p, m)``, all C-ordered float64 in metres and without the Jacobian. Returns
``(n, p, 6)`` for N, in the order xx, xy, xz, yy, yz, zz, or ``(n, p, 3)`` for
K, complex128.)doc");
}
