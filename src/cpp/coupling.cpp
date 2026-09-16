/**
 * @file coupling.cpp
 * @brief The coil-to-body coupling kernels, N and K.
 *
 * Ported from MARIE 3.0's 24 sources under
 * `src_integral_equations/src_svie/Cpp_Assembly/src/`, which differ only in
 * field component, basis term and operator. `tests/marie/` keeps those sources
 * as the oracle this is checked against.
 *
 * The integral is over one basis function's two triangles and its observer's
 * cell: a Dunavant rule on each triangle, a tensor-product Gauss rule over the
 * cell. Rows are independent, so they partition across threads.
 */

#include <cmath>
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

using complex_t = mariepy::complex_t;

using mariepy::green_k;
using mariepy::green_n;
using mariepy::kDyadicIndex;

struct Rules {
    const double *triangle_weights;
    const double *barycentric; // (n_triangle, 3)
    std::size_t n_triangle;
    const double *cell_weights; // already multiplied by the basis-term factor
    const double *cell_offsets; // (n_cell, 3)
    std::size_t n_cell;
};

void couple_rows(std::size_t begin, std::size_t end, bool electric,
                 const double *corners, const double *points, double wavenumber,
                 complex_t electric_scaling, const Rules &rules, complex_t *out)
{
    for (std::size_t row = begin; row < end; ++row) {
        const double *corner = corners + row * 12;
        const double *free_vertex[2] = {corner + 0, corner + 3};
        const double *shared[2] = {corner + 6, corner + 9};

        double edge[3];
        for (int axis = 0; axis < 3; ++axis) {
            edge[axis] = shared[1][axis] - shared[0][axis];
        }
        const double length =
            std::sqrt(edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]);

        complex_t total[3] = {complex_t(0.0), complex_t(0.0), complex_t(0.0)};

        for (int triangle = 0; triangle < 2; ++triangle) {
            const double sign = triangle == 0 ? 1.0 : -1.0;
            for (std::size_t q = 0; q < rules.n_triangle; ++q) {
                const double *lambda = rules.barycentric + q * 3;
                double source[3], rho[3];
                for (int axis = 0; axis < 3; ++axis) {
                    source[axis] = lambda[0] * free_vertex[triangle][axis]
                                   + lambda[1] * shared[0][axis]
                                   + lambda[2] * shared[1][axis];
                    rho[axis] = sign * (source[axis] - free_vertex[triangle][axis]);
                }

                for (std::size_t g = 0; g < rules.n_cell; ++g) {
                    double separation[3];
                    for (int axis = 0; axis < 3; ++axis) {
                        separation[axis] = points[row * 3 + axis]
                                           + rules.cell_offsets[g * 3 + axis]
                                           - source[axis];
                    }

                    const double weight =
                        rules.cell_weights[g] * rules.triangle_weights[q];

                    if (electric) {
                        complex_t dyad[6];
                        green_n(separation, wavenumber, dyad);
                        for (int component = 0; component < 3; ++component) {
                            complex_t value(0.0);
                            for (int column = 0; column < 3; ++column) {
                                value += dyad[kDyadicIndex[component][column]]
                                         * rho[column];
                            }
                            total[component] += weight * value;
                        }
                    } else {
                        complex_t curl[3];
                        green_k(separation, wavenumber, curl);
                        total[0] += weight * (curl[1] * rho[2] - curl[2] * rho[1]);
                        total[1] += weight * (curl[2] * rho[0] - curl[0] * rho[2]);
                        total[2] += weight * (curl[0] * rho[1] - curl[1] * rho[0]);
                    }
                }
            }
        }

        for (int component = 0; component < 3; ++component) {
            complex_t value = total[component] * length;
            if (electric) {
                value /= electric_scaling;
            }
            out[row * 3 + component] = value;
        }
    }
}

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<complex_t> couple(const Array &corners, const Array &points,
                              const Array &triangle_weights, const Array &barycentric,
                              const Array &cell_weights, const Array &cell_offsets,
                              double wavenumber, complex_t electric_scaling,
                              bool electric)
{
    const auto corner_view = corners.unchecked<3>();
    const auto point_view = points.unchecked<2>();
    const std::size_t rows = static_cast<std::size_t>(corner_view.shape(0));

    if (static_cast<std::size_t>(point_view.shape(0)) != rows) {
        throw std::invalid_argument("corners and points must pair row for row");
    }

    Rules rules{triangle_weights.data(),
                barycentric.data(),
                static_cast<std::size_t>(triangle_weights.shape(0)),
                cell_weights.data(),
                cell_offsets.data(),
                static_cast<std::size_t>(cell_weights.shape(0))};

    py::array_t<complex_t> result({static_cast<py::ssize_t>(rows),
                                   static_cast<py::ssize_t>(3)});
    complex_t *out = result.mutable_data();
    const double *corner_data = corners.data();
    const double *point_data = points.data();

    {
        py::gil_scoped_release release;

        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware == 0) {
            hardware = 1;
        }
        std::size_t workers = std::min<std::size_t>(hardware, rows);
        if (workers <= 1) {
            couple_rows(0, rows, electric, corner_data, point_data, wavenumber,
                        electric_scaling, rules, out);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(workers - 1);
            const std::size_t share = (rows + workers - 1) / workers;
            for (std::size_t worker = 1; worker < workers; ++worker) {
                const std::size_t begin = worker * share;
                const std::size_t end = std::min(begin + share, rows);
                if (begin >= end) {
                    break;
                }
                threads.emplace_back(couple_rows, begin, end, electric, corner_data,
                                     point_data, wavenumber, electric_scaling,
                                     std::cref(rules), out);
            }
            couple_rows(0, std::min(share, rows), electric, corner_data, point_data,
                        wavenumber, electric_scaling, rules, out);
            for (auto &thread : threads) {
                thread.join();
            }
        }
    }

    return result;
}

} // namespace

void bind_coupling(py::module_ &module)
{
    module.def("couple", &couple, py::arg("corners"), py::arg("points"),
               py::arg("triangle_weights"), py::arg("barycentric"),
               py::arg("cell_weights"), py::arg("cell_offsets"),
               py::arg("wavenumber"), py::arg("electric_scaling"), py::arg("electric"),
               R"doc(Integrate the N or K coupling kernel over each basis function.

Every array is C-ordered float64. ``corners`` is ``(n, 4, 3)``, ``points`` is
``(n, 3)``, ``barycentric`` is ``(n_triangle, 3)`` and ``cell_offsets`` is
``(n_cell, 3)``. ``cell_weights`` already carries the basis-term factor, and
``triangle_weights`` is already halved as MARIE's ``dunavant_rule.m`` halves it.
Returns ``(n, 3)`` complex128.)doc");
}
