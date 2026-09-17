/**
 * @file convolution.cpp
 * @brief The Fourier-domain multiply of the body operators, N and K.
 *
 * Each output component at a cell of the extended grid is a signed sum of
 * input components at the same cell, each weighted by one symbol. A symbol is
 * a Tucker core expanded through three circulant factors, and is never
 * materialised: along each z-line it is the third factor contracted with the
 * core already contracted at that x and y, which fits in cache. Each z-line of
 * the input is read whole before its output is written, so the multiply runs in
 * place on the transformed current. x-slabs are independent and
 * partition across threads.
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using complex_t = std::complex<double>;
using ComplexArray = py::array_t<complex_t, py::array::c_style | py::array::forcecast>;

struct Symbol {
    const complex_t *core;
    std::size_t r1, r2, r3;
    const complex_t *f1, *f2; // (r1, L1), (r2, L2)
    std::vector<double> f3_re, f3_im; // (r3, L3), split so the z-loops vectorise
};

struct Term {
    std::size_t row, column, symbol;
    double scale;
};

// An AVX2 clone beside the baseline one, chosen by the loader, where the
// toolchain can build one: the z-loops are where the product spends its time.
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define MARIEPY_CLONES __attribute__((target_clones("arch=haswell", "default")))
#else
#define MARIEPY_CLONES
#endif

MARIEPY_CLONES
void multiply_slabs(std::size_t begin, std::size_t end, const std::vector<Symbol> &symbols,
                    const std::vector<Term> &terms, std::size_t n_components,
                    std::size_t l1, std::size_t l2, std::size_t l3, complex_t *data)
{
    const std::size_t n_symbols = symbols.size();
    std::size_t widest = 0;
    for (const auto &s : symbols) {
        widest = std::max(widest, s.r3);
    }
    std::vector<complex_t> pencil(widest); // a core contracted at x and y
    std::vector<double> line_re(n_symbols * l3), line_im(n_symbols * l3);
    std::vector<double> in_re(n_components * l3), in_im(n_components * l3);
    std::vector<double> out_re(n_components * l3), out_im(n_components * l3);
    std::vector<std::vector<complex_t>> slabs(n_symbols);
    for (std::size_t s = 0; s < n_symbols; ++s) {
        slabs[s].resize(symbols[s].r2 * symbols[s].r3);
    }
    const std::size_t volume = l1 * l2 * l3;

    for (std::size_t x = begin; x < end; ++x) {
        for (std::size_t s = 0; s < n_symbols; ++s) {
            const Symbol &sym = symbols[s];
            auto &out = slabs[s];
            std::fill(out.begin(), out.end(), complex_t(0.0));
            for (std::size_t a = 0; a < sym.r1; ++a) {
                const complex_t w = sym.f1[a * l1 + x];
                const complex_t *row = sym.core + a * sym.r2 * sym.r3;
                for (std::size_t bc = 0; bc < sym.r2 * sym.r3; ++bc) {
                    out[bc] += w * row[bc];
                }
            }
        }
        for (std::size_t y = 0; y < l2; ++y) {
            for (std::size_t s = 0; s < n_symbols; ++s) {
                const Symbol &sym = symbols[s];
                const auto &xs = slabs[s];
                std::fill(pencil.begin(), pencil.begin() + sym.r3, complex_t(0.0));
                for (std::size_t b = 0; b < sym.r2; ++b) {
                    const complex_t w = sym.f2[b * l2 + y];
                    const complex_t *row = xs.data() + b * sym.r3;
                    for (std::size_t c = 0; c < sym.r3; ++c) {
                        pencil[c] += w * row[c];
                    }
                }
                double *const lr = line_re.data() + s * l3;
                double *const li = line_im.data() + s * l3;
                std::fill(lr, lr + l3, 0.0);
                std::fill(li, li + l3, 0.0);
                for (std::size_t c = 0; c < sym.r3; ++c) {
                    const double wr = pencil[c].real();
                    const double wi = pencil[c].imag();
                    const double *const fr = sym.f3_re.data() + c * l3;
                    const double *const fi = sym.f3_im.data() + c * l3;
                    for (std::size_t z = 0; z < l3; ++z) {
                        lr[z] += wr * fr[z] - wi * fi[z];
                        li[z] += wr * fi[z] + wi * fr[z];
                    }
                }
            }
            const std::size_t offset = (x * l2 + y) * l3;
            for (std::size_t c = 0; c < n_components; ++c) {
                const complex_t *const cell = data + c * volume + offset;
                double *const ir = in_re.data() + c * l3;
                double *const ii = in_im.data() + c * l3;
                for (std::size_t z = 0; z < l3; ++z) {
                    ir[z] = cell[z].real();
                    ii[z] = cell[z].imag();
                }
            }
            std::fill(out_re.begin(), out_re.end(), 0.0);
            std::fill(out_im.begin(), out_im.end(), 0.0);
            for (const Term &t : terms) {
                const double *const lr = line_re.data() + t.symbol * l3;
                const double *const li = line_im.data() + t.symbol * l3;
                const double *const ir = in_re.data() + t.column * l3;
                const double *const ii = in_im.data() + t.column * l3;
                double *const orr = out_re.data() + t.row * l3;
                double *const oi = out_im.data() + t.row * l3;
                const double scale = t.scale;
                for (std::size_t z = 0; z < l3; ++z) {
                    orr[z] += scale * (lr[z] * ir[z] - li[z] * ii[z]);
                    oi[z] += scale * (lr[z] * ii[z] + li[z] * ir[z]);
                }
            }
            for (std::size_t c = 0; c < n_components; ++c) {
                complex_t *const cell = data + c * volume + offset;
                const double *const orr = out_re.data() + c * l3;
                const double *const oi = out_im.data() + c * l3;
                for (std::size_t z = 0; z < l3; ++z) {
                    cell[z] = complex_t(orr[z], oi[z]);
                }
            }
        }
    }
}

void multiply_symbols(py::array_t<complex_t, py::array::c_style> buffer,
                      const std::vector<ComplexArray> &cores,
                      const std::vector<ComplexArray> &factors,
                      const std::vector<std::size_t> &rows,
                      const std::vector<std::size_t> &columns,
                      const std::vector<std::size_t> &which,
                      const std::vector<double> &scales, std::size_t n_threads)
{
    if (buffer.ndim() != 4) {
        throw std::invalid_argument("the buffer must be (components, L1, L2, L3)");
    }
    const std::size_t n_components = static_cast<std::size_t>(buffer.shape(0));
    const std::size_t l1 = static_cast<std::size_t>(buffer.shape(1));
    const std::size_t l2 = static_cast<std::size_t>(buffer.shape(2));
    const std::size_t l3 = static_cast<std::size_t>(buffer.shape(3));
    if (factors.size() != 3 * cores.size()) {
        throw std::invalid_argument("each core needs three factors");
    }
    std::vector<Symbol> symbols;
    symbols.reserve(cores.size());
    for (std::size_t s = 0; s < cores.size(); ++s) {
        const auto &core = cores[s];
        const auto &f1 = factors[3 * s];
        const auto &f2 = factors[3 * s + 1];
        const auto &f3 = factors[3 * s + 2];
        if (core.ndim() != 3 || f1.ndim() != 2 || f2.ndim() != 2 || f3.ndim() != 2
            || f1.shape(0) != core.shape(0) || f2.shape(0) != core.shape(1)
            || f3.shape(0) != core.shape(2)
            || static_cast<std::size_t>(f1.shape(1)) != l1
            || static_cast<std::size_t>(f2.shape(1)) != l2
            || static_cast<std::size_t>(f3.shape(1)) != l3) {
            throw std::invalid_argument("a symbol does not match the buffer's grid");
        }
        Symbol symbol{core.data(),
                      static_cast<std::size_t>(core.shape(0)),
                      static_cast<std::size_t>(core.shape(1)),
                      static_cast<std::size_t>(core.shape(2)),
                      f1.data(),
                      f2.data(),
                      {},
                      {}};
        const std::size_t entries = symbol.r3 * l3;
        symbol.f3_re.resize(entries);
        symbol.f3_im.resize(entries);
        const complex_t *f3_data = f3.data();
        for (std::size_t i = 0; i < entries; ++i) {
            symbol.f3_re[i] = f3_data[i].real();
            symbol.f3_im[i] = f3_data[i].imag();
        }
        symbols.push_back(std::move(symbol));
    }
    const std::size_t n_terms = rows.size();
    if (columns.size() != n_terms || which.size() != n_terms || scales.size() != n_terms) {
        throw std::invalid_argument("rows, columns, symbols and scales must align");
    }
    std::vector<Term> terms(n_terms);
    for (std::size_t t = 0; t < n_terms; ++t) {
        if (rows[t] >= n_components || columns[t] >= n_components
            || which[t] >= symbols.size()) {
            throw std::invalid_argument("a term indexes past the components or symbols");
        }
        terms[t] = {rows[t], columns[t], which[t], scales[t]};
    }
    complex_t *data = buffer.mutable_data();
    {
        py::gil_scoped_release release;
        const std::size_t workers =
            std::max<std::size_t>(1, std::min<std::size_t>(n_threads, l1));
        const std::size_t share = (l1 + workers - 1) / workers;
        std::vector<std::thread> threads;
        for (std::size_t worker = 1; worker < workers; ++worker) {
            const std::size_t begin = worker * share;
            const std::size_t end = std::min(begin + share, l1);
            if (begin >= end) {
                break;
            }
            threads.emplace_back(multiply_slabs, begin, end, std::cref(symbols),
                                 std::cref(terms), n_components, l1, l2, l3, data);
        }
        multiply_slabs(0, std::min(share, l1), symbols, terms, n_components, l1, l2, l3,
                       data);
        for (auto &thread : threads) {
            thread.join();
        }
    }
}

} // namespace

void bind_convolution(py::module_ &module)
{
    module.def("multiply_symbols", &multiply_symbols, py::arg("buffer"), py::arg("cores"),
               py::arg("factors"), py::arg("rows"), py::arg("columns"),
               py::arg("symbols"), py::arg("scales"), py::arg("n_threads"),
               R"doc(Multiply a transformed current by Tucker-compressed symbols, in place.

``buffer`` is ``(C, L1, L2, L3)`` complex128 and C-contiguous. Symbol ``s`` is
``cores[s]`` of shape ``(r1, r2, r3)`` expanded through ``factors[3 s]``,
``factors[3 s + 1]`` and ``factors[3 s + 2]``, of shapes ``(r1, L1)``,
``(r2, L2)`` and ``(r3, L3)``. On return, component ``i`` of each cell holds
the sum over terms ``t`` with ``rows[t] == i`` of
``scales[t] * symbol[symbols[t]] * component columns[t]`` of the input.)doc");
}
