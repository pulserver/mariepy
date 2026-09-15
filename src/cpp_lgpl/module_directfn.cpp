/**
 * @file module_directfn.cpp
 * @brief The DIRECTFN singular integrals, bound as `mariepy._directfn`.
 *
 * DIRECTFN evaluates the four-dimensional weakly singular integrals that arise
 * when two elements touch: coincident (self term), sharing an edge, or sharing
 * a vertex. Two families are bound here, both as distributed in MARIE 3.0 and
 * both under the LGPL, which is why they build as this module rather than into
 * `mariepy._ext`:
 *
 * - the voxel family, over the faces of two cubic cells, used by the body
 *   kernel where cells touch;
 * - the RWG family, over two planar triangles, used by the coil matrix.
 *
 * The voxel sources link together as they stand and keep the global namespace.
 * The RWG sources each redefine the same helpers, so each is compiled inside a
 * namespace of its own; see `rwg_namespace_st.cpp` and its siblings.
 *
 * DIRECTFN takes its Gauss-Legendre nodes and weights from the caller:
 * `GL_1D` is declared by the RWG headers but defined nowhere in that family.
 * `mariepy.quadrature.gauss_legendre_1d` supplies them.
 *
 * The voxel family's `kernel_type` selects which reduced kernel is integrated,
 * and `Kernels.cpp` branches on 0 through 8. It is not the same index as the
 * four surface-surface kernels the body operator reduces to: MARIE's N
 * operator reaches for `kernel_type` 1 to 4, and its K operator for 5 and 6.
 */

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using dcomplex = std::complex<double>;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// The voxel family, compiled from `directfn_vie/` into the global namespace.
void create_ST(const double r1[], const double r2[], const double r3[],
               const double r4[], int N1, int N2, int N3, int N4, double k0,
               double dx, double rq_c[], double rp_c[], double nq[], double np[],
               int ker_type, int l, int lp, dcomplex I[]);

void create_EA(const double r1[], const double r2[], const double r3[],
               const double r4[], const double r5[], const double r6[],
               const int N1, const int N2, const int N3, const int N4, double k0,
               double dx, double rq_c[], double rp_c[], double nq[], double np[],
               int ker_type, int l, int lp, dcomplex I[]);

void create_VA(const double r1[], const double r2[], const double r3[],
               const double r4[], const double r5[], const double r6[],
               const double r7[], const int N1, const int N2, const int N3,
               const int N4, double k0, double dx, double rq_c[], double rp_c[],
               double nq[], double np[], int ker_type, int l, int lp,
               dcomplex I[]);

// The RWG family, one namespace per source.
namespace marie_directfn_rwg_st {
void direct_ws_st_rwg(const double r1[], const double r2[], const double r3[],
                      const double ko, const int Np_1D, const double w[],
                      const double z[], dcomplex I_DE[]);
}
namespace marie_directfn_rwg_ea {
void direct_ws_ea_rwg(const double r1[], const double r2[], const double r3[],
                      const double r4[], const double ko, const int N_theta,
                      const int N_psi, const double w_theta[],
                      const double z_theta[], const double w_psi[],
                      const double z_psi[], dcomplex I_DE[]);
}
namespace marie_directfn_rwg_va {
void direct_ws_va_rwg(const double r1[], const double r2[], const double r3[],
                      const double r4[], const double r5[], const double ko,
                      const int N_theta_p, const int N_theta_q, const int N_psi,
                      const double w_theta_p[], const double z_theta_p[],
                      const double w_theta_q[], const double z_theta_q[],
                      const double w_psi[], const double z_psi[],
                      dcomplex I_DE[]);
}

namespace {

/// Number of entries the RWG kernels write: one per pair of triangle edges.
constexpr std::size_t RWG_TERMS = 9;

/// Check an array's shape and return a pointer to its data.
const double *rows(const Array &array, py::ssize_t expected_rows,
                   const char *name)
{
    if (array.ndim() != 2 || array.shape(0) != expected_rows ||
        array.shape(1) != 3) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (" +
            std::to_string(expected_rows) + ", 3), got " +
            std::to_string(array.ndim()) + " axes");
    }
    return array.data();
}

/// Check a one-dimensional array of the given length and return its data.
const double *line(const Array &array, py::ssize_t expected, const char *name)
{
    if (array.ndim() != 1 || array.shape(0) != expected) {
        throw std::invalid_argument(
            std::string(name) + " must have shape (" +
            std::to_string(expected) + ",)");
    }
    return array.data();
}

/// Check that a quadrature rule's weights and nodes agree in length.
void same_length(const Array &weights, const Array &nodes, const char *name)
{
    if (weights.ndim() != 1 || nodes.ndim() != 1 ||
        weights.shape(0) != nodes.shape(0)) {
        throw std::invalid_argument(
            std::string(name) + ": weights and nodes must be one-dimensional "
                                "and the same length");
    }
    if (weights.shape(0) < 1) {
        throw std::invalid_argument(std::string(name) + ": the rule is empty");
    }
}

/// Copy a mutable vector out of an array, since DIRECTFN takes non-const points.
std::vector<double> mutable_point(const Array &array, const char *name)
{
    const double *data = line(array, 3, name);
    return {data[0], data[1], data[2]};
}

py::array_t<dcomplex> to_array(const dcomplex values[], std::size_t count)
{
    auto result = py::array_t<dcomplex>(static_cast<py::ssize_t>(count));
    auto view = result.mutable_unchecked<1>();
    for (std::size_t i = 0; i < count; ++i) {
        view(static_cast<py::ssize_t>(i)) = values[i];
    }
    return result;
}

dcomplex voxel_self(const Array &vertices, const Array &centre_source,
                    const Array &centre_observer, const Array &normal_source,
                    const Array &normal_observer, double wavenumber,
                    double voxel_size, int n_points, int kernel_type, int l,
                    int lp)
{
    const double *r = rows(vertices, 4, "vertices");
    std::vector<double> rq_c = mutable_point(centre_source, "centre_source");
    std::vector<double> rp_c = mutable_point(centre_observer, "centre_observer");
    std::vector<double> nq = mutable_point(normal_source, "normal_source");
    std::vector<double> np_ = mutable_point(normal_observer, "normal_observer");

    dcomplex value = 0.0;
    create_ST(&r[0], &r[3], &r[6], &r[9], n_points, n_points, n_points, n_points,
              wavenumber, voxel_size, rq_c.data(), rp_c.data(), nq.data(),
              np_.data(), kernel_type, l, lp, &value);
    return value;
}

dcomplex voxel_edge(const Array &vertices, const Array &centre_source,
                    const Array &centre_observer, const Array &normal_source,
                    const Array &normal_observer, double wavenumber,
                    double voxel_size, int n_points, int kernel_type, int l,
                    int lp)
{
    const double *r = rows(vertices, 6, "vertices");
    std::vector<double> rq_c = mutable_point(centre_source, "centre_source");
    std::vector<double> rp_c = mutable_point(centre_observer, "centre_observer");
    std::vector<double> nq = mutable_point(normal_source, "normal_source");
    std::vector<double> np_ = mutable_point(normal_observer, "normal_observer");

    dcomplex value = 0.0;
    create_EA(&r[0], &r[3], &r[6], &r[9], &r[12], &r[15], n_points, n_points,
              n_points, n_points, wavenumber, voxel_size, rq_c.data(),
              rp_c.data(), nq.data(), np_.data(), kernel_type, l, lp, &value);
    return value;
}

dcomplex voxel_vertex(const Array &vertices, const Array &centre_source,
                      const Array &centre_observer, const Array &normal_source,
                      const Array &normal_observer, double wavenumber,
                      double voxel_size, int n_points, int kernel_type, int l,
                      int lp)
{
    const double *r = rows(vertices, 7, "vertices");
    std::vector<double> rq_c = mutable_point(centre_source, "centre_source");
    std::vector<double> rp_c = mutable_point(centre_observer, "centre_observer");
    std::vector<double> nq = mutable_point(normal_source, "normal_source");
    std::vector<double> np_ = mutable_point(normal_observer, "normal_observer");

    dcomplex value = 0.0;
    create_VA(&r[0], &r[3], &r[6], &r[9], &r[12], &r[15], &r[18], n_points,
              n_points, n_points, n_points, wavenumber, voxel_size, rq_c.data(),
              rp_c.data(), nq.data(), np_.data(), kernel_type, l, lp, &value);
    return value;
}

py::array_t<dcomplex> triangle_self(const Array &vertices, double wavenumber,
                                    const Array &weights, const Array &nodes)
{
    const double *r = rows(vertices, 3, "vertices");
    same_length(weights, nodes, "rule");

    dcomplex values[RWG_TERMS] = {};
    marie_directfn_rwg_st::direct_ws_st_rwg(
        &r[0], &r[3], &r[6], wavenumber, static_cast<int>(weights.shape(0)),
        weights.data(), nodes.data(), values);
    return to_array(values, RWG_TERMS);
}

py::array_t<dcomplex> triangle_edge(const Array &vertices, double wavenumber,
                                    const Array &weights_theta,
                                    const Array &nodes_theta,
                                    const Array &weights_psi,
                                    const Array &nodes_psi)
{
    const double *r = rows(vertices, 4, "vertices");
    same_length(weights_theta, nodes_theta, "theta rule");
    same_length(weights_psi, nodes_psi, "psi rule");

    dcomplex values[RWG_TERMS] = {};
    marie_directfn_rwg_ea::direct_ws_ea_rwg(
        &r[0], &r[3], &r[6], &r[9], wavenumber,
        static_cast<int>(weights_theta.shape(0)),
        static_cast<int>(weights_psi.shape(0)), weights_theta.data(),
        nodes_theta.data(), weights_psi.data(), nodes_psi.data(), values);
    return to_array(values, RWG_TERMS);
}

py::array_t<dcomplex> triangle_vertex(
    const Array &vertices, double wavenumber, const Array &weights_theta_p,
    const Array &nodes_theta_p, const Array &weights_theta_q,
    const Array &nodes_theta_q, const Array &weights_psi, const Array &nodes_psi)
{
    const double *r = rows(vertices, 5, "vertices");
    same_length(weights_theta_p, nodes_theta_p, "theta_p rule");
    same_length(weights_theta_q, nodes_theta_q, "theta_q rule");
    same_length(weights_psi, nodes_psi, "psi rule");

    dcomplex values[RWG_TERMS] = {};
    marie_directfn_rwg_va::direct_ws_va_rwg(
        &r[0], &r[3], &r[6], &r[9], &r[12], wavenumber,
        static_cast<int>(weights_theta_p.shape(0)),
        static_cast<int>(weights_theta_q.shape(0)),
        static_cast<int>(weights_psi.shape(0)), weights_theta_p.data(),
        nodes_theta_p.data(), weights_theta_q.data(), nodes_theta_q.data(),
        weights_psi.data(), nodes_psi.data(), values);
    return to_array(values, RWG_TERMS);
}

}  // namespace

PYBIND11_MODULE(_directfn, module)
{
    module.doc() =
        "DIRECTFN singular integrals, as distributed in MARIE 3.0 (LGPL)";

    module.def("voxel_self", &voxel_self, py::arg("vertices"),
               py::arg("centre_source"), py::arg("centre_observer"),
               py::arg("normal_source"), py::arg("normal_observer"),
               py::arg("wavenumber"), py::arg("voxel_size"),
               py::arg("n_points"), py::arg("kernel_type"), py::arg("l"),
               py::arg("lp"),
               "Coincident faces of one voxel. `vertices` is (4, 3), ordered as "
               "DIRECTFN expects. `kernel_type` runs 0 to 8 and selects the "
               "reduced kernel; `l` and `lp` run 0 to 3 and select the scalar "
               "term of the testing and basis function.");

    module.def("voxel_edge", &voxel_edge, py::arg("vertices"),
               py::arg("centre_source"), py::arg("centre_observer"),
               py::arg("normal_source"), py::arg("normal_observer"),
               py::arg("wavenumber"), py::arg("voxel_size"),
               py::arg("n_points"), py::arg("kernel_type"), py::arg("l"),
               py::arg("lp"),
               "Faces of two voxels sharing an edge. `vertices` is (6, 3).");

    module.def("voxel_vertex", &voxel_vertex, py::arg("vertices"),
               py::arg("centre_source"), py::arg("centre_observer"),
               py::arg("normal_source"), py::arg("normal_observer"),
               py::arg("wavenumber"), py::arg("voxel_size"),
               py::arg("n_points"), py::arg("kernel_type"), py::arg("l"),
               py::arg("lp"),
               "Faces of two voxels sharing a vertex. `vertices` is (7, 3).");

    module.def("triangle_self", &triangle_self, py::arg("vertices"),
               py::arg("wavenumber"), py::arg("weights"), py::arg("nodes"),
               "One triangle against itself. `vertices` is (3, 3); returns the "
               "nine edge-pair terms.");

    module.def("triangle_edge", &triangle_edge, py::arg("vertices"),
               py::arg("wavenumber"), py::arg("weights_theta"),
               py::arg("nodes_theta"), py::arg("weights_psi"),
               py::arg("nodes_psi"),
               "Two triangles sharing an edge. `vertices` is (4, 3), the shared "
               "edge first; returns the nine edge-pair terms.");

    module.def("triangle_vertex", &triangle_vertex, py::arg("vertices"),
               py::arg("wavenumber"), py::arg("weights_theta_p"),
               py::arg("nodes_theta_p"), py::arg("weights_theta_q"),
               py::arg("nodes_theta_q"), py::arg("weights_psi"),
               py::arg("nodes_psi"),
               "Two triangles sharing a vertex. `vertices` is (5, 3), the "
               "shared vertex first; returns the nine edge-pair terms.");
}
