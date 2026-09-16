/**
 * @file green.hpp
 * @brief The dyadic Green kernels the compiled integrals share.
 *
 * Mirrors `mariepy.vie.green_n` and `mariepy.vie.green_k`.
 */

#pragma once

#include <cmath>
#include <complex>

namespace mariepy {

using complex_t = std::complex<double>;

constexpr double kFourPi = 12.566370614359172;

/// The six distinct components of the double-curl dyadic Green kernel,
/// in the order xx, xy, xz, yy, yz, zz. Mirrors `mariepy.vie.green_n`.
inline void green_n(const double sep[3], double k, complex_t out[6])
{
    const double x = sep[0], y = sep[1], z = sep[2];
    const double r = std::sqrt(x * x + y * y + z * z);
    const complex_t envelope = std::exp(complex_t(0.0, -k * r)) / kFourPi;

    const double r2 = r * r, r3 = r2 * r, r4 = r3 * r, r5 = r4 * r;
    const complex_t ik(0.0, k);

    auto diagonal = [&](double c) {
        const double c2 = c * c;
        return envelope * (3.0 * c2 / r5 + 3.0 * ik * c2 / r4
                           - (1.0 + k * k * c2) / r3 - ik / r2);
    };
    auto off_diagonal = [&](double a, double b) {
        const double product = a * b;
        return envelope * (3.0 * product / r5 + 3.0 * ik * product / r4
                           - k * k * product / r3);
    };

    const complex_t gxx = diagonal(x), gyy = diagonal(y), gzz = diagonal(z);
    out[0] = -gyy - gzz;
    out[1] = off_diagonal(x, y);
    out[2] = off_diagonal(x, z);
    out[3] = -gxx - gzz;
    out[4] = off_diagonal(y, z);
    out[5] = -gxx - gyy;
}

/// The three components of the curl Green kernel. Mirrors `mariepy.vie.green_k`.
inline void green_k(const double sep[3], double k, complex_t out[3])
{
    const double r = std::sqrt(sep[0] * sep[0] + sep[1] * sep[1] + sep[2] * sep[2]);
    const complex_t envelope = std::exp(complex_t(0.0, -r * k)) / kFourPi;
    const complex_t radial = 1.0 / (r * r * r) + complex_t(0.0, k) / (r * r);
    const complex_t scale = envelope * radial;
    for (int axis = 0; axis < 3; ++axis) {
        out[axis] = scale * (-sep[axis]);
    }
}

/// Row-major index of the six stored dyad components, as `vie._DYADIC_INDEX`.
constexpr int kDyadicIndex[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

} // namespace mariepy
