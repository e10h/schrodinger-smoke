/**
 * @file test_fft.cpp
 * @brief Unit tests for the isf::fft3 wrapper around pocketfft.
 *
 * The wrapper is a thin type-safe shim: in-place complex-to-complex 3D FFT
 * over a Grid3D-shaped buffer. These tests pin the contract we need
 * elsewhere in the solver:
 *
 *   1. Forward∘inverse is an identity (round-trip within 1e-13).
 *   2. A spatially constant field has all energy in the DC bin.
 *   3. A discrete plane wave exp(2πi·(mx·i/Nx + my·j/Ny + mz·k/Nz))
 *      has all energy in the single Fourier bin (mx, my, mz).
 *
 * Together these cover sign convention, normalization, and layout.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/fft.hpp"
#include "isf/grid.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using isf::Grid3D;
using isf::fft3_forward;
using isf::fft3_inverse;
using Catch::Matchers::WithinAbs;

namespace {

/// @brief Build a contiguous row-major ℂ buffer sized for a grid.
std::vector<std::complex<double>> make_buffer(const Grid3D& g) {
    return std::vector<std::complex<double>>(g.size(),
                                              std::complex<double>(0.0, 0.0));
}

} // namespace

TEST_CASE("fft3 forward followed by inverse recovers the input", "[fft]")
{
    Grid3D g(8, 4, 2, 1.0, 1.0, 1.0);
    auto data = make_buffer(g);

    // Deterministic "random-looking" input.
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double re = std::sin(0.3 * i + 0.7 * j + 1.1 * k);
                const double im = std::cos(0.9 * i - 0.2 * j + 0.5 * k);
                data[g.index(i, j, k)] = {re, im};
            }
        }
    }
    auto original = data;

    fft3_forward(data.data(), g.nx(), g.ny(), g.nz());
    fft3_inverse(data.data(), g.nx(), g.ny(), g.nz());

    for (std::size_t n = 0; n < data.size(); ++n) {
        REQUIRE_THAT(data[n].real(), WithinAbs(original[n].real(), 1e-13));
        REQUIRE_THAT(data[n].imag(), WithinAbs(original[n].imag(), 1e-13));
    }
}

TEST_CASE("fft3 of a constant field concentrates all energy in the DC bin",
          "[fft]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
    auto data = make_buffer(g);

    const std::complex<double> c(2.5, -1.25);
    for (auto& v : data) v = c;

    fft3_forward(data.data(), g.nx(), g.ny(), g.nz());

    // With unit forward normalization (fct = 1), the DC bin equals
    // the sum of all samples = c · N.
    const double N = static_cast<double>(g.size());
    REQUIRE_THAT(data[0].real(), WithinAbs(c.real() * N, 1e-12));
    REQUIRE_THAT(data[0].imag(), WithinAbs(c.imag() * N, 1e-12));

    for (std::size_t n = 1; n < data.size(); ++n) {
        REQUIRE_THAT(data[n].real(), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(data[n].imag(), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("fft3 of a discrete plane wave hits a single Fourier bin",
          "[fft]")
{
    Grid3D g(8, 8, 8, 1.0, 1.0, 1.0);
    auto data = make_buffer(g);

    // Use the forward-convention sign: ψ(x) = exp(-2πi·m·x/N) lands in
    // bin m. We'll build it with the opposite sign so we can verify
    // which bin lights up and therefore pin the sign convention.
    const int mx = 2, my = 3, mz = 1;
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double phase =
                    two_pi * (static_cast<double>(mx * i) / g.nx() +
                              static_cast<double>(my * j) / g.ny() +
                              static_cast<double>(mz * k) / g.nz());
                // ψ(x) = exp(+i·phase). pocketfft's "forward" convention is
                // Σ_x ψ(x) · exp(-i·2π·k·x/N), so this lands in bin
                // (mx, my, mz).
                data[g.index(i, j, k)] = {std::cos(phase), std::sin(phase)};
            }
        }
    }

    fft3_forward(data.data(), g.nx(), g.ny(), g.nz());

    const double N = static_cast<double>(g.size());
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const auto v = data[g.index(i, j, k)];
                if (i == mx && j == my && k == mz) {
                    REQUIRE_THAT(v.real(), WithinAbs(N, 1e-10));
                    REQUIRE_THAT(v.imag(), WithinAbs(0.0, 1e-10));
                } else {
                    REQUIRE_THAT(std::abs(v), WithinAbs(0.0, 1e-10));
                }
            }
        }
    }
}
