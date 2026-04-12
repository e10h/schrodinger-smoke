/**
 * @file test_schrodinger.cpp
 * @brief Unit tests for the Schrödinger evolution step (Alg. 2).
 *
 * Algorithm 2 of Chern et al. 2016:
 *
 *     function SCHRÖDINGER(ψ, dt, ℏ):
 *         ψ̂     ← FFT3D(ψ)
 *         ψ̂_k   ← exp(i · ℏ · λ_k · dt / 2) · ψ̂_k
 *         return InvFFT3D(ψ̂)
 *
 * with the continuous-Laplacian eigenvalues from Eq. 18:
 *
 *     λ_k = −(2π)² · (k_x²/L_x² + k_y²/L_y² + k_z²/L_z²).
 *
 * The tests pin three properties:
 *
 *   1. **L² conservation.** The step is unitary: the discrete integral of
 *      |ψ₁|² + |ψ₂|² must be preserved to machine precision.
 *
 *   2. **DC invariance.** A spatially constant field has λ = 0, so the
 *      propagator is the identity on the only non-zero Fourier coefficient.
 *
 *   3. **Single plane wave.** A pure Fourier mode acquires exactly the
 *      phase factor exp(i · ℏ · λ · dt / 2). This pins the ℏ, dt/2 and sign
 *      conventions together.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"
#include "isf/schrodinger.hpp"

#include <cmath>
#include <complex>
#include <numbers>

using isf::Grid3D;
using isf::schrodinger_step;
using Catch::Matchers::WithinAbs;

namespace {

/// @brief Sum of |ψ₁(x)|² + |ψ₂(x)|² over all vertices.
double l2_norm_squared(const Grid3D& g)
{
    double s = 0.0;
    for (std::size_t n = 0; n < g.size(); ++n) {
        s += std::norm(g.psi1(n)) + std::norm(g.psi2(n));
    }
    return s;
}

} // namespace

TEST_CASE("Schrödinger step conserves the discrete L² norm",
          "[schrodinger]")
{
    Grid3D g(8, 8, 8, 1.0, 1.0, 1.0);

    // Populate ψ with a deterministic, non-trivial field.
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double a = 0.3 * i + 0.7 * j + 1.1 * k;
                const double b = 0.9 * i - 0.2 * j + 0.5 * k;
                g.psi1(i, j, k) = { std::sin(a),  std::cos(b) };
                g.psi2(i, j, k) = { std::cos(a), -std::sin(b) };
            }
        }
    }

    const double before = l2_norm_squared(g);
    schrodinger_step(g, /*dt=*/0.05, /*hbar=*/0.1);
    const double after  = l2_norm_squared(g);

    REQUIRE_THAT(after, WithinAbs(before, 1e-11 * before));
}

TEST_CASE("Schrödinger step leaves a spatially constant field unchanged",
          "[schrodinger]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);

    const std::complex<double> c1(0.6, 0.8);
    const std::complex<double> c2(0.8, -0.6);
    for (std::size_t n = 0; n < g.size(); ++n) {
        g.psi1(n) = c1;
        g.psi2(n) = c2;
    }

    schrodinger_step(g, /*dt=*/0.05, /*hbar=*/0.1);

    for (std::size_t n = 0; n < g.size(); ++n) {
        REQUIRE_THAT(g.psi1(n).real(), WithinAbs(c1.real(), 1e-12));
        REQUIRE_THAT(g.psi1(n).imag(), WithinAbs(c1.imag(), 1e-12));
        REQUIRE_THAT(g.psi2(n).real(), WithinAbs(c2.real(), 1e-12));
        REQUIRE_THAT(g.psi2(n).imag(), WithinAbs(c2.imag(), 1e-12));
    }
}

TEST_CASE("Schrödinger step applies exp(i·ℏ·λ·dt/2) to a single plane wave",
          "[schrodinger]")
{
    const int    N    = 8;
    const double L    = 1.0;
    const double hbar = 0.1;
    const double dt   = 0.05;
    Grid3D g(N, N, N, L, L, L);

    // Plane wave ψ₁(x) = exp(2πi·m·x/L), ψ₂ = 0.
    // Chosen small so it's well inside the band limit.
    const int mx = 2, my = -1, mz = 1;
    const double two_pi = 2.0 * std::numbers::pi_v<double>;

    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double phase =
                    two_pi * (static_cast<double>(mx * i) / N +
                              static_cast<double>(my * j) / N +
                              static_cast<double>(mz * k) / N);
                g.psi1(i, j, k) = { std::cos(phase), std::sin(phase) };
                g.psi2(i, j, k) = { 0.0, 0.0 };
            }
        }
    }

    // Expected eigenvalue of the continuous Laplacian on this mode:
    //   λ = −(2π)² · (mx² + my² + mz²) / L²          (Eq. 18 with Lx=Ly=Lz=L)
    const double pi2   = std::numbers::pi_v<double> * std::numbers::pi_v<double>;
    const double k_sq  = static_cast<double>(mx*mx + my*my + mz*mz) / (L * L);
    const double lambda = -4.0 * pi2 * k_sq;

    // Propagator phase φ = ℏ · λ · dt / 2.
    const double phi = hbar * lambda * dt / 2.0;
    const std::complex<double> rot{ std::cos(phi), std::sin(phi) };

    // Snapshot the input so we can compare against the analytic answer.
    std::vector<std::complex<double>> original(g.size());
    for (std::size_t n = 0; n < g.size(); ++n) original[n] = g.psi1(n);

    schrodinger_step(g, dt, hbar);

    for (std::size_t n = 0; n < g.size(); ++n) {
        const auto expected = original[n] * rot;
        REQUIRE_THAT(g.psi1(n).real(), WithinAbs(expected.real(), 1e-12));
        REQUIRE_THAT(g.psi1(n).imag(), WithinAbs(expected.imag(), 1e-12));
        REQUIRE_THAT(g.psi2(n).real(), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(g.psi2(n).imag(), WithinAbs(0.0, 1e-12));
    }
}
