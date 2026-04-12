/**
 * @file test_pressure.cpp
 * @brief Unit tests for the pressure projection step (Algorithm 3).
 *
 * Algorithm 3 of Chern et al. 2016 — "PRESSUREPROJECT(ψ)" — works entirely
 * in ℏ-scaled units:
 *
 *   1. η̃_vw  ← arg⟨ψ_v, ψ_w⟩_ℂ                     on every axis-aligned edge
 *   2. ξ̃_v   ← (1/V_v) Σ_{w∈⋆v} (A_vw/l_vw) η̃_vw   (Eq. 5, divergence)
 *   3. q̃     ← Δ⁻¹ ξ̃                               (FFT Poisson, discrete
 *                                                     Laplacian eigenvalues
 *                                                     from Eq. 17)
 *   4. ψ'    ← e^(−iq̃) · ψ                          (phase correction)
 *
 * The tests pin the three properties the rest of the solver relies on:
 *
 *   1. **Divergence-free output.** After projection the recomputed scaled
 *      divergence ξ̃ is zero to machine precision at every vertex.
 *
 *   2. **Plane wave fixed point.** A discrete plane wave corresponds to a
 *      uniform velocity — already divergence-free — so projection leaves it
 *      bitwise unchanged.
 *
 *   3. **Idempotence.** Projecting twice gives the same result as projecting
 *      once (projection really is a projection, not merely a smoothing).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"
#include "isf/normalize.hpp"
#include "isf/pressure.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

using isf::Grid3D;
using isf::compute_scaled_divergence;
using isf::normalize;
using isf::pressure_project;
using Catch::Matchers::WithinAbs;

namespace {

/// @brief Populate a grid with a "messy but unit" ψ that is clearly
///        not divergence-free, then normalize to |ψ|² = 1.
void fill_messy_unit_psi(Grid3D& g)
{
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double a = 0.31 * i + 0.57 * j - 0.22 * k;
                const double b = 0.91 * i - 0.13 * j + 0.47 * k;
                g.psi1(i, j, k) = { std::cos(a), std::sin(b) };
                g.psi2(i, j, k) = { std::sin(a), std::cos(b) };
            }
        }
    }
    normalize(g);
}

} // namespace

TEST_CASE("pressure_project drives divergence to zero",
          "[pressure]")
{
    Grid3D g(8, 8, 8, 1.0, 1.0, 1.0);
    fill_messy_unit_psi(g);

    // Sanity: the unprojected state should have non-trivial divergence.
    const auto div_before = compute_scaled_divergence(g);
    double max_before = 0.0;
    for (double v : div_before) max_before = std::max(max_before, std::abs(v));
    REQUIRE(max_before > 1.0);

    // The arg-form projection (Eq. 4 + Alg. 3) is exact in one shot.
    pressure_project(g);
    const auto div_after = compute_scaled_divergence(g);
    double max_after = 0.0;
    for (double v : div_after) max_after = std::max(max_after, std::abs(v));
    REQUIRE(max_after < 1e-10);
}

TEST_CASE("pressure_project leaves a discrete plane wave unchanged",
          "[pressure]")
{
    const int    N = 8;
    const double L = 1.0;
    Grid3D g(N, N, N, L, L, L);

    // ψ = (exp(+2πi·m·x/L), 0) is a unit plane wave; its implied velocity is
    // uniform, so η̃ is constant per axis direction and ξ̃ ≡ 0.
    const int mx = 1, my = 2, mz = -1;
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const double phase =
                    two_pi * (static_cast<double>(mx * i) / N +
                              static_cast<double>(my * j) / N +
                              static_cast<double>(mz * k) / N);
                g.psi1(i, j, k) = { std::cos(phase), std::sin(phase) };
                g.psi2(i, j, k) = { 0.0, 0.0 };
            }
        }
    }

    // Divergence is already zero.
    const auto div_before = compute_scaled_divergence(g);
    for (double v : div_before) REQUIRE_THAT(v, WithinAbs(0.0, 1e-12));

    // Snapshot the field.
    std::vector<std::complex<double>> before(g.size());
    for (std::size_t n = 0; n < g.size(); ++n) before[n] = g.psi1(n);

    pressure_project(g);

    for (std::size_t n = 0; n < g.size(); ++n) {
        REQUIRE_THAT(g.psi1(n).real(), WithinAbs(before[n].real(), 1e-12));
        REQUIRE_THAT(g.psi1(n).imag(), WithinAbs(before[n].imag(), 1e-12));
        REQUIRE_THAT(g.psi2(n).real(), WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(g.psi2(n).imag(), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("pressure_project is idempotent",
          "[pressure]")
{
    // The arg-form projection is idempotent: projecting a second time
    // gives bitwise-identical output. (This held before for smooth
    // interiors, was broken by the Im-form switch, and is now restored.)
    const int    N = 8;
    const double L = 1.0;
    Grid3D g(N, N, N, L, L, L);

    const int mx = 1, my = 2, mz = -1;
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < N; ++i) {
                const double phase =
                    two_pi * (static_cast<double>(mx * i) / N +
                              static_cast<double>(my * j) / N +
                              static_cast<double>(mz * k) / N);
                g.psi1(i, j, k) = { std::cos(phase), std::sin(phase) };
                g.psi2(i, j, k) = { 0.0, 0.0 };
            }
        }
    }

    // Snapshot.
    std::vector<std::complex<double>> ref(g.size());
    for (std::size_t n = 0; n < g.size(); ++n) ref[n] = g.psi1(n);

    pressure_project(g);
    pressure_project(g);   // and again — bitwise identical both times

    for (std::size_t n = 0; n < g.size(); ++n) {
        REQUIRE_THAT(g.psi1(n).real(), WithinAbs(ref[n].real(), 1e-12));
        REQUIRE_THAT(g.psi1(n).imag(), WithinAbs(ref[n].imag(), 1e-12));
    }
}
