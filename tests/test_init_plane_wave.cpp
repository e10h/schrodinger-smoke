/**
 * @file test_init_plane_wave.cpp
 * @brief Unit tests for the plane-wave initial condition (Sec. 3.2).
 *
 * From Section 3.2 of Chern et al. 2016:
 *
 *     ψ^(k,ℏ)(x) = exp(i⟨k,x⟩ − iℏ|k|²t/2)
 *
 * At t = 0 we set ψ = (exp(i⟨k,x⟩), 0)ᵀ, which under ISF recovers the
 * uniform velocity field v = ℏk. For the discrete periodic grid to
 * represent this plane wave without boundary artefacts, each component
 * k_α must be an integer multiple of 2π/L_α.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"
#include "isf/init.hpp"
#include "isf/pressure.hpp"  // compute_scaled_divergence

#include <cmath>
#include <complex>
#include <numbers>

using isf::Grid3D;
using isf::compute_scaled_divergence;
using isf::init_plane_wave;
using Catch::Matchers::WithinAbs;

TEST_CASE("init_plane_wave sets ψ to the analytic plane wave",
          "[init][plane-wave]")
{
    const int    N = 8;
    const double L = 2.0;
    Grid3D g(N, N, N, L, L, L);

    // Integer mode so the plane wave is exactly periodic on the grid.
    const int    mx = 1, my = -2, mz = 3;
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double kx = two_pi * mx / L;
    const double ky = two_pi * my / L;
    const double kz = two_pi * mz / L;

    init_plane_wave(g, kx, ky, kz);

    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double x = i * g.dx();
                const double y = j * g.dy();
                const double z = k * g.dz();
                const double phase = kx * x + ky * y + kz * z;
                REQUIRE_THAT(g.psi1(i, j, k).real(),
                             WithinAbs(std::cos(phase), 1e-14));
                REQUIRE_THAT(g.psi1(i, j, k).imag(),
                             WithinAbs(std::sin(phase), 1e-14));
                REQUIRE_THAT(g.psi2(i, j, k).real(), WithinAbs(0.0, 1e-15));
                REQUIRE_THAT(g.psi2(i, j, k).imag(), WithinAbs(0.0, 1e-15));
            }
        }
    }
}

TEST_CASE("init_plane_wave produces a divergence-free ψ", "[init][plane-wave]")
{
    const int    N = 8;
    const double L = 1.0;
    Grid3D g(N, N, N, L, L, L);

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    init_plane_wave(g, two_pi * 2 / L, two_pi * 1 / L, two_pi * -3 / L);

    const auto xi = compute_scaled_divergence(g);
    // Tolerance reflects arg/trig roundoff amplified by the 1/dx² factor in
    // the discrete divergence formula — measured ~6e-13 on this grid.
    for (double v : xi) {
        REQUIRE_THAT(v, WithinAbs(0.0, 1e-11));
    }
}

TEST_CASE("init_plane_wave yields |ψ|² = 1 at every vertex",
          "[init][plane-wave]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
    init_plane_wave(g, 0.0, 0.0, 0.0);  // Zero-k special case: ψ₁ ≡ 1.

    for (std::size_t n = 0; n < g.size(); ++n) {
        const double ns =
            std::norm(g.psi1(n)) + std::norm(g.psi2(n));
        REQUIRE_THAT(ns, WithinAbs(1.0, 1e-15));
    }
}
