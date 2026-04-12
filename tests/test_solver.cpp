/**
 * @file test_solver.cpp
 * @brief Integration tests for Algorithm 1 (the full basic ISF step) plus
 *        velocity recovery and the vortex-ring initializer.
 *
 * Algorithm 1 of Chern et al. 2016 is the composition
 *
 *     ψ ← Schrödinger(ψ, dt, ℏ)
 *     ψ ← Normalize(ψ)
 *     ψ ← PressureProject(ψ)
 *
 * repeated every time step. These tests exercise the whole chain end to
 * end:
 *
 *   1. **Plane-wave fixed point.** Init ψ to a plane wave (uniform flow
 *      v = ℏk). Every piece of Alg. 1 leaves a plane wave invariant (up
 *      to a global phase that cancels in the velocity recovery), so
 *      after N steps the recovered velocity must still be ℏk to
 *      round-off.
 *
 *   2. **|ψ|² preserved.** After a handful of solver steps, |ψ₁|² +
 *      |ψ₂|² is ≈ 1 everywhere. (Normalize guarantees this immediately
 *      after line 3; pressure projection is a unit-modulus phase
 *      rotation so it preserves it.)
 *
 *   3. **Vortex ring is a valid state.** The slab construction of
 *      Sec. 3.1 produces a ψ that, after normalize + projection, is
 *      unit-normalized and divergence-free.
 *
 *   4. **Vortex ring survives the solver.** After several solver steps
 *      the vortex-ring ψ still has |ψ|² ≈ 1 and residual divergence
 *      ≈ 0 — a basic sanity check that nothing blows up.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"
#include "isf/init.hpp"
#include "isf/pressure.hpp"
#include "isf/solver.hpp"
#include "isf/velocity.hpp"

#include <cmath>
#include <complex>
#include <numbers>

using isf::Grid3D;
using isf::VelocityField;
using isf::compute_scaled_divergence;
using isf::init_plane_wave;
using isf::init_vortex_ring_z;
using isf::recover_velocity;
using isf::solve_step;
using Catch::Matchers::WithinAbs;

TEST_CASE("recover_velocity on a plane wave yields v = ℏk",
          "[velocity][plane-wave]")
{
    // The arg-based edge 1-form (Eq. 4) gives arg⟨ψ_v, ψ_w⟩ = k·dx
    // exactly for a unit plane wave, so the recovered vertex velocity
    // is v = ℏk — exact to machine precision.
    const int    N    = 8;
    const double L    = 1.0;
    const double hbar = 0.1;
    Grid3D g(N, N, N, L, L, L);

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double kx = two_pi * 2 / L;   // integer mode (2, -1, 0)
    const double ky = two_pi * (-1) / L;
    const double kz = 0.0;
    init_plane_wave(g, kx, ky, kz);

    const VelocityField v = recover_velocity(g, hbar);

    const double v_x_expected = hbar * kx;
    const double v_y_expected = hbar * ky;
    const double v_z_expected = hbar * kz;

    for (std::size_t n = 0; n < g.size(); ++n) {
        REQUIRE_THAT(v.vx[n], WithinAbs(v_x_expected, 1e-11));
        REQUIRE_THAT(v.vy[n], WithinAbs(v_y_expected, 1e-11));
        REQUIRE_THAT(v.vz[n], WithinAbs(v_z_expected, 1e-11));
    }
}

TEST_CASE("solve_step is a fixed point on a plane wave", "[solver][plane-wave]")
{
    const int    N    = 8;
    const double L    = 1.0;
    const double hbar = 0.1;
    const double dt   = 0.05;
    Grid3D g(N, N, N, L, L, L);

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double kx = two_pi * 1 / L;
    const double ky = 0.0;
    const double kz = 0.0;
    init_plane_wave(g, kx, ky, kz);

    // Run ten solver steps.
    for (int step = 0; step < 10; ++step) {
        solve_step(g, dt, hbar);
    }

    // |ψ|² still one everywhere.
    for (std::size_t n = 0; n < g.size(); ++n) {
        const double ns = std::norm(g.psi1(n)) + std::norm(g.psi2(n));
        REQUIRE_THAT(ns, WithinAbs(1.0, 1e-12));
    }

    // Velocity still matches v = ℏk (arg-form, Eq. 4).
    const double v_x_expected = hbar * kx;
    const VelocityField v = recover_velocity(g, hbar);
    for (std::size_t n = 0; n < g.size(); ++n) {
        REQUIRE_THAT(v.vx[n], WithinAbs(v_x_expected, 1e-10));
        REQUIRE_THAT(v.vy[n], WithinAbs(0.0,          1e-10));
        REQUIRE_THAT(v.vz[n], WithinAbs(0.0,          1e-10));
    }
}

TEST_CASE("init_vortex_ring_z yields a normalized, divergence-free state",
          "[init][vortex-ring]")
{
    const int    N = 16;
    const double L = 1.0;
    Grid3D g(N, N, N, L, L, L);

    init_vortex_ring_z(g,
                       /*cx=*/ L / 2, /*cy=*/ L / 2, /*cz=*/ L / 2,
                       /*R =*/ 0.25,  /*r =*/ 0.05);

    // Unit norm everywhere.
    for (std::size_t n = 0; n < g.size(); ++n) {
        const double ns = std::norm(g.psi1(n)) + std::norm(g.psi2(n));
        REQUIRE_THAT(ns, WithinAbs(1.0, 1e-12));
    }

    // Divergence-free (within the same tolerance pressure_project achieves).
    const auto xi = compute_scaled_divergence(g);
    for (double val : xi) {
        REQUIRE_THAT(val, WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("solve_step keeps the vortex ring normalized and divergence-free",
          "[solver][vortex-ring]")
{
    const int    N    = 16;
    const double L    = 1.0;
    const double hbar = 0.02;
    const double dt   = 1.0 / 48.0;
    Grid3D g(N, N, N, L, L, L);

    init_vortex_ring_z(g,
                       /*cx=*/ L / 2, /*cy=*/ L / 2, /*cz=*/ L / 2,
                       /*R =*/ 0.25,  /*r =*/ 0.05);

    for (int step = 0; step < 5; ++step) {
        solve_step(g, dt, hbar);
    }

    for (std::size_t n = 0; n < g.size(); ++n) {
        const double ns = std::norm(g.psi1(n)) + std::norm(g.psi2(n));
        REQUIRE_THAT(ns, WithinAbs(1.0, 1e-11));
    }

    // The arg-form pressure projection is exact (one-shot) so
    // divergence stays at machine precision after each step.
    const auto xi = compute_scaled_divergence(g);
    for (double val : xi) {
        REQUIRE_THAT(val, WithinAbs(0.0, 1e-8));
    }
}
