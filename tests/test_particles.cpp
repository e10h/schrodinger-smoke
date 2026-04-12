/**
 * @file test_particles.cpp
 * @brief Unit tests for the CPU tracer-particle system.
 *
 * The particle system is a passive-tracer cloud advected by the
 * velocity field recovered from ψ. Two pieces are tested:
 *
 *   1. **Trilinear sampling** of a scalar field on the periodic grid.
 *      Pins the interpolation contract (exact at vertices, linear
 *      between them, periodic wrap for out-of-range positions).
 *
 *   2. **RK2 advection** of particles through a VelocityField. Pins
 *      the advection contract (zero-velocity no-op, constant-velocity
 *      drift, periodic wrap of positions).
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"
#include "isf/init.hpp"
#include "isf/velocity.hpp"
#include "viz/particles.hpp"

#include <cmath>
#include <numbers>
#include <vector>

using isf::Grid3D;
using isf::VelocityField;
using isf::init_plane_wave;
using isf::recover_velocity;
using viz::ParticleSystem;
using viz::advect_particles_rk2;
using viz::sample_trilinear;
using Catch::Matchers::WithinAbs;

namespace {

/// @brief Build a scalar field equal to f(x,y,z) = i + 10·j + 100·k,
///        so that every vertex has a unique, easily-checked value.
std::vector<double> index_field(const Grid3D& g)
{
    std::vector<double> f(g.size());
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                f[g.index(i, j, k)] =
                    static_cast<double>(i) +
                    10.0 * static_cast<double>(j) +
                    100.0 * static_cast<double>(k);
            }
        }
    }
    return f;
}

} // namespace

TEST_CASE("sample_trilinear reproduces vertex values at vertex positions",
          "[particles][trilinear]")
{
    Grid3D g(8, 8, 8, 1.0, 1.0, 1.0);
    const auto f = index_field(g);

    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                const double x = i * g.dx();
                const double y = j * g.dy();
                const double z = k * g.dz();
                const double expected =
                    static_cast<double>(i) +
                    10.0 * static_cast<double>(j) +
                    100.0 * static_cast<double>(k);
                REQUIRE_THAT(sample_trilinear(g, f, x, y, z),
                             WithinAbs(expected, 1e-12));
            }
        }
    }
}

TEST_CASE("sample_trilinear averages at the midpoint of two vertices",
          "[particles][trilinear]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
    const auto f = index_field(g);

    // Midpoint along x between (1,2,0) and (2,2,0):
    //   expected = 0.5 * (1 + 10*2 + 100*0 + 2 + 10*2 + 100*0) = 21.5
    const double x = 1.5 * g.dx();
    const double y = 2.0 * g.dy();
    const double z = 0.0;
    REQUIRE_THAT(sample_trilinear(g, f, x, y, z), WithinAbs(21.5, 1e-12));
}

TEST_CASE("sample_trilinear wraps periodically for out-of-range positions",
          "[particles][trilinear]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
    const auto f = index_field(g);

    // x = L + small → should sample near x = 0.
    const double dx = g.dx();
    const double x_inside  = 0.25 * dx;
    const double x_wrapped = 1.0  + 0.25 * dx;
    REQUIRE_THAT(sample_trilinear(g, f, x_wrapped, 0.0, 0.0),
                 WithinAbs(sample_trilinear(g, f, x_inside, 0.0, 0.0), 1e-12));

    // x = -small → should sample near x = L - small.
    const double x_neg      = -0.25 * dx;
    const double x_neg_wrap = 1.0 - 0.25 * dx;
    REQUIRE_THAT(sample_trilinear(g, f, x_neg, 0.0, 0.0),
                 WithinAbs(sample_trilinear(g, f, x_neg_wrap, 0.0, 0.0), 1e-12));
}

TEST_CASE("advect_particles_rk2 is a no-op in a zero-velocity field",
          "[particles][advection]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);
    VelocityField vf;
    vf.vx.assign(g.size(), 0.0);
    vf.vy.assign(g.size(), 0.0);
    vf.vz.assign(g.size(), 0.0);

    ParticleSystem ps;
    ps.x = { 0.1, 0.5, 0.9 };
    ps.y = { 0.2, 0.5, 0.8 };
    ps.z = { 0.3, 0.5, 0.7 };

    advect_particles_rk2(ps, g, vf, /*dt=*/0.1);

    REQUIRE_THAT(ps.x[0], WithinAbs(0.1, 1e-15));
    REQUIRE_THAT(ps.x[1], WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(ps.x[2], WithinAbs(0.9, 1e-15));
    REQUIRE_THAT(ps.y[1], WithinAbs(0.5, 1e-15));
    REQUIRE_THAT(ps.z[1], WithinAbs(0.5, 1e-15));
}

TEST_CASE("advect_particles_rk2 drifts particles by v·dt in a uniform flow",
          "[particles][advection][plane-wave]")
{
    // The arg-form recovery (Eq. 4) returns v = ℏk exactly for a
    // discrete plane wave. We test against the actual recovered
    // velocity to keep this an integration test of "particles drift
    // at the recovered v".
    const int    N    = 8;
    const double L    = 1.0;
    const double hbar = 0.1;
    Grid3D g(N, N, N, L, L, L);

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double kx = two_pi * 1 / L;
    init_plane_wave(g, kx, 0.0, 0.0);
    const VelocityField vf = recover_velocity(g, hbar);

    ParticleSystem ps;
    ps.x = { 0.25, 0.5, 0.75 };
    ps.y = { 0.5,  0.5, 0.5  };
    ps.z = { 0.5,  0.5, 0.5  };
    const auto x0 = ps.x;

    const double dt       = 0.01;
    const double expected = hbar * kx * dt;

    advect_particles_rk2(ps, g, vf, dt);

    for (std::size_t i = 0; i < ps.x.size(); ++i) {
        REQUIRE_THAT(ps.x[i] - x0[i], WithinAbs(expected, 1e-12));
        REQUIRE_THAT(ps.y[i],         WithinAbs(0.5,      1e-12));
        REQUIRE_THAT(ps.z[i],         WithinAbs(0.5,      1e-12));
    }
}

TEST_CASE("advect_particles_rk2 keeps particles inside the periodic box",
          "[particles][advection][periodic]")
{
    const int    N    = 8;
    const double L    = 1.0;
    const double hbar = 0.1;
    Grid3D g(N, N, N, L, L, L);

    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    init_plane_wave(g, two_pi * 1 / L, 0.0, 0.0);
    const VelocityField vf = recover_velocity(g, hbar);

    ParticleSystem ps;
    ps.x = { 0.95 };          // near the upper boundary
    ps.y = { 0.5  };
    ps.z = { 0.5  };

    // Big enough dt that the single particle definitely crosses x = L.
    advect_particles_rk2(ps, g, vf, /*dt=*/1.0);

    REQUIRE(ps.x[0] >= 0.0);
    REQUIRE(ps.x[0] <  L);
}
