/**
 * @file bench_solver.cpp
 * @brief CPU-side benchmark for the ISF solver loop.
 *
 * Runs a fixed number of iterations of the basic per-frame work that
 * the viewer does:
 *
 *   1. solve_step           — Schrödinger → normalize → pressure project
 *   2. recover_velocity     — vertex-centred velocity from ψ
 *   3. advect_particles_rk2 — RK2 trilinear-sample advection
 *
 * and reports total wall-clock time, per-step time, an effective FPS,
 * and a per-phase breakdown so we can see which piece dominates the
 * frame budget. No OpenGL, no rendering, no ImGui — purely the work
 * we want to optimize.
 *
 * Usage:
 *   bench_solver [--iters N] [--grid N] [--particles N]
 *
 * Defaults match the viewer (64³, 150 000 particles, 100 iterations).
 */

#include "isf/grid.hpp"
#include "isf/init.hpp"
#include "isf/normalize.hpp"
#include "isf/pressure.hpp"
#include "isf/schrodinger.hpp"
#include "isf/solver.hpp"
#include "isf/velocity.hpp"
#include "viz/particles.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

namespace {

using hires_clock   = std::chrono::high_resolution_clock;
using ms_double = std::chrono::duration<double, std::milli>;

/// @brief Parse a long-form `--name N` integer argument from argv.
int parse_int_flag(int argc, char** argv, const char* name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

/// @brief Pretty-print a millisecond duration with a fixed width.
void print_phase(const char* label, double total_ms, int iters)
{
    std::printf("  %-22s  %10.3f ms total   %8.4f ms / iter\n",
                label, total_ms, total_ms / iters);
}

} // namespace

int main(int argc, char** argv)
{
    const int iters    = parse_int_flag(argc, argv, "--iters",     100);
    const int N        = parse_int_flag(argc, argv, "--grid",      64);
    const int n_part   = parse_int_flag(argc, argv, "--particles", 150000);

    const double L     = 1.0;
    const double hbar  = 0.03;
    const double dt    = 1.0 / 60.0;

    std::printf("=== ISF CPU benchmark ===\n");
    std::printf("  grid:      %d^3  (%d vertices)\n", N, N * N * N);
    std::printf("  particles: %d\n", n_part);
    std::printf("  iters:     %d\n", iters);
    std::printf("  hbar:      %.3f\n", hbar);
    std::printf("  dt:        %.5f\n", dt);
    std::printf("\n");

    // ----- setup --------------------------------------------------
    isf::Grid3D g(N, N, N, L, L, L);
    isf::init_vortex_ring_z(g, 0.5 * L, 0.5 * L, 0.5 * L,
                            /*R=*/0.20, /*r=*/0.04);

    viz::ParticleSystem ps;
    ps.x.resize(static_cast<std::size_t>(n_part));
    ps.y.resize(static_cast<std::size_t>(n_part));
    ps.z.resize(static_cast<std::size_t>(n_part));
    {
        std::mt19937 rng(0xC0FFEEu);
        std::uniform_real_distribution<double> uxy(0.20, 0.80);
        std::uniform_real_distribution<double> uz (0.35, 0.55);
        for (int i = 0; i < n_part; ++i) {
            ps.x[i] = uxy(rng);
            ps.y[i] = uxy(rng);
            ps.z[i] = uz(rng);
        }
    }

    // ----- pre-allocated scratch (no per-frame malloc) ------------
    isf::PressureWorkspace ws;
    isf::VelocityField     vf;

    // ----- timed loop ---------------------------------------------
    double t_schrod  = 0.0;
    double t_norm    = 0.0;
    double t_press   = 0.0;
    double t_recover = 0.0;
    double t_advect  = 0.0;

    const auto t0 = hires_clock::now();
    for (int i = 0; i < iters; ++i) {
        const auto a = hires_clock::now();
        isf::schrodinger_step(g, dt, hbar);
        const auto b = hires_clock::now();
        isf::normalize(g);
        const auto c = hires_clock::now();
        isf::pressure_project(g, ws);
        const auto d = hires_clock::now();
        isf::recover_velocity_from_workspace(g, hbar, ws, vf);
        const auto e = hires_clock::now();
        viz::advect_particles_rk2(ps, g, vf, dt);
        const auto f = hires_clock::now();

        t_schrod  += ms_double(b - a).count();
        t_norm    += ms_double(c - b).count();
        t_press   += ms_double(d - c).count();
        t_recover += ms_double(e - d).count();
        t_advect  += ms_double(f - e).count();
    }
    const auto t1 = hires_clock::now();

    const double total_ms      = ms_double(t1 - t0).count();
    const double per_iter_ms   = total_ms / iters;
    const double effective_fps = 1000.0 / per_iter_ms;

    // ----- report -------------------------------------------------
    std::printf("=== results ===\n");
    std::printf("  total:                %10.3f ms\n", total_ms);
    std::printf("  per iter:             %10.4f ms\n", per_iter_ms);
    std::printf("  effective FPS:        %10.2f\n", effective_fps);
    std::printf("\n");
    std::printf("  per-phase breakdown:\n");
    print_phase("schrodinger_step",      t_schrod,  iters);
    print_phase("normalize",             t_norm,    iters);
    print_phase("pressure_project",      t_press,   iters);
    print_phase("recover_velocity",      t_recover, iters);
    print_phase("advect_particles_rk2",  t_advect,  iters);

    return 0;
}
