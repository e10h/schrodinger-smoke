/**
 * @file main.cpp
 * @brief Entry point for the interactive Schrödinger's Smoke viewer (WebGPU + SDL3).
 *
 * Runs the ISF fluid simulation and particle advection on CPU, rendering
 * into a WebGPU frame using wgfx with an SDL3 window and Dear ImGui overlay.
 */

#include "isf/grid.hpp"
#include "isf/init.hpp"
#include "isf/normalize.hpp"
#include "isf/pressure.hpp"
#include "isf/schrodinger.hpp"
#include "isf/solver.hpp"
#include "isf/velocity.hpp"

#include "viz/camera.hpp"
#include "viz/gif_recorder.hpp"
#include "viz/particle_renderer.hpp"
#include "viz/particles.hpp"
#include "viz/window.hpp"

#include <SDL3/SDL.h>
#include <imgui.h>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {

inline void resize_particles(viz::ParticleSystem& ps, std::size_t n)
{
    ps.x.resize(n);
    ps.y.resize(n);
    ps.z.resize(n);
    ps.r.resize(n);
    ps.g.resize(n);
    ps.b.resize(n);
}

constexpr int    kInitialGridN     = 96;     ///< startup lattice size
constexpr int    kInitialParticles = 150000; ///< startup tracer count
constexpr double kDomainSize       = 1.0;    ///< physical domain (m)
constexpr double kRingRadius       = 0.20;   ///< vortex ring radius (m)
constexpr double kRingHalfThk      = 0.04;   ///< vortex ring slab half-thickness

constexpr double kBallCenter1    = 0.30;
constexpr double kBallCenter2    = 0.70;
constexpr double kBallRadius     = 0.15;
constexpr double kHbarDefault    = 0.02;
constexpr double kBallSpeed      = 1.0;   // m/s

enum class Scene {
    VortexRing,
    CollidingBalls,
};

void reset_vortex_ring(isf::Grid3D& g)
{
    isf::init_vortex_ring_z(
        g,
        /*cx=*/ 0.5 * kDomainSize,
        /*cy=*/ 0.5 * kDomainSize,
        /*cz=*/ 0.5 * kDomainSize,
        /*R =*/ kRingRadius,
        /*r =*/ kRingHalfThk);
}

void reset_two_balls(isf::Grid3D& g, double hbar)
{
    const double cy = 0.5 * kDomainSize;
    const double cz = 0.5 * kDomainSize;
    isf::init_two_velocity_balls(
        g,
        /*cx1=*/ kBallCenter1, cy, cz,
        /*cx2=*/ kBallCenter2, cy, cz,
        /*R  =*/ kBallRadius,
        /*v1 =*/ +kBallSpeed, 0.0, 0.0,
        /*v2 =*/ -kBallSpeed, 0.0, 0.0,
        hbar,
        /*iters=*/ 10);
}

void seed_particles_ring(viz::ParticleSystem& ps,
                         std::mt19937& rng,
                         int n_particles)
{
    resize_particles(ps, static_cast<std::size_t>(n_particles));

    std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * std::numbers::pi_v<double>);
    std::uniform_real_distribution<double> dist_psi(0.0, 2.0 * std::numbers::pi_v<double>);
    std::uniform_real_distribution<double> dist_u(0.0, 1.0);

    const double cx = 0.5 * kDomainSize;
    const double cy = 0.5 * kDomainSize;
    const double cz = 0.5 * kDomainSize;

    for (int i = 0; i < n_particles; ++i) {
        const double phi = dist_phi(rng);
        const double psi = dist_psi(rng);
        const double rho = kRingHalfThk * std::sqrt(dist_u(rng));

        const double ring_r = kRingRadius + rho * std::cos(psi);
        const double px     = cx + ring_r * std::cos(phi);
        const double py     = cy + ring_r * std::sin(phi);
        const double pz     = cz + rho * std::sin(psi);

        ps.x[i] = px;
        ps.y[i] = py;
        ps.z[i] = pz;

        const float shade = 0.70f + 0.30f * static_cast<float>(std::cos(phi));
        ps.r[i] = 0.55f * shade;
        ps.g[i] = 0.75f * shade;
        ps.b[i] = 1.00f * shade;
    }
}

void seed_particles_two_balls(viz::ParticleSystem& ps,
                              std::mt19937& rng,
                              int n_particles)
{
    resize_particles(ps, static_cast<std::size_t>(n_particles));

    std::uniform_real_distribution<double> dist_dir(-1.0, 1.0);
    std::uniform_real_distribution<double> dist_u(0.0, 1.0);

    const double cx1 = kBallCenter1;
    const double cx2 = kBallCenter2;
    const double cy  = 0.5 * kDomainSize;
    const double cz  = 0.5 * kDomainSize;

    auto seed_ball = [&](int start_idx, int end_idx,
                         double bx, double by, double bz,
                         float rr, float gg, float bb)
    {
        int p = start_idx;
        while (p < end_idx) {
            double rx = dist_dir(rng);
            double ry = dist_dir(rng);
            double rz = dist_dir(rng);
            double r2 = rx*rx + ry*ry + rz*rz;
            if (r2 > 1.0 || r2 == 0.0) continue;

            double radius = kBallRadius * std::cbrt(dist_u(rng));
            double inv_len = radius / std::sqrt(r2);

            ps.x[p] = bx + rx * inv_len;
            ps.y[p] = by + ry * inv_len;
            ps.z[p] = bz + rz * inv_len;

            ps.r[p] = rr;
            ps.g[p] = gg;
            ps.b[p] = bb;
            ++p;
        }
    };

    const int half = n_particles / 2;
    seed_ball(0,    half,        cx1, cy, cz, 1.00f, 0.25f, 0.20f);
    seed_ball(half, n_particles, cx2, cy, cz, 0.20f, 0.45f, 1.00f);
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
try {
    int    grid_n      = kInitialGridN;
    int    n_particles = kInitialParticles;
    isf::Grid3D g(grid_n, grid_n, grid_n,
                  kDomainSize, kDomainSize, kDomainSize);

    viz::ParticleSystem ps;
    std::mt19937 rng(0xC0FFEEu);

    Scene  scene = Scene::VortexRing;
    double hbar  = kHbarDefault;
    double dt    = 1.0 / 60.0;
    float  point_size = 4.0f;
    bool   pause_sim  = false;

    auto reset_scene = [&]() {
        switch (scene) {
        case Scene::VortexRing:
            reset_vortex_ring(g);
            seed_particles_ring(ps, rng, n_particles);
            break;
        case Scene::CollidingBalls:
            reset_two_balls(g, hbar);
            seed_particles_two_balls(ps, rng, n_particles);
            break;
        }
    };
    reset_scene();

    auto rebuild_at_resolution = [&](int new_n) {
        grid_n = new_n;
        g = isf::Grid3D(grid_n, grid_n, grid_n,
                        kDomainSize, kDomainSize, kDomainSize);
        reset_scene();
    };

    isf::PressureWorkspace press_ws;
    isf::VelocityField     vf;

    viz::Window  win(1280, 720, "Schroedinger's Smoke (WebGPU)");
    viz::ParticleRenderer renderer;
    viz::GifRecorder      recorder;
    viz::OrbitCamera      cam(
        glm::vec3{ 0.5f * float(kDomainSize) },
        1.6f * float(kDomainSize));

    uint64_t perf_freq  = SDL_GetPerformanceFrequency();
    uint64_t perf_prev  = SDL_GetPerformanceCounter();
    double   t_accum    = 0.0;
    int      frames_accum = 0;
    float    fps_display  = 0.0f;

    while (!win.should_close()) {
        win.poll_events();

        const auto& inp = win.input();
        if (inp.left_mouse_down && !inp.imgui_captures) {
            cam.orbit(-0.005f * inp.mouse_dx,
                      -0.005f * inp.mouse_dy);
        }
        if (inp.scroll_dy != 0.0f && !inp.imgui_captures) {
            cam.zoom(std::exp(-0.1f * inp.scroll_dy));
        }

        // --- physics ---
        if (!pause_sim) {
            isf::schrodinger_step(g, dt, hbar);
            isf::normalize(g);
            isf::pressure_project(g, press_ws);
            isf::recover_velocity_from_workspace(g, hbar, press_ws, vf);
            viz::advect_particles_rk2(ps, g, vf, dt);
        }

        // --- render ---
        win.begin_frame();

        // FPS counter
        uint64_t perf_now = SDL_GetPerformanceCounter();
        double frame_dt   = static_cast<double>(perf_now - perf_prev) / static_cast<double>(perf_freq);
        perf_prev         = perf_now;

        t_accum      += frame_dt;
        frames_accum += 1;
        if (t_accum >= 0.5) {
            fps_display  = static_cast<float>(frames_accum / t_accum);
            t_accum      = 0.0;
            frames_accum = 0;
        }

        // ImGui control panel
        ImGui::Begin("Schroedinger's Smoke");
        ImGui::Text("FPS: %.1f (WebGPU / SDL3)", fps_display);
        ImGui::Text("Grid: %d^3  particles: %zu", grid_n, ps.size());

        static const int  res_options[] = { 48, 64, 80, 96, 128, 160 };
        static const char* res_labels[]  = { "48^3", "64^3", "80^3", "96^3", "128^3", "160^3" };
        int res_idx = 0;
        for (int i = 0; i < IM_ARRAYSIZE(res_options); ++i) {
            if (res_options[i] == grid_n) { res_idx = i; break; }
        }
        if (ImGui::Combo("resolution", &res_idx, res_labels, IM_ARRAYSIZE(res_labels))) {
            rebuild_at_resolution(res_options[res_idx]);
        }

        const char* scene_labels[] = { "vortex ring", "colliding spheres" };
        int scene_idx = static_cast<int>(scene);
        if (ImGui::Combo("scene", &scene_idx, scene_labels, IM_ARRAYSIZE(scene_labels))) {
            scene = static_cast<Scene>(scene_idx);
            reset_scene();
        }

        ImGui::Separator();
        float h_f = static_cast<float>(hbar);
        if (ImGui::SliderFloat("hbar", &h_f, 0.005f, 0.10f, "%.3f")) {
            hbar = h_f;
        }
        float dt_f = static_cast<float>(dt);
        if (ImGui::SliderFloat("dt", &dt_f, 1.0f/240.0f, 1.0f/15.0f, "%.4f")) {
            dt = dt_f;
        }
        ImGui::SliderFloat("point size", &point_size, 1.0f, 12.0f, "%.1f");
        ImGui::SliderInt("particles", &n_particles, 10000, 600000);
        ImGui::Checkbox("pause", &pause_sim);
        if (ImGui::Button("reset ψ")) {
            switch (scene) {
            case Scene::VortexRing:    reset_vortex_ring(g);     break;
            case Scene::CollidingBalls:reset_two_balls(g, hbar); break;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("reset particles")) {
            switch (scene) {
            case Scene::VortexRing:    seed_particles_ring(ps, rng, n_particles);      break;
            case Scene::CollidingBalls:seed_particles_two_balls(ps, rng, n_particles); break;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("reset all")) {
            reset_scene();
        }
        ImGui::Separator();
        recorder.imgui_panel();
        ImGui::End();

        // Particle draw call into WebGPU render pass
        int fbw = 0, fbh = 0;
        win.framebuffer_size(fbw, fbh);
        const float aspect = fbh > 0 ? float(fbw) / float(fbh) : 1.0f;
        const glm::mat4 view_proj = cam.projection_matrix(aspect) * cam.view_matrix();

        renderer.upload(ps);

        wgfx::RenderPass* scene_pass = win.render_pass();
        scene_pass->prepare();
        renderer.draw(scene_pass, view_proj, point_size,
                      static_cast<float>(fbw), static_cast<float>(fbh));
        scene_pass->end();

        win.end_frame();
    }
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return EXIT_FAILURE;
}
