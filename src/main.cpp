/**
 * @file main.cpp
 * @brief Entry point for the interactive Schrödinger's Smoke viewer.
 *
 * Brings together the Phase-1 CPU solver and the Phase-2 GL viewer:
 *
 *   1. Allocates a Grid3D and drops in a vortex ring via the slab
 *      construction (Sec. 3.1 of Chern et al. 2016).
 *   2. Seeds a cloud of passive tracer particles around the ring.
 *   3. Each frame: runs one `solve_step` (Alg. 1), recovers the
 *      velocity field, advects the particles by RK2, uploads their
 *      positions to the GPU and draws them as additive point
 *      sprites.
 *   4. An ImGui overlay exposes ℏ, dt, point size, a reset button,
 *      and an FPS readout.
 *
 * macOS only for now — uses OpenGL 4.1 core. The physics runs on the
 * CPU; Phase 3 will port the solver to compute shaders.
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

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

namespace {

// Initial / default values for the runtime-tunable knobs.
constexpr int    kInitialGridN     = 96;     ///< startup lattice size
constexpr int    kInitialParticles = 150000; ///< startup tracer count
constexpr double kDomainSize       = 1.0;    ///< physical domain (m)
constexpr double kRingRadius       = 0.20;   ///< vortex ring radius (m)
constexpr double kRingHalfThk      = 0.04;   ///< vortex ring slab half-thickness

// Colliding-spheres geometry.  R = 0.15 with centres at 0.30 / 0.70
// leaves a 0.10 m gap between ball surfaces for the collision and
// 0.15 m to the nearest periodic wall — enough clearance that the
// secondary vortex rings don't immediately re-enter on the torus.
// At 96³ the ball has ~14 cells of radius (well resolved); at 64³
// it is ~10 cells (marginal but usable).
constexpr double kBallCenter1    = 0.30;
constexpr double kBallCenter2    = 0.70;
constexpr double kBallRadius     = 0.15;

// Default ℏ — 0.02 matches the paper's Fig. 6 (Bunny / Teapot).
constexpr double kHbarDefault    = 0.02;

// Prescribed ball velocity for the colliding-spheres demo.
//
// The paper's Bunny/Teapot demo (Fig. 6) uses v = 1 m/s at ℏ = 0.02,
// giving k = v/ℏ = 50 /m and k·dx ≈ 1.56 on their 128³ grid.  The
// high k ensures the plane-wave phase wraps many times within the
// ball, so the vortex sheet at the ball boundary has enough curl to
// survive pressure projection as a persistent incompressible flow.
//
// At lower k (e.g. mode 1, k·dx ≈ 0.1) the prescribed velocity is
// almost entirely irrotational and pressure projection correctly
// removes it — giving zero flow.  Matching the paper's k·dx regime
// is what makes the impulsive collision work without per-frame drive.
constexpr double kBallSpeed      = 1.0;   // m/s, matching the paper

/** @brief Which scene the viewer is currently showing. */
enum class Scene {
    VortexRing,
    CollidingBalls,
};

/// @brief (Re)initialise ψ with a single vortex ring at the centre.
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

/// @brief (Re)initialise ψ with two velocity balls pointing toward
///        each other along the x-axis.
void reset_two_balls(isf::Grid3D& g, double hbar)
{
    const double cy = 0.5 * kDomainSize, cz = 0.5 * kDomainSize;
    const double cx1 = kBallCenter1 * kDomainSize;
    const double cx2 = kBallCenter2 * kDomainSize;
    isf::init_two_velocity_balls(
        g,
        cx1, cy, cz,
        cx2, cy, cz,
        kBallRadius,
        +kBallSpeed, 0.0, 0.0,        // ball 1 → +x
        -kBallSpeed, 0.0, 0.0,        // ball 2 → −x
        hbar);
}

/// @brief Tracer-particle layout for the vortex-ring scene: a flat
///        slab of dye that the moving ring sweeps through.
void seed_particles_ring(viz::ParticleSystem& ps, std::mt19937& rng,
                         int n_particles)
{
    ps.x.resize(n_particles);
    ps.y.resize(n_particles);
    ps.z.resize(n_particles);
    ps.r.assign(n_particles, 0.85f);
    ps.g.assign(n_particles, 0.90f);
    ps.b.assign(n_particles, 1.00f);

    std::uniform_real_distribution<double> uxy(0.20, 0.80);
    std::uniform_real_distribution<double> uz (0.35, 0.55);
    for (int i = 0; i < n_particles; ++i) {
        ps.x[i] = uxy(rng);
        ps.y[i] = uxy(rng);
        ps.z[i] = uz(rng);
    }
}

/// @brief Tracer-particle layout for the colliding-balls scene: half
///        the particles seeded inside ball 1 (red), half inside ball 2
///        (blue), so the user sees the two coloured inks meet and mix.
void seed_particles_two_balls(viz::ParticleSystem& ps, std::mt19937& rng,
                              int n_particles)
{
    ps.x.resize(n_particles);
    ps.y.resize(n_particles);
    ps.z.resize(n_particles);
    ps.r.resize(n_particles);
    ps.g.resize(n_particles);
    ps.b.resize(n_particles);

    const double cy = 0.5 * kDomainSize, cz = 0.5 * kDomainSize;
    const double cx1 = kBallCenter1 * kDomainSize;
    const double cx2 = kBallCenter2 * kDomainSize;
    const double R   = kBallRadius;

    auto seed_ball = [&](int begin, int end,
                          double cx, double cy_, double cz_,
                          float rr, float gg, float bb) {
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        int p = begin;
        while (p < end) {
            // Rejection-sample inside the ball.
            const double dx = u(rng), dy = u(rng), dz = u(rng);
            if (dx*dx + dy*dy + dz*dz > 1.0) continue;
            ps.x[p] = cx  + R * dx;
            ps.y[p] = cy_ + R * dy;
            ps.z[p] = cz_ + R * dz;
            ps.r[p] = rr;
            ps.g[p] = gg;
            ps.b[p] = bb;
            ++p;
        }
    };

    const int half = n_particles / 2;
    // Ball 1 → red. Ball 2 → blue.
    seed_ball(0,    half,        cx1, cy, cz, 1.00f, 0.25f, 0.20f);
    seed_ball(half, n_particles, cx2, cy, cz, 0.20f, 0.45f, 1.00f);
}

} // namespace

int main(int /*argc*/, char** /*argv*/)
try {
    // ----------------------- Solver state ------------------------
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

    // Helper that resets ψ and particles for the currently selected
    // scene (used at startup, on scene change, and on the reset
    // buttons).
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

    // Helper that rebuilds the grid at a new resolution and re-runs
    // the current scene's init. Called when the user picks a new
    // resolution from the ImGui combo.
    auto rebuild_at_resolution = [&](int new_n) {
        grid_n = new_n;
        g = isf::Grid3D(grid_n, grid_n, grid_n,
                        kDomainSize, kDomainSize, kDomainSize);
        reset_scene();
    };

    // Reusable scratch — pre-allocated once, reused every frame.
    isf::PressureWorkspace press_ws;
    isf::VelocityField     vf;

    // ----------------------- Viewer ------------------------------
    viz::Window  win(1280, 720, "Schroedinger's Smoke");
    viz::ParticleRenderer renderer;
    viz::GifRecorder      recorder;
    viz::OrbitCamera      cam(
        glm::vec3{ 0.5f * float(kDomainSize) },
        1.6f * float(kDomainSize));

    // Timing for FPS display.
    double t_prev = glfwGetTime();
    double t_accum = 0.0;
    int    frames_accum = 0;
    float  fps_display  = 0.0f;

    // ----------------------- Main loop ---------------------------
    while (!win.should_close()) {
        win.poll_events();

        // --- camera input ---
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
            // Inlined Algorithm 1 + workspace-aware overloads, so we
            // get zero per-frame allocations from the solver path.
            isf::schrodinger_step(g, dt, hbar);
            isf::normalize(g);
            isf::pressure_project(g, press_ws);
            isf::recover_velocity_from_workspace(g, hbar, press_ws, vf);
            viz::advect_particles_rk2(ps, g, vf, dt);
        }

        // --- render ---
        win.begin_frame();

        // FPS counter.
        const double t_now = glfwGetTime();
        t_accum      += t_now - t_prev;
        t_prev        = t_now;
        frames_accum += 1;
        if (t_accum >= 0.5) {
            fps_display  = static_cast<float>(frames_accum / t_accum);
            t_accum      = 0.0;
            frames_accum = 0;
        }

        // ImGui control panel.
        ImGui::Begin("Schroedinger's Smoke");
        ImGui::Text("FPS: %.1f", fps_display);
        ImGui::Text("Grid: %d^3  particles: %zu",
                    grid_n, ps.size());

        // Resolution selector. Switching reallocates the Grid3D and
        // re-runs the current scene's init at the new resolution —
        // this can take ~½ s at 128³ because of the constraint-
        // projection iterations during init.
        static const int  res_options[] = { 48, 64, 80, 96, 128, 160 };
        static const char* res_labels[]  = { "48^3", "64^3",
                                             "80^3", "96^3", "128^3",
                                             "160^3" };
        int res_idx = 0;
        for (int i = 0; i < IM_ARRAYSIZE(res_options); ++i) {
            if (res_options[i] == grid_n) { res_idx = i; break; }
        }
        if (ImGui::Combo("resolution", &res_idx,
                         res_labels, IM_ARRAYSIZE(res_labels))) {
            rebuild_at_resolution(res_options[res_idx]);
        }

        // Scene selector — changing the scene resets ψ and particles.
        const char* scene_labels[] = { "vortex ring",
                                        "colliding spheres" };
        int scene_idx = static_cast<int>(scene);
        if (ImGui::Combo("scene", &scene_idx,
                         scene_labels, IM_ARRAYSIZE(scene_labels))) {
            scene = static_cast<Scene>(scene_idx);
            reset_scene();
        }

        ImGui::Separator();
        float h_f = static_cast<float>(hbar);
        if (ImGui::SliderFloat("hbar", &h_f, 0.005f, 0.10f, "%.3f")) {
            hbar = h_f;
        }
        float dt_f = static_cast<float>(dt);
        if (ImGui::SliderFloat("dt",   &dt_f, 1.0f/240.0f, 1.0f/15.0f,
                               "%.4f")) {
            dt = dt_f;
        }
        ImGui::SliderFloat("point size", &point_size, 1.0f, 12.0f, "%.1f");
        // Particle count slider — applies on next reset.
        ImGui::SliderInt("particles", &n_particles, 10000, 600000);
        ImGui::Checkbox("pause",   &pause_sim);
        if (ImGui::Button("reset ψ")) {
            switch (scene) {
            case Scene::VortexRing:    reset_vortex_ring(g);   break;
            case Scene::CollidingBalls:reset_two_balls(g, hbar); break;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("reset particles")) {
            switch (scene) {
            case Scene::VortexRing:
                seed_particles_ring(ps, rng, n_particles);     break;
            case Scene::CollidingBalls:
                seed_particles_two_balls(ps, rng, n_particles); break;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("reset all")) {
            reset_scene();
        }
        ImGui::Separator();
        recorder.imgui_panel();
        ImGui::End();

        // Particle draw call.
        int fbw = 0, fbh = 0;
        win.framebuffer_size(fbw, fbh);
        const float aspect = fbh > 0 ? float(fbw) / float(fbh) : 1.0f;
        const glm::mat4 view_proj =
            cam.projection_matrix(aspect) * cam.view_matrix();
        renderer.upload(ps);
        renderer.draw(view_proj, point_size);

        // GIF capture — after scene draw, before ImGui overlay.
        if (recorder.is_recording()) {
            recorder.capture_frame();
        }

        win.end_frame();
    }
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return EXIT_FAILURE;
}
