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
#include "viz/gpu_solver.hpp"
#include "viz/particle_renderer.hpp"
#include "viz/particles.hpp"
#include "viz/volume_renderer.hpp"
#include "viz/window.hpp"

#include <SDL3/SDL.h>
#include <imgui.h>

#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>
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

constexpr int    kInitialGridN     = 128;    ///< startup lattice size (power of 2)
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

void seed_particles_uniform(viz::ParticleSystem& ps,
                            std::mt19937& rng,
                            int n_particles)
{
    resize_particles(ps, static_cast<std::size_t>(n_particles));
    std::uniform_real_distribution<double> dist(0.0, kDomainSize);
    for (int i = 0; i < n_particles; ++i) {
        ps.x[i] = dist(rng);
        ps.y[i] = dist(rng);
        ps.z[i] = dist(rng);
        ps.r[i] = 0.40f;
        ps.g[i] = 0.80f;
        ps.b[i] = 1.00f;
    }
}

void seed_particles_from_psi(viz::ParticleSystem& ps,
                             const isf::Grid3D& g,
                             std::mt19937& rng,
                             int n_particles,
                             viz::DensitySampleMode mode)
{
    resize_particles(ps, static_cast<std::size_t>(n_particles));
    const std::size_t total_cells = g.size();
    std::vector<double> weights(total_cells);

    for (std::size_t i = 0; i < total_cells; ++i) {
        const auto c1 = g.psi1(i);
        const double rho1 = std::norm(c1);
        if (mode == viz::DensitySampleMode::Psi1Density) {
            weights[i] = std::clamp(rho1, 0.0, 1.0);
        } else if (mode == viz::DensitySampleMode::Psi2Density) {
            const auto c2 = g.psi2(i);
            weights[i] = std::clamp(std::norm(c2), 0.0, 1.0);
        } else {
            // VortexSmoke: 1 - |psi1|^2 (vortex core / smoke ring)
            weights[i] = std::max(0.0, 1.0 - rho1);
        }
    }

    double sum = 0.0;
    for (double w : weights) sum += w;
    if (sum <= 1e-9) {
        seed_particles_uniform(ps, rng, n_particles);
        return;
    }

    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    std::uniform_real_distribution<double> jitter(-0.5, 0.5);

    const int nx = g.nx();
    const int ny = g.ny();
    const double dx = g.dx();
    const double dy = g.dy();
    const double dz = g.dz();

    for (int p = 0; p < n_particles; ++p) {
        const std::size_t idx = dist(rng);
        const int ix = static_cast<int>(idx % nx);
        const int iy = static_cast<int>((idx / nx) % ny);
        const int iz = static_cast<int>(idx / (nx * ny));

        double px = (ix + 0.5 + jitter(rng)) * dx;
        double py = (iy + 0.5 + jitter(rng)) * dy;
        double pz = (iz + 0.5 + jitter(rng)) * dz;

        ps.x[p] = std::clamp(px, 0.0, g.lx());
        ps.y[p] = std::clamp(py, 0.0, g.ly());
        ps.z[p] = std::clamp(pz, 0.0, g.lz());

        ps.r[p] = 0.15f;
        ps.g[p] = 0.85f;
        ps.b[p] = 1.00f;
    }
}

} // namespace

enum class RenderMode : int {
    VolumeRaymarch = 0, ///< Volumetric raymarching of Eulerian fields (default)
    TracerParticles = 1, ///< Passive Lagrangian tracer particles
};

enum class SeedPattern : int {
    SceneGeometry  = 0,
    UniformBox     = 1,
    BornPsiDensity = 2
};

int main(int argc, char** argv)
try {
    bool test_render_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--render-test") {
            test_render_mode = true;
        }
    }

    viz::Window           win(1280, 720, "Schroedinger's Smoke (WebGPU)");
    viz::VolumeRenderer   vol_renderer;
    viz::ParticleRenderer renderer;
    viz::GifRecorder      recorder;
    viz::OrbitCamera      cam(
        glm::vec3{ 0.5f * float(kDomainSize) },
        1.55f * float(kDomainSize));
    // Orbit camera to view head-on collision crown
    cam.orbit(0.45f, 0.22f);

    int    grid_n      = kInitialGridN;
    int    n_particles = kInitialParticles;
    isf::Grid3D g(grid_n, grid_n, grid_n,
                  kDomainSize, kDomainSize, kDomainSize);

    viz::GpuFluidSolver gpu_solver(grid_n, kDomainSize);
    bool use_gpu = true;

    // Raytraced voxels (volumetric) is the default rendering mode
    RenderMode render_mode = RenderMode::VolumeRaymarch;
    viz::VolumeRenderParams vol_params;
    bool render_requested = false;
    std::string last_render_path;

    viz::ParticleAdvectParams part_params;
    SeedPattern seed_pattern = SeedPattern::SceneGeometry;

    viz::ParticleSystem ps;
    std::mt19937 rng(0xC0FFEEu);

    Scene  scene = Scene::CollidingBalls;
    double hbar  = kHbarDefault;
    double dt    = 0.020;
    float  point_size = 4.0f;
    bool   pause_sim  = false;
    int    sim_frame  = 0;

    struct SequenceRenderJob {
        bool active = false;
        int  current_frame = 0;
        int  start_frame = 0;
        int  end_frame = 100;
        int  render_grid_n = 512;
        int  render_w = 1920;
        int  render_h = 1080;
        int  orig_grid_n = 128;
        std::string out_dir;
    };
    SequenceRenderJob seq_job;
    int  seq_range[2]    = { 0, 100 };
    int  seq_grid_choice = 2; // 0: 128, 1: 256, 2: 512 (default to 512)
    int  seq_res_choice  = 1; // 0: Native, 1: 1080p, 2: 720p, 3: 4K (default to 1080p)
    bool pause_at_end    = false;

    auto get_next_seq_dir = []() -> std::string {
        std::error_code ec;
        std::filesystem::create_directories("renders", ec);
        int idx = 1;
        while (true) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "seq_%04d", idx);
            auto p = std::filesystem::path("renders") / buf;
            if (!std::filesystem::exists(p, ec)) {
                std::filesystem::create_directories(p, ec);
                return p.string();
            }
            idx++;
        }
    };

    auto seed_active_particles = [&]() {
        if (seed_pattern == SeedPattern::UniformBox) {
            seed_particles_uniform(ps, rng, n_particles);
        } else if (seed_pattern == SeedPattern::BornPsiDensity) {
            seed_particles_from_psi(ps, g, rng, n_particles, part_params.density_sample_mode);
        } else {
            switch (scene) {
            case Scene::VortexRing:     seed_particles_ring(ps, rng, n_particles);      break;
            case Scene::CollidingBalls: seed_particles_two_balls(ps, rng, n_particles); break;
            }
        }
        renderer.upload(ps);
    };

    auto reset_scene = [&]() {
        sim_frame = 0;
        switch (scene) {
        case Scene::VortexRing:     reset_vortex_ring(g);     break;
        case Scene::CollidingBalls: reset_two_balls(g, hbar); break;
        }
        seed_active_particles();
        gpu_solver.upload_grid(g);
        if (use_gpu) {
            // Initial micro-step so velocity buffer is immediately ready for raymarching
            gpu_solver.step(0.0001, hbar);
        }
    };
    reset_scene();

    auto rebuild_at_resolution = [&](int new_n) {
        grid_n = new_n;
        g = isf::Grid3D(grid_n, grid_n, grid_n,
                        kDomainSize, kDomainSize, kDomainSize);
        gpu_solver.resize(grid_n);
        reset_scene();
    };

    isf::PressureWorkspace press_ws;
    isf::VelocityField     vf;

    uint64_t perf_freq  = SDL_GetPerformanceFrequency();
    uint64_t perf_prev  = SDL_GetPerformanceCounter();
    double   t_accum    = 0.0;
    int      frames_accum = 0;
    float    fps_display  = 0.0f;
    if (test_render_mode) {
        for (int i = 0; i < 5; ++i) {
            gpu_solver.step(dt, hbar);
        }
        int fbw = 0, fbh = 0;
        win.framebuffer_size(fbw, fbh);
        const float aspect = fbh > 0 ? float(fbw) / float(fbh) : 1.0f;
        const glm::mat4 view_proj = cam.projection_matrix(aspect) * cam.view_matrix();
        const glm::mat4 inv_view_proj = glm::inverse(view_proj);

        std::string saved = vol_renderer.render_to_file(
            gpu_solver.velocity_buffer(),
            gpu_solver.psi1_buffer(),
            gpu_solver.psi2_buffer(),
            grid_n,
            static_cast<float>(kDomainSize),
            inv_view_proj,
            cam.position(),
            vol_params,
            fbw,
            fbh,
            "renders"
        );
        std::printf("Render test produced: %s\n", saved.c_str());
        return saved.empty() ? EXIT_FAILURE : EXIT_SUCCESS;
    }

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

        // --- physics & sequence render ---
        if (seq_job.active) {
            // Advance simulation step-by-step for the offline sequence render
            if (seq_job.current_frame > seq_job.start_frame) {
                if (use_gpu) {
                    gpu_solver.step(dt, hbar);
                } else {
                    isf::schrodinger_step(g, dt, hbar);
                    isf::normalize(g);
                    isf::pressure_project(g, press_ws);
                    isf::recover_velocity_from_workspace(g, hbar, press_ws, vf);
                    gpu_solver.upload_velocity(vf);
                }
            }
            sim_frame = seq_job.current_frame;

            char fname[64];
            std::snprintf(fname, sizeof(fname), "frame_%04d.bmp", seq_job.current_frame);

            const float render_aspect = float(seq_job.render_w) / float(seq_job.render_h);
            const glm::mat4 render_vp = cam.projection_matrix(render_aspect) * cam.view_matrix();
            const glm::mat4 render_inv_vp = glm::inverse(render_vp);

            vol_renderer.render_to_file(
                gpu_solver.velocity_buffer(),
                gpu_solver.psi1_buffer(),
                gpu_solver.psi2_buffer(),
                grid_n,
                static_cast<float>(kDomainSize),
                render_inv_vp,
                cam.position(),
                vol_params,
                seq_job.render_w,
                seq_job.render_h,
                seq_job.out_dir,
                fname
            );

            seq_job.current_frame++;
            if (seq_job.current_frame > seq_job.end_frame) {
                seq_job.active = false;
                if (seq_job.render_grid_n != seq_job.orig_grid_n) {
                    rebuild_at_resolution(seq_job.orig_grid_n);
                }
                last_render_path = std::format("{}/ ({} frames)", seq_job.out_dir, seq_job.end_frame - seq_job.start_frame + 1);
            }
        } else if (!pause_sim) {
            if (pause_at_end && sim_frame >= seq_range[1]) {
                pause_sim = true;
            } else {
                sim_frame++;
                if (use_gpu) {
                    gpu_solver.step(dt, hbar);
                    if (render_mode == RenderMode::TracerParticles) {
                        part_params.dt = static_cast<float>(dt);
                        part_params.frame_seed += 1013904223u;
                        gpu_solver.advect_particles(renderer.particle_buffer(),
                                                    renderer.particle_count(),
                                                    part_params);
                    }
                } else {
                    isf::schrodinger_step(g, dt, hbar);
                    isf::normalize(g);
                    isf::pressure_project(g, press_ws);
                    isf::recover_velocity_from_workspace(g, hbar, press_ws, vf);
                    gpu_solver.upload_velocity(vf);
                    if (render_mode == RenderMode::TracerParticles) {
                        viz::advect_particles_rk2(ps, g, vf, dt);
                        renderer.upload(ps);
                    }
                }
            }
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
        ImGui::Text("Grid: %d^3  particles: %zu", grid_n, static_cast<size_t>(n_particles));

        ImGui::Checkbox("GPU Compute Solver (WGSL)", &use_gpu);

        static const char* render_mode_labels[] = {
            "Raytraced Voxels (Volumetric)",
            "Tracer Particles"
        };
        int rm_idx = static_cast<int>(render_mode);
        if (ImGui::Combo("render mode", &rm_idx, render_mode_labels, IM_ARRAYSIZE(render_mode_labels))) {
            render_mode = static_cast<RenderMode>(rm_idx);
        }

        if (render_mode == RenderMode::VolumeRaymarch) {
            static const char* field_labels[] = {
                "Vorticity |curl v|",
                "Speed |v|",
                "Enstrophy |curl v|^2",
                "Psi_1 Phase arg(psi1)",
                "Psi_2 Phase arg(psi2)",
                "Psi_1 Density |psi1|^2",
                "Spinor Relative Phase"
            };
            int f_idx = static_cast<int>(vol_params.field_mode);
            if (ImGui::Combo("field", &f_idx, field_labels, IM_ARRAYSIZE(field_labels))) {
                vol_params.field_mode = static_cast<viz::VolumeFieldMode>(f_idx);
                // Auto-suggest spectral rainbow palette when selecting phase modes
                if ((f_idx == 3 || f_idx == 4 || f_idx == 6) && vol_params.palette == viz::VolumePalette::NeonCyan) {
                    vol_params.palette = viz::VolumePalette::PhaseRainbow;
                }
            }

            static const char* pal_labels[] = {
                "Neon Cyan (Quantum)",
                "Fire Plasma",
                "Deep Ocean",
                "Wispy Smoke",
                "Spectral Rainbow (Phase)",
                "SIGGRAPH 2016 Collision (Red/Blue)"
            };
            int pal_idx = static_cast<int>(vol_params.palette);
            if (ImGui::Combo("palette", &pal_idx, pal_labels, IM_ARRAYSIZE(pal_labels))) {
                vol_params.palette = static_cast<viz::VolumePalette>(pal_idx);
            }

            ImGui::SliderFloat("density", &vol_params.density_scale, 1.0f, 60.0f, "%.1f");
            ImGui::SliderFloat("threshold", &vol_params.threshold, 0.001f, 0.5f, "%.3f");

            static const char* prev_res_labels[] = {
                "Native (Full Window)",
                "720p (1280x720)",
                "480p (854x480) [Fast RT]",
                "360p (640x360) [Ultra Fast]",
                "240p (426x240) [Max FPS]",
                "180p (320x180) [Retro Low-Res]",
                "144p (256x144) [Pixel Art]",
                "72p  (128x72)  [Ultra Chunky]",
                "Custom Scale"
            };
            int pr_idx = static_cast<int>(vol_params.preview_res);
            if (ImGui::Combo("preview resolution", &pr_idx, prev_res_labels, IM_ARRAYSIZE(prev_res_labels))) {
                vol_params.preview_res = static_cast<viz::PreviewResolution>(pr_idx);
            }

            if (vol_params.preview_res == viz::PreviewResolution::Custom) {
                ImGui::SliderFloat("res scale", &vol_params.preview_scale, 0.05f, 1.0f, "%.2fx");
            }

            ImGui::Checkbox("bilinear filter", &vol_params.filter_linear);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Checked: Smooth bilinear hardware upscaling.\n"
                                  "Unchecked: Chunky nearest-neighbor pixels (makes low-res obvious!).");
            }

            int cur_fbw = 0, cur_fbh = 0;
            win.framebuffer_size(cur_fbw, cur_fbh);
            auto [cur_w, cur_h] = viz::get_preview_dimensions(cur_fbw, cur_fbh, vol_params);
            float ray_ratio = (cur_fbw > 0 && cur_fbh > 0) ? (float(cur_w * cur_h) / float(cur_fbw * cur_fbh)) : 1.0f;
            if (ray_ratio < 0.99f) {
                ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f),
                                   "RT Res: %dx%d (%.0f%% rays, ~%.1fx faster)",
                                   cur_w, cur_h, ray_ratio * 100.0f, 1.0f / ray_ratio);
            } else {
                ImGui::TextDisabled("RT Res: %dx%d (Native 100%%)", cur_fbw, cur_fbh);
            }

            ImGui::Checkbox("preview lighting", &vol_params.preview_lighting);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Unchecked: Fast unlit preview for smooth interaction.\n"
                                  "Checked: Real-time volumetric shadows in viewport.");
            }
            if (vol_params.preview_lighting) {
                ImGui::SliderFloat("preview shadow", &vol_params.shadow_strength, 0.0f, 1.5f, "%.2f");
            } else {
                ImGui::TextDisabled("Preview mode: Unlit (Click Render for full lighting)");
            }

            if (ImGui::TreeNode("Key Light & Shadow Config")) {
                ImGui::SliderFloat("light X", &vol_params.light_x, -1.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("light Y", &vol_params.light_y, 0.1f, 1.0f, "%.2f");
                ImGui::SliderFloat("light Z", &vol_params.light_z, -1.0f, 1.0f, "%.2f");
                ImGui::SliderFloat("shadow strength", &vol_params.shadow_strength, 0.0f, 1.5f, "%.2f");
                ImGui::TreePop();
            }

            if (ImGui::Button("Render Frame (With Lighting) -> ./renders", ImVec2(-1, 28))) {
                render_requested = true;
            }
            if (!last_render_path.empty()) {
                ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.0f), "Rendered: %s", last_render_path.c_str());
            }
        }

        static const int  res_options[] = { 32, 64, 128, 256, 512 };
        static const char* res_labels[]  = { "32^3", "64^3", "128^3", "256^3", "512^3 (High Detail)" };
        int res_idx = 2; // default to 128
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
        if (render_mode == RenderMode::TracerParticles) {
            ImGui::SliderFloat("point size", &point_size, 1.0f, 16.0f, "%.1f");
            if (ImGui::SliderInt("particles", &n_particles, 10000, 600000)) {
                // will take effect on next reset particles
            }

            ImGui::SeparatorText("Psi Particle Motion & Visuals");

            static const char* color_mode_labels[] = {
                "Default (Scene Ink)",
                "Vortex Core Glow (1 - |psi1|^2)",
                "Psi_1 Phase arg(psi1)",
                "Spinor Relative Phase",
                "Velocity Speed |v|"
            };
            int cm_idx = static_cast<int>(part_params.color_mode);
            if (ImGui::Combo("particle color", &cm_idx, color_mode_labels, IM_ARRAYSIZE(color_mode_labels))) {
                part_params.color_mode = static_cast<viz::ParticleColorMode>(cm_idx);
            }

            ImGui::SliderFloat("psi drift", &part_params.psi_drift, -3.0f, 3.0f, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("< 0: Attract into vortex filaments (|psi1|^2 -> 0)\n"
                                  "  0: Pure fluid tracer flow\n"
                                  "> 0: Attract into fluid bulk (|psi1|^2 -> 1)");
            }

            if (ImGui::SmallButton("Trap in Core (-1.0)")) {
                part_params.psi_drift = -1.0f;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Pure Flow (0.0)")) {
                part_params.psi_drift = 0.0f;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Repel (+1.0)")) {
                part_params.psi_drift = 1.0f;
            }

            ImGui::Checkbox("fade non-core particles", &part_params.fade_non_core);

            ImGui::SeparatorText("Born Rule Density Sampling");

            static const char* sample_mode_labels[] = {
                "Off (Standard Passive Advection)",
                "Vortex Smoke (1 - |psi1|^2) [Born Relaxation]",
                "Psi_1 Density (|psi1|^2)",
                "Psi_2 Density (|psi2|^2)"
            };
            int sm_idx = static_cast<int>(part_params.density_sample_mode);
            if (ImGui::Combo("density follow", &sm_idx, sample_mode_labels, IM_ARRAYSIZE(sample_mode_labels))) {
                part_params.density_sample_mode = static_cast<viz::DensitySampleMode>(sm_idx);
            }

            ImGui::SliderFloat("relaxation rate", &part_params.resample_rate, 0.0f, 0.15f, "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Continuous renewal rate: particles outside the density volume\n"
                                  "continuously respawn into the 3D psi density so the particle cloud\n"
                                  "permanently forms the 3D shape of psi.");
            }

            if (ImGui::Button("Reseed to |psi|^2 Density")) {
                seed_particles_from_psi(ps, g, rng, n_particles, part_params.density_sample_mode);
                renderer.upload(ps);
            }

            static const char* seed_labels[] = {
                "Scene Geometry (Toroid / Spheres)",
                "Uniform Box Fill (Watch Ring Condense)",
                "Born |psi|^2 Density (Quantum Cloud)"
            };
            int s_idx = static_cast<int>(seed_pattern);
            if (ImGui::Combo("seed pattern", &s_idx, seed_labels, IM_ARRAYSIZE(seed_labels))) {
                seed_pattern = static_cast<SeedPattern>(s_idx);
                seed_active_particles();
            }
        }

        ImGui::Separator();
        ImGui::SeparatorText("Timeline & Playback");
        ImGui::Text("Frame: %d  (Target Range: %d to %d)", sim_frame, seq_range[0], seq_range[1]);
        if (pause_sim) {
            if (ImGui::Button("Play", ImVec2(65, 0))) pause_sim = false;
        } else {
            if (ImGui::Button("Pause", ImVec2(65, 0))) pause_sim = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step +1", ImVec2(65, 0))) {
            sim_frame++;
            if (use_gpu) gpu_solver.step(dt, hbar);
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset to 0", ImVec2(80, 0))) {
            reset_scene();
        }
        ImGui::Checkbox("Pause at end frame", &pause_at_end);

        if (ImGui::Button("reset psi")) {
            switch (scene) {
            case Scene::VortexRing:    reset_vortex_ring(g);     break;
            case Scene::CollidingBalls:reset_two_balls(g, hbar); break;
            }
            gpu_solver.upload_grid(g);
        }
        ImGui::SameLine();
        if (ImGui::Button("reset particles")) {
            seed_active_particles();
        }
        ImGui::SameLine();
        if (ImGui::Button("reset all")) {
            reset_scene();
        }

        ImGui::Separator();
        ImGui::SeparatorText("Rendering & Sequence Export");
        if (ImGui::Button("Render Current Frame -> ./renders", ImVec2(-1, 28))) {
            render_requested = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Renders the current 3D quantum smoke field with volumetric self-shadowing\n"
                              "and gradient normal illumination, saving to ./renders/render_XXXX.bmp.");
        }

        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f), "Sequence Export (Frames 0 to 100):");
        ImGui::DragIntRange2("frame range", &seq_range[0], &seq_range[1], 1, 0, 2000, "Start: %d", "End: %d");

        static const int   seq_grid_opts[]   = { 128, 256, 512 };
        static const char* seq_grid_labels[] = { "128^3", "256^3", "512^3 (High Detail Cinema)" };
        ImGui::Combo("sequence grid res", &seq_grid_choice, seq_grid_labels, IM_ARRAYSIZE(seq_grid_labels));

        static const char* seq_res_labels[] = {
            "Native Window Size",
            "1080p Full HD (1920x1080)",
            "720p HD (1280x720)",
            "4K UHD (3840x2160)"
        };
        ImGui::Combo("sequence image res", &seq_res_choice, seq_res_labels, IM_ARRAYSIZE(seq_res_labels));

        char btn_label[128];
        std::snprintf(btn_label, sizeof(btn_label), "Render Sequence (%d to %d at %s) -> ./renders",
                      seq_range[0], seq_range[1], seq_grid_labels[seq_grid_choice]);

        if (ImGui::Button(btn_label, ImVec2(-1, 32))) {
            seq_job.orig_grid_n   = grid_n;
            seq_job.render_grid_n = seq_grid_opts[seq_grid_choice];
            seq_job.start_frame   = seq_range[0];
            seq_job.end_frame     = seq_range[1];
            seq_job.current_frame = seq_range[0];
            seq_job.out_dir       = get_next_seq_dir();

            int fbw_curr = 0, fbh_curr = 0;
            win.framebuffer_size(fbw_curr, fbh_curr);
            switch (seq_res_choice) {
            case 0: // Native
                seq_job.render_w = fbw_curr > 0 ? fbw_curr : 1280;
                seq_job.render_h = fbh_curr > 0 ? fbh_curr : 720;
                break;
            case 1: // 1080p
                seq_job.render_w = 1920;
                seq_job.render_h = 1080;
                break;
            case 2: // 720p
                seq_job.render_w = 1280;
                seq_job.render_h = 720;
                break;
            case 3: // 4K
                seq_job.render_w = 3840;
                seq_job.render_h = 2160;
                break;
            }

            if (seq_job.render_grid_n != grid_n) {
                rebuild_at_resolution(seq_job.render_grid_n);
            }
            reset_scene();
            seq_job.active = true;
        }

        if (!last_render_path.empty()) {
            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.0f), "Saved: %s", last_render_path.c_str());
        }

        if (seq_job.active) {
            ImGui::OpenPopup("Rendering Sequence...");
        }
        if (ImGui::BeginPopupModal("Rendering Sequence...", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            const int total_frames = seq_job.end_frame - seq_job.start_frame + 1;
            const int done_frames  = seq_job.current_frame - seq_job.start_frame;
            const float progress   = total_frames > 0 ? float(done_frames) / float(total_frames) : 0.0f;

            ImGui::Text("Exporting sequence to: %s/", seq_job.out_dir.c_str());
            ImGui::Text("Render size: %dx%d | Simulation grid: %d^3", seq_job.render_w, seq_job.render_h, grid_n);
            ImGui::ProgressBar(progress, ImVec2(340, 26));
            ImGui::Text("Frame: %d / %d (%.1f%% complete)", seq_job.current_frame, seq_job.end_frame, progress * 100.0f);

            if (ImGui::Button("Cancel Sequence Render", ImVec2(340, 28))) {
                seq_job.active = false;
                if (seq_job.render_grid_n != seq_job.orig_grid_n) {
                    rebuild_at_resolution(seq_job.orig_grid_n);
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        recorder.imgui_panel();
        ImGui::End();

        // Draw call into WebGPU render pass
        int fbw = 0, fbh = 0;
        win.framebuffer_size(fbw, fbh);
        const float aspect = fbh > 0 ? float(fbw) / float(fbh) : 1.0f;
        const glm::mat4 view_proj = cam.projection_matrix(aspect) * cam.view_matrix();
        const glm::mat4 inv_view_proj = glm::inverse(view_proj);

        // Offline high-quality render with lighting to disk
        if (render_requested) {
            render_requested = false;
            last_render_path = vol_renderer.render_to_file(
                gpu_solver.velocity_buffer(),
                gpu_solver.psi1_buffer(),
                gpu_solver.psi2_buffer(),
                grid_n,
                static_cast<float>(kDomainSize),
                inv_view_proj,
                cam.position(),
                vol_params,
                fbw,
                fbh,
                "renders"
            );
        }

        wgfx::RenderPass* scene_pass = win.render_pass();

        if (render_mode == RenderMode::VolumeRaymarch) {
            vol_renderer.render(scene_pass,
                                gpu_solver.velocity_buffer(),
                                gpu_solver.psi1_buffer(),
                                gpu_solver.psi2_buffer(),
                                grid_n,
                                static_cast<float>(kDomainSize),
                                inv_view_proj,
                                cam.position(),
                                vol_params,
                                fbw,
                                fbh);
        } else {
            scene_pass->prepare();
            renderer.draw(scene_pass, view_proj, point_size,
                          static_cast<float>(fbw), static_cast<float>(fbh));
            scene_pass->end();
        }

        win.end_frame();
    }
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return EXIT_FAILURE;
}
