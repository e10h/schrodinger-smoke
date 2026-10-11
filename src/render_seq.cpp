/**
 * @file render_seq.cpp
 * @brief Standalone offline renderer for Schrödinger Smoke animations.
 *
 * Runs the quantum fluid simulation (using OpenMP on CPU or WebGPU) and
 * renders offline frames with full volumetric lighting, raymarching, and
 * self-shadowing directly to sequential BMP files without blocking any UI.
 *
 * Usage:
 *   schroedinger_render --start 0 --end 100 --grid 512 --width 1920 --height 1080 --out renders/seq_0001
 */

#define WGPU_IMPLEMENTATION
#include <wgfx.h>

#include "isf/grid.hpp"
#include "isf/init.hpp"
#include "isf/normalize.hpp"
#include "isf/pressure.hpp"
#include "isf/schrodinger.hpp"
#include "isf/velocity.hpp"
#include "viz/camera.hpp"
#include "viz/gpu_solver.hpp"
#include "viz/volume_renderer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

int parse_int(int argc, char** argv, const char* flag, int def_val)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return def_val;
}

float parse_float(int argc, char** argv, const char* flag, float def_val)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return static_cast<float>(std::atof(argv[i + 1]));
        }
    }
    return def_val;
}

std::string parse_str(int argc, char** argv, const char* flag, const std::string& def_val)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            return argv[i + 1];
        }
    }
    return def_val;
}

} // namespace

int main(int argc, char** argv)
try {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("Usage: schroedinger_render [options]\n\n"
                        "Options:\n"
                        "  --start N        Start frame (default: 0)\n"
                        "  --end N          End frame (default: 100)\n"
                        "  --grid N         Lattice size (e.g. 128, 256, 512, default: 512)\n"
                        "  --width W        Image width in pixels (default: 1920)\n"
                        "  --height H       Image height in pixels (default: 1080)\n"
                        "  --dt T           Simulation time step (default: 0.020)\n"
                        "  --hbar H         Planck constant (default: 0.020)\n"
                        "  --domain L       Physical domain size in meters (default: 1.0)\n"
                        "  --scene S        Scene type: 'balls' (default) or 'ring'\n"
                        "  --solver TYPE    'gpu' (default) or 'cpu'\n"
                        "  --out DIR        Output directory (default: renders/seq_0001)\n"
                        "  --yaw Y          Camera orbit yaw in radians (default: 0.45)\n"
                        "  --pitch P        Camera orbit pitch in radians (default: 0.22)\n"
                        "  --dist D         Camera distance multiplier (default: 1.55)\n"
                        "  --fps N          Frame rate for MP4 output (default: 30)\n"
                        "  --no-mp4         Skip MP4 generation\n");
            return 0;
        }
    }

    const int   start_frame = parse_int(argc, argv, "--start", 0);
    const int   end_frame   = parse_int(argc, argv, "--end", 100);
    const int   grid_n      = parse_int(argc, argv, "--grid", 512);
    const int   img_w       = parse_int(argc, argv, "--width", 1920);
    const int   img_h       = parse_int(argc, argv, "--height", 1080);
    const int   fps         = parse_int(argc, argv, "--fps", 30);
    const float dt          = parse_float(argc, argv, "--dt", 0.020f);
    const float hbar        = parse_float(argc, argv, "--hbar", 0.020f);
    const float domain_size = parse_float(argc, argv, "--domain", 1.0f);
    const std::string scene_str  = parse_str(argc, argv, "--scene", "balls");
    const std::string solver_str = parse_str(argc, argv, "--solver", "gpu");
    const std::string out_dir    = parse_str(argc, argv, "--out", "renders/seq_0001");
    const float cam_yaw   = parse_float(argc, argv, "--yaw", 0.45f);
    const float cam_pitch = parse_float(argc, argv, "--pitch", 0.22f);
    const float cam_dist  = parse_float(argc, argv, "--dist", 1.55f);

    bool make_mp4 = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-mp4") == 0) {
            make_mp4 = false;
        }
    }

    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);

#ifdef _OPENMP
    std::printf("[OpenMP] Active with %d threads\n", omp_get_max_threads());
#else
    std::printf("[OpenMP] Disabled\n");
#endif

    std::printf("[Renderer] Initializing headless WebGPU...\n");
    wgfx::init(); // Headless WebGPU device setup

    std::printf("[Renderer] Setting up 3D grid (%d^3 = %zu cells, domain=%.2fm)...\n",
                grid_n, static_cast<size_t>(grid_n) * grid_n * grid_n, domain_size);

    isf::Grid3D g(grid_n, grid_n, grid_n, domain_size, domain_size, domain_size);
    if (scene_str == "ring") {
        isf::init_vortex_ring_z(g,
                                0.5 * domain_size, 0.5 * domain_size, 0.5 * domain_size,
                                domain_size * 0.20, domain_size * 0.04);
    } else {
        const double cy = 0.5 * domain_size;
        const double cz = 0.5 * domain_size;
        isf::init_two_velocity_balls(g,
                                     domain_size * 0.30, cy, cz,
                                     domain_size * 0.70, cy, cz,
                                     domain_size * 0.15,
                                     +1.0, 0.0, 0.0,
                                     -1.0, 0.0, 0.0,
                                     hbar,
                                     /*iters=*/ 10);
    }

    // Camera setup
    viz::OrbitCamera cam(glm::vec3{ 0.5f * domain_size }, cam_dist * domain_size);
    cam.orbit(cam_yaw, cam_pitch);

    const float aspect = float(img_w) / float(img_h);
    const glm::mat4 view_proj = cam.projection_matrix(aspect) * cam.view_matrix();
    const glm::mat4 inv_view_proj = glm::inverse(view_proj);

    // Visual rendering parameters
    viz::VolumeRenderParams vol_params;
    vol_params.field_mode       = viz::VolumeFieldMode::VorticityMagnitude;
    vol_params.palette          = viz::VolumePalette::SiggraphCollision;
    vol_params.density_scale    = 60.0f;
    vol_params.threshold        = 0.189f;
    vol_params.shadow_strength  = 1.50f;
    vol_params.preview_lighting = true;
    vol_params.step_count       = 240.0f;

    viz::VolumeRenderer vol_renderer;
    viz::GpuFluidSolver gpu_solver(grid_n, domain_size);
    gpu_solver.upload_grid(g);
    gpu_solver.step(0.0001, hbar);

    isf::PressureWorkspace press_ws;
    isf::VelocityField vf;

    std::printf("[Renderer] Starting batch render: frames %d -> %d (%d frames) at %dx%d\n",
                start_frame, end_frame, end_frame - start_frame + 1, img_w, img_h);
    std::printf("[Renderer] Output folder: %s\n", out_dir.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame <= end_frame; ++frame) {
        if (frame > 0) {
            if (solver_str == "gpu") {
                gpu_solver.step(dt, hbar);
            } else {
                isf::schrodinger_step(g, dt, hbar);
                isf::normalize(g);
                isf::pressure_project(g, press_ws);
                isf::recover_velocity_from_workspace(g, hbar, press_ws, vf);
                gpu_solver.upload_velocity(vf);
            }
        }

        if (frame >= start_frame) {
            auto tf0 = std::chrono::high_resolution_clock::now();

            char fname[64];
            std::snprintf(fname, sizeof(fname), "frame_%04d.bmp", frame);

            std::string saved = vol_renderer.render_to_file(
                gpu_solver.velocity_buffer(),
                gpu_solver.psi1_buffer(),
                gpu_solver.psi2_buffer(),
                grid_n,
                domain_size,
                inv_view_proj,
                cam.position(),
                vol_params,
                img_w,
                img_h,
                out_dir,
                fname
            );

            auto tf1 = std::chrono::high_resolution_clock::now();
            double f_sec = std::chrono::duration<double>(tf1 - tf0).count();

            std::printf("  [Frame %4d/%4d] Saved: %s (%.2f s)\n",
                        frame, end_frame, fname, f_sec);
            std::fflush(stdout);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    std::printf("\n[Renderer] Successfully finished %d frames in %.2f s (avg %.2f s/frame)!\n",
                end_frame - start_frame + 1, total_sec,
                total_sec / std::max(1, end_frame - start_frame + 1));
    std::printf("[Renderer] Sequence location: %s\n", out_dir.c_str());

    // Encode MP4 video using ffmpeg
    if (make_mp4) {
        std::filesystem::path mp4_path = std::filesystem::path(out_dir) / "output.mp4";
        std::printf("\n[Renderer] Compiling frames into MP4 video (H.264 @ %d fps)...\n", fps);

        // Find ffmpeg binary
        std::string ffmpeg_bin = "ffmpeg";
        if (std::system("which ffmpeg > /dev/null 2>&1") != 0) {
            if (std::filesystem::exists("/opt/homebrew/bin/ffmpeg")) {
                ffmpeg_bin = "/opt/homebrew/bin/ffmpeg";
            } else if (std::filesystem::exists("/usr/local/bin/ffmpeg")) {
                ffmpeg_bin = "/usr/local/bin/ffmpeg";
            }
        }

        char cmd[1024];
        std::snprintf(cmd, sizeof(cmd),
            "%s -y -framerate %d -start_number %d -i \"%s/frame_%%04d.bmp\" "
            "-c:v libx264 -pix_fmt yuv420p -crf 18 \"%s\" > /dev/null 2>&1",
            ffmpeg_bin.c_str(),
            fps,
            start_frame,
            out_dir.c_str(),
            mp4_path.string().c_str());

        int ret = std::system(cmd);
        if (ret == 0 && std::filesystem::exists(mp4_path)) {
            std::error_code size_ec;
            double mb = double(std::filesystem::file_size(mp4_path, size_ec)) / (1024.0 * 1024.0);
            std::printf("[Renderer] MP4 video created successfully: %s (%.2f MB)\n",
                        mp4_path.string().c_str(), mb);
        } else {
            std::printf("[Renderer] Notice: ffmpeg encoding returned code %d (check if ffmpeg is in PATH).\n", ret);
        }
    }

    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "\n[Renderer] Fatal error: %s\n", e.what());
    return 1;
}
