/**
 * @file gpu_solver.hpp
 * @brief WebGPU compute-shader ISF fluid solver and particle advector.
 *
 * Runs the full Incompressible Schrödinger Flow (Algorithm 1–3) entirely
 * on the GPU using WebGPU compute shaders, including:
 *   - 3D Stockham Radix-2 FFT in workgroup shared memory
 *   - Kinetic propagator phase rotation
 *   - Pointwise ψ normalization
 *   - Edge 1-form divergence extraction
 *   - Discrete Poisson solve in Fourier space
 *   - Pointwise phase correction and velocity field recovery
 *   - RK2 tracer particle advection
 */

#pragma once

#include <wgfx.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace isf {
class Grid3D;
struct VelocityField;
} // namespace isf

namespace viz {

struct ParticleSystem;

/**
 * @brief GPU-accelerated ISF fluid simulation and particle advection engine.
 */
class GpuFluidSolver {
public:
    /**
     * @brief Construct solver for grid resolution N^3 and physical domain size L.
     * @param grid_n Lattice resolution (must be power of 2: 32, 64, 128, 256).
     * @param domain_size Physical extent of periodic domain (e.g. 1.0 m).
     */
    GpuFluidSolver(int grid_n, double domain_size);
    ~GpuFluidSolver();

    GpuFluidSolver(const GpuFluidSolver&)            = delete;
    GpuFluidSolver& operator=(const GpuFluidSolver&) = delete;

    /// @brief Current grid resolution N along each axis.
    [[nodiscard]] int grid_n() const noexcept { return grid_n_; }

    /// @brief Reconfigure solver for a different grid resolution.
    void resize(int new_grid_n);

    /**
     * @brief Upload ψ state from CPU Grid3D into GPU storage buffers.
     */
    void upload_grid(const isf::Grid3D& g);

    /**
     * @brief Upload CPU velocity field into GPU velocity buffer.
     */
    void upload_velocity(const isf::VelocityField& vf);

    /**
     * @brief Advance the fluid simulation by one time step on GPU.
     * @param dt   Time step.
     * @param hbar Planck constant ℏ.
     */
    void step(double dt, double hbar);

    /**
     * @brief Advect particles on GPU using the computed velocity field.
     * @param particle_buffer WebGPU buffer containing ParticleData array.
     * @param particle_count  Number of active particles.
     * @param dt              Time step.
     */
    void advect_particles(wgpu::Buffer particle_buffer, std::size_t particle_count, double dt);

    /// @brief Storage buffer holding the 3D Eulerian velocity field (vec4<f32> * N^3).
    [[nodiscard]] wgpu::Buffer velocity_buffer() const noexcept { return b_vel_; }

    /// @brief Storage buffer holding ψ₁ spinor component (vec2<f32> * N^3).
    [[nodiscard]] wgpu::Buffer psi1_buffer() const noexcept { return b_psi1_; }

    /// @brief Storage buffer holding ψ₂ spinor component (vec2<f32> * N^3).
    [[nodiscard]] wgpu::Buffer psi2_buffer() const noexcept { return b_psi2_; }

private:
    void init_buffers();
    void init_pipelines();
    void destroy_resources();

    int    grid_n_      = 64;
    double domain_size_ = 1.0;

    // GPU Storage Buffers
    wgpu::Buffer b_psi1_;      // vec2<f32> * N^3
    wgpu::Buffer b_psi2_;      // vec2<f32> * N^3
    wgpu::Buffer b_scratch_;   // vec2<f32> * N^3
    wgpu::Buffer b_vel_;       // vec4<f32> * N^3
    wgpu::Buffer u_sim_param_; // Uniform buffer for simulation params
    wgpu::Buffer u_adv_param_; // Uniform buffer for particle advection params

    // Compute Pipelines
    wgpu::ComputePipeline pipe_fft_x_fwd_;
    wgpu::ComputePipeline pipe_fft_x_inv_;
    wgpu::ComputePipeline pipe_fft_y_fwd_;
    wgpu::ComputePipeline pipe_fft_y_inv_;
    wgpu::ComputePipeline pipe_fft_z_fwd_;
    wgpu::ComputePipeline pipe_fft_z_inv_;
    wgpu::ComputePipeline pipe_propagate_;
    wgpu::ComputePipeline pipe_normalize_;
    wgpu::ComputePipeline pipe_divergence_;
    wgpu::ComputePipeline pipe_poisson_;
    wgpu::ComputePipeline pipe_phase_vel_;
    wgpu::ComputePipeline pipe_advect_;

    // Bind Group Layouts
    wgpu::BindGroupLayout bgl_fft_;
    wgpu::BindGroupLayout bgl_propagate_;
    wgpu::BindGroupLayout bgl_normalize_;
    wgpu::BindGroupLayout bgl_divergence_;
    wgpu::BindGroupLayout bgl_poisson_;
    wgpu::BindGroupLayout bgl_phase_vel_;
    wgpu::BindGroupLayout bgl_advect_;

    // Bind Groups
    wgpu::BindGroup bg_fft_psi1_;
    wgpu::BindGroup bg_fft_psi2_;
    wgpu::BindGroup bg_fft_scratch_;
    wgpu::BindGroup bg_propagate_psi1_;
    wgpu::BindGroup bg_propagate_psi2_;
    wgpu::BindGroup bg_normalize_;
    wgpu::BindGroup bg_divergence_;
    wgpu::BindGroup bg_poisson_;
    wgpu::BindGroup bg_phase_vel_;

    void execute_fft_3d(wgpu::CommandEncoder& enc, wgpu::BindGroup bg, bool inverse);
};

} // namespace viz
