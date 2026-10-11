/**
 * @file particle_renderer.hpp
 * @brief WebGPU (wgfx) renderer for the tracer-particle cloud.
 *
 * Draws a ParticleSystem as soft additively-blended point billboard sprites
 * via WebGPU WGSL compute/render pipelines.
 */

#pragma once

#include <wgfx.h>
#include <glm/mat4x4.hpp>
#include <vector>

namespace viz {

struct ParticleSystem;

/**
 * @brief WebGPU renderer for tracer particle cloud using billboard quads in WGSL.
 */
class ParticleRenderer {
public:
    ParticleRenderer();
    ~ParticleRenderer();

    ParticleRenderer(const ParticleRenderer&)            = delete;
    ParticleRenderer& operator=(const ParticleRenderer&) = delete;

    /**
     * @brief Upload particle positions and colors to WebGPU storage buffer.
     */
    void upload(const ParticleSystem& ps);

    /**
     * @brief WebGPU storage buffer handle for GPU compute advection.
     */
    [[nodiscard]] wgpu::Buffer particle_buffer() const noexcept {
        return s_particles_ ? s_particles_->buffer : nullptr;
    }

    [[nodiscard]] std::size_t particle_count() const noexcept {
        return particle_count_;
    }

    void set_particle_count(std::size_t count) noexcept;

    /**
     * @brief Draw the particle cloud into the active render pass.
     *
     * @param pass          Target wgfx render pass.
     * @param view_proj     Camera view · projection matrix.
     * @param point_size    Base point size in screen pixels.
     * @param screen_width  Framebuffer pixel width.
     * @param screen_height Framebuffer pixel height.
     */
    void draw(wgfx::RenderPass* pass, const glm::mat4& view_proj, float point_size,
              float screen_width, float screen_height);

private:
    struct CameraUniforms {
        glm::mat4 view_proj;
        glm::vec4 cam_pos{0.0f};
        float     point_size    = 4.0f;
        float     screen_width  = 1280.0f;
        float     screen_height = 720.0f;
        float     pad0          = 0.0f;
    };

    struct GpuParticle {
        float x, y, z, pad0;
        float r, g, b, pad1;
    };

    wgfx::Pipeline*       pipeline_      = nullptr;
    wgfx::Uniform*        u_camera_      = nullptr;
    wgfx::Uniform*        s_particles_   = nullptr;
    std::size_t           particle_count_ = 0;
    std::size_t           buffer_capacity_ = 0;

    std::vector<GpuParticle> scratch_particles_;
};

} // namespace viz
