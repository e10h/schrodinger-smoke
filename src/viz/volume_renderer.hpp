/**
 * @file volume_renderer.hpp
 * @brief Volumetric raymarching renderer for Eulerian fluid fields.
 *
 * Casts rays through the 3D Eulerian domain and evaluates vorticity magnitude,
 * velocity speed, and enstrophy via trilinear sampling with Beer-Lambert absorption
 * and emission palettes.
 */

#pragma once

#include <wgfx.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace viz {

enum class VolumeFieldMode : int {
    VorticityMagnitude  = 0, ///< |∇ × v| (illuminates vortex rings and filaments)
    Speed               = 1, ///< |v| (flow velocity magnitude)
    Enstrophy           = 2, ///< |∇ × v|² (high-contrast vortex cores)
    Psi1Phase           = 3, ///< arg(ψ₁) (quantum phase angle of first spinor component)
    Psi2Phase           = 4, ///< arg(ψ₂) (quantum phase angle of second spinor component)
    Psi1Density         = 5, ///< |ψ₁|² (probability density; drops to 0 at vortex core)
    SpinorRelativePhase = 6, ///< arg(⟨ψ₁, ψ₂⟩) (spinor cross-phase winding vortex filaments)
};

enum class VolumePalette : int {
    NeonCyan          = 0, ///< Electric cyan/blue quantum filaments
    FirePlasma        = 1, ///< Hot thermal flame / plasma
    DeepOcean         = 2, ///< Deep marine azure to seafoam
    WispySmoke        = 3, ///< Classic realistic additive smoke
    PhaseRainbow      = 4, ///< Cyclic HSV spectral rainbow for quantum phase
    SiggraphCollision = 5, ///< SIGGRAPH 2016 Head-on collision (Red & Blue dye mixing into Purple)
};

struct VolumeRenderParams {
    VolumeFieldMode field_mode      = VolumeFieldMode::VorticityMagnitude;
    VolumePalette   palette         = VolumePalette::SiggraphCollision; ///< Default to iconic SIGGRAPH 2016 look
    float           density_scale   = 18.0f;
    float           threshold       = 0.025f;
    float           step_count      = 160.0f;
    float           shadow_strength = 0.80f; ///< Volumetric self-shadowing (0 to 1.5)
    float           light_x         = 0.40f; ///< Key light direction X
    float           light_y         = 0.85f; ///< Key light direction Y
    float           light_z         = 0.45f; ///< Key light direction Z
};

class VolumeRenderer {
public:
    VolumeRenderer();
    ~VolumeRenderer();

    VolumeRenderer(const VolumeRenderer&)            = delete;
    VolumeRenderer& operator=(const VolumeRenderer&) = delete;

    /**
     * @brief Render the Eulerian fluid field and/or ψ wavefunction as raytraced voxels.
     *
     * @param pass          Active wgfx render pass.
     * @param vel_buffer    WebGPU storage buffer holding vec4<f32> * N^3 velocity field.
     * @param psi1_buffer   WebGPU storage buffer holding vec2<f32> * N^3 ψ₁ spinor component.
     * @param psi2_buffer   WebGPU storage buffer holding vec2<f32> * N^3 ψ₂ spinor component.
     * @param grid_n        Grid resolution N.
     * @param domain_size   Domain size (meters).
     * @param inv_view_proj Inverse camera view · projection matrix.
     * @param cam_pos       World camera position.
     * @param params        Visual tuning parameters.
     */
    void render(wgfx::RenderPass* pass,
                wgpu::Buffer vel_buffer,
                wgpu::Buffer psi1_buffer,
                wgpu::Buffer psi2_buffer,
                int grid_n,
                float domain_size,
                const glm::mat4& inv_view_proj,
                const glm::vec3& cam_pos,
                const VolumeRenderParams& params);

private:
    struct UniformData {
        glm::mat4 inv_view_proj;
        glm::vec4 cam_pos;       // xyz: cam_pos, w: grid_n
        glm::vec4 domain_params; // xyz: domain size L, w: num_steps
        glm::vec4 render_params; // x: field_mode, y: density_scale, z: threshold, w: palette
        glm::vec4 light_params;  // xyz: light_dir, w: shadow_strength
    };

    wgfx::Pipeline* pipeline_      = nullptr;
    wgfx::Uniform*  u_params_      = nullptr;
    wgfx::Uniform*  s_velocity_    = nullptr;
    wgfx::Uniform*  s_psi1_        = nullptr;
    wgfx::Uniform*  s_psi2_        = nullptr;
    wgpu::Buffer    bound_buffer_  = nullptr;
    wgpu::Buffer    bound_psi1_    = nullptr;
    wgpu::Buffer    bound_psi2_    = nullptr;
    int             bound_grid_n_  = 0;
};

} // namespace viz
