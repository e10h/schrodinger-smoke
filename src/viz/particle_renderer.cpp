/**
 * @file particle_renderer.cpp
 * @brief WebGPU implementation of tracer-particle billboard renderer.
 */

#include "viz/particle_renderer.hpp"
#include "viz/particles.hpp"

#include <cstring>
#include <iostream>

namespace viz {

namespace {

constexpr const char* kParticleShaderWGSL = R"wgsl(
struct CameraUniforms {
    view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    light_dir: vec4<f32>,
    point_size: f32,
    screen_width: f32,
    screen_height: f32,
    _pad0: f32,
};

struct ParticleData {
    pos: vec4<f32>,
    col: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_cam: CameraUniforms;
@group(0) @binding(1) var<storage, read> s_particles: array<ParticleData>;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) sphere_radius: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
        vec2<f32>(-0.5, -0.5)
    );

    let corner = corners[in.vertex_index];
    let p = s_particles[in.instance_index];

    let center_clip = u_cam.view_proj * vec4<f32>(p.pos.xyz, 1.0);

    // Perspective point size attenuation:
    let pt_pixels = u_cam.point_size / max(center_clip.w, 0.1);
    let offset_x = corner.x * (pt_pixels / u_cam.screen_width) * 2.0 * center_clip.w;
    let offset_y = corner.y * (pt_pixels / u_cam.screen_height) * 2.0 * center_clip.w;

    var out: VertexOutput;
    out.position = center_clip + vec4<f32>(offset_x, offset_y, 0.0, 0.0);
    out.uv = corner * 2.0; // [-1.0, 1.0] across billboard
    out.color = p.col;
    out.world_pos = p.pos.xyz;
    out.sphere_radius = (u_cam.point_size / 720.0) * center_clip.w * 0.5;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }

    // 3D Sphere Normal: z = sqrt(1 - x^2 - y^2) in view tangent space
    let nz = sqrt(max(0.0, 1.0 - r2));
    let view_normal = normalize(vec3<f32>(in.uv.x, in.uv.y, nz));

    // Reconstruct world normal from camera view
    let to_cam = normalize(u_cam.cam_pos.xyz - in.world_pos);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_cam));
    let up = cross(to_cam, right);
    let world_normal = normalize(right * view_normal.x + up * view_normal.y + to_cam * view_normal.z);

    // Key Directional Light (with warm rim / wrap diffuse like Mantra PBR)
    let light = normalize(u_cam.light_dir.xyz);
    let NdotL = dot(world_normal, light);
    let diffuse = max(0.0, (NdotL + 0.3) / 1.3); // Half-Lambert wrap for soft cloud shading

    // Ambient Occlusion from sphere curvature
    let curvature_ao = smoothstep(0.0, 0.8, nz);

    // Ambient fill light (cool blue sky fill)
    let ambient = vec3<f32>(0.22, 0.24, 0.32);

    // Combine diffuse and ambient
    let final_lighting = ambient + vec3<f32>(0.95, 0.90, 0.85) * diffuse * curvature_ao;

    // Albedo color (from particles: red / blue)
    let final_rgb = in.color.rgb * final_lighting;

    // Edge antialiasing
    let edge_alpha = smoothstep(1.0, 0.85, sqrt(r2)) * in.color.a;

    return vec4<f32>(final_rgb, edge_alpha);
}
)wgsl";

} // namespace

ParticleRenderer::ParticleRenderer()
{
    pipeline_ = wgfx::loadPipeline(kParticleShaderWGSL);
    pipeline_->useDepth  = true; // Enable depth testing so overlapping spheres form a continuous opaque 3D volume
    pipeline_->depthFormat = wgpu::TextureFormat::Depth24Plus;
    pipeline_->vertices  = 6; // 6 vertices per quad instance
    pipeline_->instances = 1;

    // Uniform binding 0: Camera uniforms
    u_camera_ = pipeline_->addUniform(0, sizeof(CameraUniforms));

    // Storage buffer binding 1: particle array
    // Pre-allocate for up to 600,000 particles (slider max)
    buffer_capacity_ = 600000;
    scratch_particles_.resize(buffer_capacity_);
    s_particles_ = pipeline_->addStorage(1, buffer_capacity_ * sizeof(GpuParticle),
                                         scratch_particles_.data(), /*readOnly=*/true);

    // Standard alpha blending: SrcAlpha + OneMinusSrcAlpha for solid opaque clouds
    pipeline_->customBlend = true;
    pipeline_->fragmentBlend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    pipeline_->fragmentBlend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    pipeline_->fragmentBlend.color.operation = wgpu::BlendOperation::Add;
    pipeline_->fragmentBlend.alpha.srcFactor = wgpu::BlendFactor::One;
    pipeline_->fragmentBlend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    pipeline_->fragmentBlend.alpha.operation = wgpu::BlendOperation::Add;

    pipeline_->init(nullptr);
}

ParticleRenderer::~ParticleRenderer()
{
}

void ParticleRenderer::upload(const ParticleSystem& ps)
{
    particle_count_ = ps.size();
    if (particle_count_ == 0) return;

    if (particle_count_ > buffer_capacity_) {
        buffer_capacity_ = particle_count_;
        scratch_particles_.resize(buffer_capacity_);
    }

    const bool has_cols = ps.has_colors();
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(particle_count_); ++i) {
        const float r = has_cols ? ps.r[i] : 0.2f;
        const float g = has_cols ? ps.g[i] : 0.8f;
        const float b = has_cols ? ps.b[i] : 1.0f;
        uint32_t packed = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f)) |
                         (static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f)) << 8) |
                         (static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f)) << 16) |
                         (0xFFu << 24);
        float packed_f;
        std::memcpy(&packed_f, &packed, sizeof(float));

        scratch_particles_[i] = {
            static_cast<float>(ps.x[i]),
            static_cast<float>(ps.y[i]),
            static_cast<float>(ps.z[i]),
            packed_f,
            r,
            g,
            b,
            1.0f
        };
    }

    pipeline_->updateStorageBuffer(s_particles_, scratch_particles_.data(),
                                   particle_count_ * sizeof(GpuParticle), 0);
    pipeline_->instances = static_cast<int>(particle_count_);
}

void ParticleRenderer::set_particle_count(std::size_t count) noexcept
{
    particle_count_ = count;
    if (pipeline_) {
        pipeline_->instances = static_cast<int>(count);
    }
}

void ParticleRenderer::draw(wgfx::RenderPass* pass, const glm::mat4& view_proj,
                            const glm::vec3& cam_pos, float point_size,
                            float screen_width, float screen_height,
                            const glm::vec3& light_dir)
{
    if (particle_count_ == 0 || !pass) return;

    CameraUniforms cam_u{};
    cam_u.view_proj     = view_proj;
    cam_u.cam_pos       = glm::vec4(cam_pos, 1.0f);
    cam_u.light_dir     = glm::vec4(glm::normalize(light_dir), 0.0f);
    cam_u.point_size    = point_size;
    cam_u.screen_width  = screen_width;
    cam_u.screen_height = screen_height;

    pipeline_->updateUniform(u_camera_, reinterpret_cast<const float*>(&cam_u));
    pass->draw(pipeline_);
}

} // namespace viz
