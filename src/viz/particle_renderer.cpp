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
    @location(1) color: vec3<f32>,
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

    // Exact OpenGL gl_PointSize perspective depth attenuation:
    // gl_PointSize = u_point_size / max(clip.w, 0.1);
    // In clip-space [-1, 1], pixel size is (gl_PointSize / screen_dim) * 2.0 * clip.w
    let pt_pixels = u_cam.point_size / max(center_clip.w, 0.1);
    let offset_x = corner.x * (pt_pixels / u_cam.screen_width) * 2.0 * center_clip.w;
    let offset_y = corner.y * (pt_pixels / u_cam.screen_height) * 2.0 * center_clip.w;

    var out: VertexOutput;
    out.position = center_clip + vec4<f32>(offset_x, offset_y, 0.0, 0.0);
    out.uv = corner + vec2<f32>(0.5, 0.5);
    out.color = p.col.rgb;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = in.uv - vec2<f32>(0.5, 0.5);
    let r = length(d);
    if (r > 0.5) {
        discard;
    }

    // Exact OpenGL fragment equation:
    // float a = smoothstep(0.5, 0.0, r);
    // out_color = vec4(v_color, a * 0.22);
    // Combined with glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    let a = smoothstep(0.5, 0.0, r) * 0.22;
    return vec4<f32>(in.color, a);
}
)wgsl";

} // namespace

ParticleRenderer::ParticleRenderer()
{
    pipeline_ = wgfx::loadPipeline(kParticleShaderWGSL);
    pipeline_->useDepth  = false;
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

    // Exact match to OpenGL: glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    pipeline_->customBlend = true;
    pipeline_->fragmentBlend.color.srcFactor = wgpu::BlendFactor::SrcAlpha;
    pipeline_->fragmentBlend.color.dstFactor = wgpu::BlendFactor::One;
    pipeline_->fragmentBlend.color.operation = wgpu::BlendOperation::Add;
    pipeline_->fragmentBlend.alpha.srcFactor = wgpu::BlendFactor::Zero;
    pipeline_->fragmentBlend.alpha.dstFactor = wgpu::BlendFactor::One;
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

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(particle_count_); ++i) {
        scratch_particles_[i] = {
            static_cast<float>(ps.x[i]),
            static_cast<float>(ps.y[i]),
            static_cast<float>(ps.z[i]),
            0.0f,
            ps.r[i],
            ps.g[i],
            ps.b[i],
            0.0f
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
                            float point_size, float screen_width, float screen_height)
{
    if (particle_count_ == 0 || !pass) return;

    CameraUniforms cam_u{};
    cam_u.view_proj     = view_proj;
    cam_u.point_size    = point_size;
    cam_u.screen_width  = screen_width;
    cam_u.screen_height = screen_height;

    pipeline_->updateUniform(u_camera_, reinterpret_cast<const float*>(&cam_u));
    pass->draw(pipeline_);
}

} // namespace viz
