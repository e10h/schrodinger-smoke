/**
 * @file volume_renderer.cpp
 * @brief Volumetric raymarching implementation for Eulerian fluid simulation.
 */

#include "viz/volume_renderer.hpp"

#include <iostream>

namespace viz {

namespace {

constexpr const char* kVolumeShaderWGSL = R"wgsl(
struct VolumeUniforms {
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,       // xyz: cam_pos, w: grid_n
    domain_params: vec4<f32>, // xyz: domain size L, w: num_steps
    render_params: vec4<f32>, // x: field_mode, y: density_scale, z: threshold, w: palette
};

@group(0) @binding(0) var<uniform> u_vol: VolumeUniforms;
@group(0) @binding(1) var<storage, read> s_vel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_psi1: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> s_psi2: array<vec2<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VertexOutput;
    out.position = vec4<f32>(pos[vid], 0.0, 1.0);
    out.uv = pos[vid];
    return out;
}

fn cell_index(i: u32, j: u32, k: u32, n: u32) -> u32 {
    return i + n * (j + n * k);
}

fn wrap_pos(v: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    let r = v - floor(v / l) * l;
    return clamp(r, vec3<f32>(0.0), l - vec3<f32>(1e-5));
}

fn sample_vel(p_in: vec3<f32>, n: u32, l: vec3<f32>) -> vec3<f32> {
    let p = wrap_pos(p_in, l);
    let u = p.x / l.x * f32(n);
    let v = p.y / l.y * f32(n);
    let w = p.z / l.z * f32(n);

    let i0 = u32(clamp(floor(u), 0.0, f32(n - 1u)));
    let j0 = u32(clamp(floor(v), 0.0, f32(n - 1u)));
    let k0 = u32(clamp(floor(w), 0.0, f32(n - 1u)));
    let i1 = (i0 + 1u) % n;
    let j1 = (j0 + 1u) % n;
    let k1 = (k0 + 1u) % n;

    let fx = clamp(u - floor(u), 0.0, 1.0);
    let fy = clamp(v - floor(v), 0.0, 1.0);
    let fz = clamp(w - floor(w), 0.0, 1.0);

    let c000 = s_vel[cell_index(i0, j0, k0, n)].xyz;
    let c100 = s_vel[cell_index(i1, j0, k0, n)].xyz;
    let c010 = s_vel[cell_index(i0, j1, k0, n)].xyz;
    let c110 = s_vel[cell_index(i1, j1, k0, n)].xyz;
    let c001 = s_vel[cell_index(i0, j0, k1, n)].xyz;
    let c101 = s_vel[cell_index(i1, j0, k1, n)].xyz;
    let c011 = s_vel[cell_index(i0, j1, k1, n)].xyz;
    let c111 = s_vel[cell_index(i1, j1, k1, n)].xyz;

    let c00 = mix(c000, c100, fx);
    let c10 = mix(c010, c110, fx);
    let c01 = mix(c001, c101, fx);
    let c11 = mix(c011, c111, fx);

    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);

    return mix(c0, c1, fz);
}

fn sample_psi1(p_in: vec3<f32>, n: u32, l: vec3<f32>) -> vec2<f32> {
    let p = wrap_pos(p_in, l);
    let u = p.x / l.x * f32(n);
    let v = p.y / l.y * f32(n);
    let w = p.z / l.z * f32(n);

    let i0 = u32(clamp(floor(u), 0.0, f32(n - 1u)));
    let j0 = u32(clamp(floor(v), 0.0, f32(n - 1u)));
    let k0 = u32(clamp(floor(w), 0.0, f32(n - 1u)));
    let i1 = (i0 + 1u) % n;
    let j1 = (j0 + 1u) % n;
    let k1 = (k0 + 1u) % n;

    let fx = clamp(u - floor(u), 0.0, 1.0);
    let fy = clamp(v - floor(v), 0.0, 1.0);
    let fz = clamp(w - floor(w), 0.0, 1.0);

    let c000 = s_psi1[cell_index(i0, j0, k0, n)];
    let c100 = s_psi1[cell_index(i1, j0, k0, n)];
    let c010 = s_psi1[cell_index(i0, j1, k0, n)];
    let c110 = s_psi1[cell_index(i1, j1, k0, n)];
    let c001 = s_psi1[cell_index(i0, j0, k1, n)];
    let c101 = s_psi1[cell_index(i1, j0, k1, n)];
    let c011 = s_psi1[cell_index(i0, j1, k1, n)];
    let c111 = s_psi1[cell_index(i1, j1, k1, n)];

    let c00 = mix(c000, c100, fx);
    let c10 = mix(c010, c110, fx);
    let c01 = mix(c001, c101, fx);
    let c11 = mix(c011, c111, fx);

    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);

    return mix(c0, c1, fz);
}

fn sample_psi2(p_in: vec3<f32>, n: u32, l: vec3<f32>) -> vec2<f32> {
    let p = wrap_pos(p_in, l);
    let u = p.x / l.x * f32(n);
    let v = p.y / l.y * f32(n);
    let w = p.z / l.z * f32(n);

    let i0 = u32(clamp(floor(u), 0.0, f32(n - 1u)));
    let j0 = u32(clamp(floor(v), 0.0, f32(n - 1u)));
    let k0 = u32(clamp(floor(w), 0.0, f32(n - 1u)));
    let i1 = (i0 + 1u) % n;
    let j1 = (j0 + 1u) % n;
    let k1 = (k0 + 1u) % n;

    let fx = clamp(u - floor(u), 0.0, 1.0);
    let fy = clamp(v - floor(v), 0.0, 1.0);
    let fz = clamp(w - floor(w), 0.0, 1.0);

    let c000 = s_psi2[cell_index(i0, j0, k0, n)];
    let c100 = s_psi2[cell_index(i1, j0, k0, n)];
    let c010 = s_psi2[cell_index(i0, j1, k0, n)];
    let c110 = s_psi2[cell_index(i1, j1, k0, n)];
    let c001 = s_psi2[cell_index(i0, j0, k1, n)];
    let c101 = s_psi2[cell_index(i1, j0, k1, n)];
    let c011 = s_psi2[cell_index(i0, j1, k1, n)];
    let c111 = s_psi2[cell_index(i1, j1, k1, n)];

    let c00 = mix(c000, c100, fx);
    let c10 = mix(c010, c110, fx);
    let c01 = mix(c001, c101, fx);
    let c11 = mix(c011, c111, fx);

    let c0 = mix(c00, c10, fy);
    let c1 = mix(c01, c11, fy);

    return mix(c0, c1, fz);
}

fn sample_vorticity(p: vec3<f32>, n: u32, l: vec3<f32>) -> f32 {
    let h = (l.x / f32(n)) * 1.0;

    let xp = sample_vel(p + vec3<f32>(h, 0.0, 0.0), n, l);
    let xm = sample_vel(p - vec3<f32>(h, 0.0, 0.0), n, l);
    let yp = sample_vel(p + vec3<f32>(0.0, h, 0.0), n, l);
    let ym = sample_vel(p - vec3<f32>(0.0, h, 0.0), n, l);
    let zp = sample_vel(p + vec3<f32>(0.0, 0.0, h), n, l);
    let zm = sample_vel(p - vec3<f32>(0.0, 0.0, h), n, l);

    let inv_2h = 0.5 / h;
    let curl_x = (yp.z - ym.z) * inv_2h - (zp.y - zm.y) * inv_2h;
    let curl_y = (zp.x - zm.x) * inv_2h - (xp.z - xm.z) * inv_2h;
    let curl_z = (xp.y - ym.y) * inv_2h - (yp.x - ym.x) * inv_2h;

    return length(vec3<f32>(curl_x, curl_y, curl_z));
}

struct FieldResult {
    density_val: f32,
    color_val: f32,
    is_cyclic: bool,
};

fn evaluate_field(p: vec3<f32>, mode: i32, n: u32, l: vec3<f32>) -> FieldResult {
    var res: FieldResult;
    res.is_cyclic = false;

    if (mode == 0) {
        // Vorticity magnitude (normalized to ~[0, 10] range)
        let v = sample_vorticity(p, n, l) * 0.1;
        res.density_val = v;
        res.color_val   = v;
    } else if (mode == 1) {
        // Velocity speed (magnitude in m/s, ~[0, 3] range)
        let v = length(sample_vel(p, n, l));
        res.density_val = v;
        res.color_val   = v;
    } else if (mode == 2) {
        // Enstrophy (|curl|^2)
        let vort = sample_vorticity(p, n, l) * 0.1;
        let v = vort * vort * 0.5;
        res.density_val = v;
        res.color_val   = v;
    } else if (mode == 3) {
        // ψ₁ Phase arg(ψ₁)
        let c = sample_psi1(p, n, l);
        var ph = (atan2(c.y, c.x) + 3.14159265) / 6.2831853;
        if (ph < 0.0) { ph += 1.0; }
        let mag = length(c);
        // Phase is visible where wavefunction exists; density emphasizes phase boundaries
        res.density_val = mag;
        res.color_val   = ph;
        res.is_cyclic   = true;
    } else if (mode == 4) {
        // ψ₂ Phase arg(ψ₂)
        let c = sample_psi2(p, n, l);
        var ph = (atan2(c.y, c.x) + 3.14159265) / 6.2831853;
        if (ph < 0.0) { ph += 1.0; }
        let mag = length(c);
        res.density_val = mag;
        res.color_val   = ph;
        res.is_cyclic   = true;
    } else if (mode == 5) {
        // ψ₁ Probability Density |ψ₁|² (drops to zero at vortex filaments)
        let c = sample_psi1(p, n, l);
        let rho = dot(c, c);
        res.density_val = rho;
        res.color_val   = rho;
    } else {
        // Spinor Cross Phase arg(⟨ψ₁, ψ₂⟩) — helical phase around vortex lines
        let p1 = sample_psi1(p, n, l);
        let p2 = sample_psi2(p, n, l);
        let re = p1.x * p2.x + p1.y * p2.y;
        let im = p1.x * p2.y - p1.y * p2.x;
        var ph = (atan2(im, re) + 3.14159265) / 6.2831853;
        if (ph < 0.0) { ph += 1.0; }
        let mag = length(p1) * length(p2) * 2.0;
        res.density_val = mag;
        res.color_val   = ph;
        res.is_cyclic   = true;
    }
    return res;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let hue = fract(h) * 6.0;
    let c = v * s;
    let x = c * (1.0 - abs(fract(hue * 0.5) * 2.0 - 1.0));
    let m = v - c;
    if (hue < 1.0) { return vec3<f32>(c + m, x + m, m); }
    if (hue < 2.0) { return vec3<f32>(x + m, c + m, m); }
    if (hue < 3.0) { return vec3<f32>(m, c + m, x + m); }
    if (hue < 4.0) { return vec3<f32>(m, x + m, c + m); }
    if (hue < 5.0) { return vec3<f32>(x + m, m, c + m); }
    return vec3<f32>(c + m, m, x + m);
}

fn get_color(t_norm: f32, palette: i32, phase_norm: f32, is_cyclic: bool) -> vec3<f32> {
    if (palette == 4 || (is_cyclic && palette == 0)) {
        // Phase Rainbow / Quantum Phase: cyclic spectral hues
        let ph = select(t_norm, phase_norm, is_cyclic);
        return hsv_to_rgb(ph, 0.90, 0.25 + 0.75 * clamp(t_norm, 0.0, 1.0));
    }

    let t = clamp(t_norm, 0.0, 1.0);
    if (palette == 0) {
        // Neon Cyan: dark sapphire -> bright electric cyan -> white
        let c0 = vec3<f32>(0.05, 0.15, 0.45);
        let c1 = vec3<f32>(0.10, 0.75, 1.00);
        let c2 = vec3<f32>(0.95, 0.98, 1.00);
        return select(mix(c0, c1, t * 2.0), mix(c1, c2, (t - 0.5) * 2.0), t >= 0.5);
    } else if (palette == 1) {
        // Fire Plasma: deep red -> orange -> yellow -> white
        let c0 = vec3<f32>(0.50, 0.05, 0.02);
        let c1 = vec3<f32>(1.00, 0.45, 0.05);
        let c2 = vec3<f32>(1.00, 0.95, 0.60);
        return select(mix(c0, c1, t * 2.0), mix(c1, c2, (t - 0.5) * 2.0), t >= 0.5);
    } else if (palette == 2) {
        // Deep Ocean: deep navy -> cyan -> seafoam
        let c0 = vec3<f32>(0.02, 0.10, 0.30);
        let c1 = vec3<f32>(0.15, 0.60, 0.70);
        let c2 = vec3<f32>(0.60, 0.95, 0.85);
        return select(mix(c0, c1, t * 2.0), mix(c1, c2, (t - 0.5) * 2.0), t >= 0.5);
    } else {
        // Wispy Smoke: soft pearl grey with slight cool tint
        return mix(vec3<f32>(0.45, 0.50, 0.60), vec3<f32>(0.95, 0.95, 0.98), t);
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Unproject to world-space far plane
    let far_h  = u_vol.inv_view_proj * vec4<f32>(in.uv.x, in.uv.y, 1.0, 1.0);
    let far_pt = far_h.xyz / far_h.w;

    let ro = u_vol.cam_pos.xyz;
    let rd = normalize(far_pt - ro);

    let box_min = vec3<f32>(0.0);
    let box_max = u_vol.domain_params.xyz;

    // Ray-box intersection (slab method)
    // Avoid division by exact zero which yields NaNs
    let safe_rd = select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let inv_rd = 1.0 / safe_rd;
    let t0 = (box_min - ro) * inv_rd;
    let t1 = (box_max - ro) * inv_rd;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit  = min(min(tmax.x, tmax.y), tmax.z);

    if (t_exit <= max(t_enter, 0.0)) {
        discard;
    }

    let t_start = max(t_enter, 0.0);
    let t_dist  = t_exit - t_start;
    let num_steps = max(u_vol.domain_params.w, 16.0);
    let dt = t_dist / num_steps;

    let grid_n       = u32(u_vol.cam_pos.w);
    let field_mode   = i32(u_vol.render_params.x);
    let density_s    = u_vol.render_params.y;
    let threshold    = u_vol.render_params.z;
    let palette_idx  = i32(u_vol.render_params.w);

    var acc_color = vec3<f32>(0.0);
    var acc_alpha = 0.0;

    var t = t_start + 0.5 * dt;
    for (var s = 0; s < 256; s = s + 1) {
        if (f32(s) >= num_steps || t >= t_exit) { break; }

        let p = ro + t * rd;
        let f = evaluate_field(p, field_mode, grid_n, box_max);

        if (f.density_val > threshold) {
            let density = (f.density_val - threshold) * density_s;
            let sample_alpha = 1.0 - exp(-density * dt);
            let norm_val = clamp((f.density_val - threshold) / max(threshold * 8.0, 0.02), 0.0, 1.0);
            let col = get_color(norm_val, palette_idx, f.color_val, f.is_cyclic);

            // Front-to-back compositing
            acc_color = acc_color + (1.0 - acc_alpha) * col * sample_alpha;
            acc_alpha = acc_alpha + (1.0 - acc_alpha) * sample_alpha;

            if (acc_alpha >= 0.98) { break; }
        }

        t = t + dt;
    }

    if (acc_alpha <= 0.001) {
        discard;
    }

    return vec4<f32>(acc_color, acc_alpha);
}
)wgsl";

} // namespace

VolumeRenderer::VolumeRenderer()
{
    pipeline_ = wgfx::loadPipeline(kVolumeShaderWGSL);
    pipeline_->useDepth  = false;
    pipeline_->vertices  = 3; // single full-screen triangle
    pipeline_->instances = 1;

    u_params_ = pipeline_->addUniform(0, sizeof(UniformData));

    // Storage buffer bindings:
    // Binding 1: velocity field (vec4<f32> * N^3)
    // Binding 2: ψ₁ spinor component (vec2<f32> * N^3)
    // Binding 3: ψ₂ spinor component (vec2<f32> * N^3)
    s_velocity_ = pipeline_->addStorage(1, 16, nullptr, /*readOnly=*/true);
    s_psi1_     = pipeline_->addStorage(2, 16, nullptr, /*readOnly=*/true);
    s_psi2_     = pipeline_->addStorage(3, 16, nullptr, /*readOnly=*/true);

    // Ensure minBindingSize = 0 for arbitrary resolutions (32^3 to 256^3)
    for (auto& layout : pipeline_->uniforms.layouts) {
        if (layout.binding >= 1) {
            layout.buffer.minBindingSize = 0;
        }
    }

    // Premultiplied alpha blending for luminous volumetric smoke
    pipeline_->customBlend = true;
    pipeline_->fragmentBlend.color.srcFactor = wgpu::BlendFactor::One;
    pipeline_->fragmentBlend.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    pipeline_->fragmentBlend.color.operation = wgpu::BlendOperation::Add;
    pipeline_->fragmentBlend.alpha.srcFactor = wgpu::BlendFactor::One;
    pipeline_->fragmentBlend.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    pipeline_->fragmentBlend.alpha.operation = wgpu::BlendOperation::Add;

    pipeline_->init(nullptr);
}

VolumeRenderer::~VolumeRenderer()
{
}

void VolumeRenderer::render(wgfx::RenderPass* pass,
                            wgpu::Buffer vel_buffer,
                            wgpu::Buffer psi1_buffer,
                            wgpu::Buffer psi2_buffer,
                            int grid_n,
                            float domain_size,
                            const glm::mat4& inv_view_proj,
                            const glm::vec3& cam_pos,
                            const VolumeRenderParams& params)
{
    if (!pass || !vel_buffer || !psi1_buffer || !psi2_buffer || grid_n <= 0) return;

    // Recreate bind group whenever buffers or resolution change
    if (bound_buffer_ != vel_buffer || bound_psi1_ != psi1_buffer || bound_psi2_ != psi2_buffer || bound_grid_n_ != grid_n) {
        bound_buffer_ = vel_buffer;
        bound_psi1_   = psi1_buffer;
        bound_psi2_   = psi2_buffer;
        bound_grid_n_ = grid_n;
        const size_t vel_size  = static_cast<size_t>(grid_n) * grid_n * grid_n * sizeof(float) * 4;
        const size_t cplx_size = static_cast<size_t>(grid_n) * grid_n * grid_n * sizeof(float) * 2;

        std::vector<wgpu::BindGroupEntry> bg_entries(4);
        bg_entries[0].binding = 0;
        bg_entries[0].buffer  = u_params_->buffer;
        bg_entries[0].offset  = 0;
        bg_entries[0].size    = sizeof(UniformData);

        bg_entries[1].binding = 1;
        bg_entries[1].buffer  = vel_buffer;
        bg_entries[1].offset  = 0;
        bg_entries[1].size    = vel_size;

        bg_entries[2].binding = 2;
        bg_entries[2].buffer  = psi1_buffer;
        bg_entries[2].offset  = 0;
        bg_entries[2].size    = cplx_size;

        bg_entries[3].binding = 3;
        bg_entries[3].buffer  = psi2_buffer;
        bg_entries[3].offset  = 0;
        bg_entries[3].size    = cplx_size;

        wgpu::BindGroupDescriptor bgDesc{};
        bgDesc.layout     = pipeline_->uniforms.bindGroupLayout;
        bgDesc.entryCount = bg_entries.size();
        bgDesc.entries    = bg_entries.data();

        if (pipeline_->uniforms.bindGroup) {
            pipeline_->uniforms.bindGroup.release();
        }
        pipeline_->uniforms.bindGroup = wgfx::device.createBindGroup(bgDesc);
    }

    UniformData u{};
    u.inv_view_proj  = inv_view_proj;
    u.cam_pos        = glm::vec4(cam_pos, static_cast<float>(grid_n));
    u.domain_params  = glm::vec4(domain_size, domain_size, domain_size, params.step_count);
    u.render_params  = glm::vec4(static_cast<float>(params.field_mode),
                                 params.density_scale,
                                 params.threshold,
                                 static_cast<float>(params.palette));

    pipeline_->updateUniform(u_params_, reinterpret_cast<const float*>(&u));
    pass->draw(pipeline_);
}

} // namespace viz
