/**
 * @file volume_renderer.cpp
 * @brief Volumetric raymarching implementation for Eulerian fluid simulation.
 */

#include "viz/volume_renderer.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <algorithm>

namespace viz {

namespace {

constexpr const char* kVolumeShaderWGSL = R"wgsl(
struct VolumeUniforms {
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,       // xyz: cam_pos, w: grid_n
    domain_params: vec4<f32>, // xyz: domain size L, w: num_steps
    render_params: vec4<f32>, // x: field_mode, y: density_scale, z: threshold, w: palette
    light_params: vec4<f32>,  // xyz: light_dir, w: shadow_strength
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

fn get_field_density(p: vec3<f32>, mode: i32, n: u32, l: vec3<f32>) -> f32 {
    let f = evaluate_field(p, mode, n, l);
    return f.density_val;
}

fn compute_volumetric_shadow(p: vec3<f32>, mode: i32, n: u32, l: vec3<f32>, light_dir: vec3<f32>,
                             threshold: f32, density_s: f32, strength: f32) -> f32 {
    if (strength <= 0.01) { return 1.0; }
    let shadow_step = l.x / 20.0;
    var tau = 0.0;
    for (var i = 1; i <= 5; i = i + 1) {
        let sp = p + f32(i) * shadow_step * light_dir;
        if (sp.x >= 0.0 && sp.x <= l.x &&
            sp.y >= 0.0 && sp.y <= l.y &&
            sp.z >= 0.0 && sp.z <= l.z) {
            let dens = get_field_density(sp, mode, n, l);
            if (dens > threshold) {
                tau = tau + (dens - threshold) * density_s * shadow_step;
            }
        }
    }
    return exp(-tau * strength * 2.2);
}

fn compute_gradient_normal(p: vec3<f32>, mode: i32, n: u32, l: vec3<f32>) -> vec3<f32> {
    let eps = l.x / f32(n) * 1.0;
    let dx = get_field_density(p + vec3<f32>(eps, 0.0, 0.0), mode, n, l) -
             get_field_density(p - vec3<f32>(eps, 0.0, 0.0), mode, n, l);
    let dy = get_field_density(p + vec3<f32>(0.0, eps, 0.0), mode, n, l) -
             get_field_density(p - vec3<f32>(0.0, eps, 0.0), mode, n, l);
    let dz = get_field_density(p + vec3<f32>(0.0, 0.0, eps), mode, n, l) -
             get_field_density(p - vec3<f32>(0.0, 0.0, eps), mode, n, l);
    let g = vec3<f32>(dx, dy, dz);
    let glen = length(g);
    return select(vec3<f32>(0.0, 1.0, 0.0), -g / glen, glen > 1e-4);
}

fn get_color(t_norm: f32, palette: i32, phase_norm: f32, is_cyclic: bool, p: vec3<f32>, vel: vec3<f32>, l: vec3<f32>) -> vec3<f32> {
    if (palette == 5) {
        // SIGGRAPH 2016 Head-on collision (Red & Blue dye mixing into Purple)
        // Left stream (p.x < 0.5, vel.x > 0) is vibrant magenta / red-pink
        // Right stream (p.x > 0.5, vel.x < 0) is deep royal blue / cerulean
        // Collision center mixes into deep violet / purple
        let stream = (p.x / l.x - 0.5) * 3.5 + clamp(vel.x * 0.8, -0.8, 0.8);
        let mix_factor = clamp(stream * 0.5 + 0.5, 0.0, 1.0);

        let red_dye  = vec3<f32>(0.92, 0.16, 0.45); // vibrant magenta-red
        let blue_dye = vec3<f32>(0.20, 0.42, 0.98); // deep royal blue
        let purple   = vec3<f32>(0.58, 0.16, 0.82); // rich velvet purple

        var base_col: vec3<f32>;
        if (mix_factor < 0.45) {
            base_col = mix(red_dye, purple, mix_factor / 0.45);
        } else if (mix_factor > 0.55) {
            base_col = mix(purple, blue_dye, (mix_factor - 0.55) / 0.45);
        } else {
            base_col = purple;
        }

        let core_boost = clamp(t_norm * 0.4, 0.0, 0.5);
        return mix(base_col, vec3<f32>(1.0, 0.88, 0.95), core_boost);
    }

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

    let light_dir       = normalize(u_vol.light_params.xyz);
    let shadow_strength = u_vol.light_params.w;

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

            // Shading & illumination (fast unlit preview if shadow_strength <= 0.01)
            var illum = 1.0;
            if (shadow_strength > 0.01) {
                let shadow = compute_volumetric_shadow(p, field_mode, grid_n, box_max, light_dir, threshold, density_s, shadow_strength);
                let normal = compute_gradient_normal(p, field_mode, grid_n, box_max);
                let ndotl  = clamp(dot(normal, light_dir), 0.0, 1.0);

                let direct = (0.22 + 0.78 * ndotl) * shadow;
                let fill   = 0.20 * clamp(dot(normal, vec3<f32>(-light_dir.x, -0.35, -light_dir.z)), 0.0, 1.0);
                let rim    = pow(clamp(dot(rd, light_dir) * 0.5 + 0.5, 0.0, 1.0), 3.0) * 0.30 * shadow;
                illum  = direct + fill + rim;
            }

            let v_sample = sample_vel(p, grid_n, box_max);
            let raw_col  = get_color(norm_val, palette_idx, f.color_val, f.is_cyclic, p, v_sample, box_max);
            let col      = raw_col * illum;

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

    // Studio dark gradient background
    let bg_col = mix(vec3<f32>(0.07, 0.07, 0.08), vec3<f32>(0.015, 0.015, 0.02), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
    let final_rgb = acc_color + (1.0 - acc_alpha) * bg_col;

    return vec4<f32>(final_rgb, 1.0);
}
)wgsl";

constexpr const char* kBlitShaderWGSL = R"wgsl(
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(in_vertex_index) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@group(0) @binding(0) var s_sampler: sampler;
@group(0) @binding(1) var s_texture: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(s_texture, s_sampler, in.uv);
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

    // --- Sub-resolution preview upscaling pipeline ---
    wgpu::ShaderModuleWGSLDescriptor sm_wgsl{};
    sm_wgsl.chain.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
    sm_wgsl.code        = kBlitShaderWGSL;

    wgpu::ShaderModuleDescriptor sm_desc{};
    sm_desc.nextInChain = &sm_wgsl.chain;
    wgpu::ShaderModule sm = wgfx::device.createShaderModule(sm_desc);

    std::vector<wgpu::BindGroupLayoutEntry> bgl_entries(2);
    bgl_entries[0].binding                 = 0;
    bgl_entries[0].visibility              = wgpu::ShaderStage::Fragment;
    bgl_entries[0].sampler.type            = wgpu::SamplerBindingType::Filtering;

    bgl_entries[1].binding                 = 1;
    bgl_entries[1].visibility              = wgpu::ShaderStage::Fragment;
    bgl_entries[1].texture.sampleType      = wgpu::TextureSampleType::Float;
    bgl_entries[1].texture.viewDimension   = wgpu::TextureViewDimension::_2D;
    bgl_entries[1].texture.multisampled    = false;

    wgpu::BindGroupLayoutDescriptor bgl_desc{};
    bgl_desc.entryCount = static_cast<uint32_t>(bgl_entries.size());
    bgl_desc.entries    = bgl_entries.data();
    blit_bgl_ = wgfx::device.createBindGroupLayout(bgl_desc);

    WGPUBindGroupLayout raw_bgl = blit_bgl_;
    wgpu::PipelineLayoutDescriptor pl_desc{};
    pl_desc.bindGroupLayoutCount = 1;
    pl_desc.bindGroupLayouts     = &raw_bgl;
    wgpu::PipelineLayout pl      = wgfx::device.createPipelineLayout(pl_desc);

    wgpu::ColorTargetState color_target{};
    color_target.format    = wgfx::surfaceFormat;
    color_target.writeMask = wgpu::ColorWriteMask::All;
    color_target.blend     = nullptr;

    wgpu::FragmentState frag_state{};
    frag_state.module      = sm;
    frag_state.entryPoint  = "fs_main";
    frag_state.targetCount = 1;
    frag_state.targets     = &color_target;

    wgpu::RenderPipelineDescriptor rpd{};
    rpd.layout             = pl;
    rpd.vertex.module      = sm;
    rpd.vertex.entryPoint  = "vs_main";
    rpd.fragment           = &frag_state;
    rpd.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    rpd.primitive.cullMode = wgpu::CullMode::None;
    rpd.multisample.count  = 1;
    rpd.multisample.mask   = ~0u;
    rpd.multisample.alphaToCoverageEnabled = false;
    blit_pipeline_         = wgfx::device.createRenderPipeline(rpd);

    wgpu::SamplerDescriptor samp_desc{};
    samp_desc.minFilter     = wgpu::FilterMode::Linear;
    samp_desc.magFilter     = wgpu::FilterMode::Linear;
    samp_desc.mipmapFilter  = wgpu::MipmapFilterMode::Linear;
    samp_desc.addressModeU  = wgpu::AddressMode::ClampToEdge;
    samp_desc.addressModeV  = wgpu::AddressMode::ClampToEdge;
    samp_desc.addressModeW  = wgpu::AddressMode::ClampToEdge;
    samp_desc.maxAnisotropy = 1;
    samp_desc.lodMinClamp   = 0.0f;
    samp_desc.lodMaxClamp   = 32.0f;
    linear_sampler_         = wgfx::device.createSampler(samp_desc);

    samp_desc.minFilter     = wgpu::FilterMode::Nearest;
    samp_desc.magFilter     = wgpu::FilterMode::Nearest;
    samp_desc.mipmapFilter  = wgpu::MipmapFilterMode::Nearest;
    nearest_sampler_        = wgfx::device.createSampler(samp_desc);
}

VolumeRenderer::~VolumeRenderer()
{
    if (preview_tex_) {
        preview_tex_.destroy();
        preview_tex_.release();
        preview_tex_ = nullptr;
    }
    if (preview_view_) {
        preview_view_.release();
        preview_view_ = nullptr;
    }
    if (blit_bg_) {
        blit_bg_.release();
        blit_bg_ = nullptr;
    }
    if (blit_bgl_) {
        blit_bgl_.release();
        blit_bgl_ = nullptr;
    }
    if (blit_pipeline_) {
        blit_pipeline_.release();
        blit_pipeline_ = nullptr;
    }
    if (linear_sampler_) {
        linear_sampler_.release();
        linear_sampler_ = nullptr;
    }
    if (nearest_sampler_) {
        nearest_sampler_.release();
        nearest_sampler_ = nullptr;
    }
}

void VolumeRenderer::draw_to_encoder(wgpu::RenderPassEncoder rpe,
                                     wgpu::Buffer vel_buffer,
                                     wgpu::Buffer psi1_buffer,
                                     wgpu::Buffer psi2_buffer,
                                     int grid_n,
                                     float domain_size,
                                     const glm::mat4& inv_view_proj,
                                     const glm::vec3& cam_pos,
                                     const VolumeRenderParams& params,
                                     bool force_lighting)
{
    if (!rpe || !vel_buffer || !psi1_buffer || !psi2_buffer || grid_n <= 0) return;

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

    glm::vec3 l_dir = glm::normalize(glm::vec3(params.light_x, params.light_y, params.light_z));
    const float eff_shadow = (params.preview_lighting || force_lighting) ? params.shadow_strength : 0.0f;
    u.light_params   = glm::vec4(l_dir, eff_shadow);

    pipeline_->updateUniform(u_params_, reinterpret_cast<const float*>(&u));

    pipeline_->uniforms.clearForNewFrame();
    rpe.setBindGroup(0, pipeline_->uniforms.bindGroup, pipeline_->uniforms.dynamicUniformCount, pipeline_->uniforms.dynamicOffsets.data());
    rpe.setPipeline(pipeline_->pipeline);
    rpe.draw(pipeline_->vertices, pipeline_->instances, 0, 0);
}

void VolumeRenderer::ensure_preview_texture(int w, int h, bool linear)
{
    bool need_new_tex = (!preview_tex_ || preview_w_ != w || preview_h_ != h);
    bool need_new_bg  = (need_new_tex || !blit_bg_ || blit_linear_ != linear);

    if (need_new_tex) {
        if (preview_tex_) {
            preview_tex_.destroy();
            preview_tex_.release();
            preview_tex_ = nullptr;
        }
        if (preview_view_) {
            preview_view_.release();
            preview_view_ = nullptr;
        }

        preview_w_ = w;
        preview_h_ = h;

        wgpu::TextureDescriptor t_desc{};
        t_desc.dimension       = wgpu::TextureDimension::_2D;
        t_desc.format          = wgfx::surfaceFormat;
        t_desc.size            = { static_cast<uint32_t>(preview_w_), static_cast<uint32_t>(preview_h_), 1 };
        t_desc.sampleCount     = 1;
        t_desc.mipLevelCount   = 1;
        t_desc.usage           = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        preview_tex_           = wgfx::device.createTexture(t_desc);
        preview_view_          = preview_tex_.createView();
    }

    if (need_new_bg) {
        if (blit_bg_) {
            blit_bg_.release();
            blit_bg_ = nullptr;
        }
        blit_linear_ = linear;

        std::vector<wgpu::BindGroupEntry> bg_entries(2);
        bg_entries[0].binding     = 0;
        bg_entries[0].sampler     = linear ? linear_sampler_ : nearest_sampler_;
        bg_entries[1].binding     = 1;
        bg_entries[1].textureView = preview_view_;

        wgpu::BindGroupDescriptor bg_desc{};
        bg_desc.layout     = blit_bgl_;
        bg_desc.entryCount = bg_entries.size();
        bg_desc.entries    = bg_entries.data();
        blit_bg_           = wgfx::device.createBindGroup(bg_desc);
    }
}

std::pair<int, int> get_preview_dimensions(int fbw, int fbh, const VolumeRenderParams& params)
{
    if (fbw <= 0 || fbh <= 0) return { fbw, fbh };
    int target_h = 0;
    switch (params.preview_res) {
    case PreviewResolution::Native:
        return { fbw, fbh };
    case PreviewResolution::p720:
        target_h = 720;
        break;
    case PreviewResolution::p480:
        target_h = 480;
        break;
    case PreviewResolution::p360:
        target_h = 360;
        break;
    case PreviewResolution::p240:
        target_h = 240;
        break;
    case PreviewResolution::p180:
        target_h = 180;
        break;
    case PreviewResolution::p144:
        target_h = 144;
        break;
    case PreviewResolution::p72:
        target_h = 72;
        break;
    case PreviewResolution::Custom:
        target_h = std::clamp(static_cast<int>(fbh * params.preview_scale), 24, fbh);
        break;
    }
    if (target_h >= fbh) {
        return { fbw, fbh };
    }
    int target_w = std::max(1, (fbw * target_h) / fbh);
    return { target_w, target_h };
}

void VolumeRenderer::render(wgfx::RenderPass* pass,
                            wgpu::Buffer vel_buffer,
                            wgpu::Buffer psi1_buffer,
                            wgpu::Buffer psi2_buffer,
                            int grid_n,
                            float domain_size,
                            const glm::mat4& inv_view_proj,
                            const glm::vec3& cam_pos,
                            const VolumeRenderParams& params,
                            int fbw,
                            int fbh)
{
    if (!pass) return;

    auto [target_w, target_h] = get_preview_dimensions(fbw, fbh, params);

    // If native resolution or invalid dimensions, render directly into the scene pass
    if (fbw <= 0 || fbh <= 0 || (target_w >= fbw && target_h >= fbh)) {
        pass->prepare();
        draw_to_encoder(pass->renderPass, vel_buffer, psi1_buffer, psi2_buffer,
                        grid_n, domain_size, inv_view_proj, cam_pos, params, false);
        pass->end();
        return;
    }

    // 1. Render raymarching offscreen at the lower preview resolution FIRST
    // NOTE: In WebGPU, a CommandEncoder can only have one active render pass encoder at any time.
    // The offscreen low_res_pass must begin, draw, and end BEFORE pass->prepare() opens the swapchain pass.
    ensure_preview_texture(target_w, target_h, params.filter_linear);

    wgpu::RenderPassColorAttachment ca{};
    ca.view          = preview_view_;
    ca.resolveTarget = nullptr;
    ca.loadOp        = wgpu::LoadOp::Clear;
    ca.storeOp       = wgpu::StoreOp::Store;
    ca.clearValue    = { 0.02, 0.03, 0.06, 1.0 };
#ifdef __EMSCRIPTEN__
    ca.depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    wgpu::RenderPassDescriptor rp_desc{};
    rp_desc.colorAttachmentCount   = 1;
    rp_desc.colorAttachments       = &ca;
    rp_desc.depthStencilAttachment = nullptr;

    wgpu::RenderPassEncoder low_res_pass = wgfx::encoder.beginRenderPass(rp_desc);
    draw_to_encoder(low_res_pass, vel_buffer, psi1_buffer, psi2_buffer,
                    grid_n, domain_size, inv_view_proj, cam_pos, params, false);
    low_res_pass.end();
    low_res_pass.release();

    // 2. Now open the swapchain pass and upscale blit onto the full screen
    pass->prepare();
    pass->renderPass.setPipeline(blit_pipeline_);
    pass->renderPass.setBindGroup(0, blit_bg_, 0, nullptr);
    pass->renderPass.draw(3, 1, 0, 0);
    pass->end();
}

std::string VolumeRenderer::render_to_file(wgpu::Buffer vel_buffer,
                                           wgpu::Buffer psi1_buffer,
                                           wgpu::Buffer psi2_buffer,
                                           int grid_n,
                                           float domain_size,
                                           const glm::mat4& inv_view_proj,
                                           const glm::vec3& cam_pos,
                                           const VolumeRenderParams& params,
                                           int width,
                                           int height,
                                           const std::string& output_dir)
{
    if (width <= 0 || height <= 0 || !vel_buffer || !psi1_buffer || !psi2_buffer || grid_n <= 0) {
        return {};
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

    // Find next available sequential filename
    int index = 1;
    std::string out_path;
    while (true) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "render_%04d.bmp", index);
        auto p = std::filesystem::path(output_dir) / buf;
        if (!std::filesystem::exists(p, ec)) {
            out_path = p.string();
            break;
        }
        index++;
    }

    // 1. Create offscreen render target texture
    wgpu::TextureDescriptor tex_desc{};
    tex_desc.dimension       = wgpu::TextureDimension::_2D;
    tex_desc.format          = wgfx::surfaceFormat;
    tex_desc.size            = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    tex_desc.sampleCount     = 1;
    tex_desc.mipLevelCount   = 1;
    tex_desc.usage           = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc;

    wgpu::Texture offscreen_tex = wgfx::device.createTexture(tex_desc);
    if (!offscreen_tex) return {};
    wgpu::TextureView offscreen_view = offscreen_tex.createView();

    // 2. Offscreen render pass with full lighting
    wgpu::CommandEncoder encoder = wgfx::device.createCommandEncoder();

    wgpu::RenderPassColorAttachment color_att{};
    color_att.view          = offscreen_view;
    color_att.resolveTarget = nullptr;
    color_att.loadOp        = wgpu::LoadOp::Clear;
    color_att.storeOp       = wgpu::StoreOp::Store;
    color_att.clearValue    = { 0.02, 0.03, 0.06, 1.0 };
#ifdef __EMSCRIPTEN__
    color_att.depthSlice    = WGPU_DEPTH_SLICE_UNDEFINED;
#endif

    wgpu::RenderPassDescriptor rp_desc{};
    rp_desc.colorAttachmentCount   = 1;
    rp_desc.colorAttachments       = &color_att;
    rp_desc.depthStencilAttachment = nullptr;

    wgpu::RenderPassEncoder rpe = encoder.beginRenderPass(rp_desc);

    VolumeRenderParams lit_params = params;
    lit_params.shadow_strength = std::max(0.80f, params.shadow_strength);
    lit_params.step_count      = std::max(200.0f, params.step_count);

    draw_to_encoder(rpe, vel_buffer, psi1_buffer, psi2_buffer,
                    grid_n, domain_size, inv_view_proj, cam_pos,
                    lit_params, /*force_lighting=*/true);

    rpe.end();
    rpe.release();

    // 3. Staging buffer (aligned to 256 bytes per row)
    const uint32_t bytes_per_pixel = 4;
    const uint32_t pitch           = (static_cast<uint32_t>(width) * bytes_per_pixel + 255) & ~255;
    const uint32_t total_bytes     = pitch * static_cast<uint32_t>(height);

    wgpu::BufferDescriptor buf_desc{};
    buf_desc.label            = "Offline Render Readback Staging Buffer";
    buf_desc.usage            = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    buf_desc.size             = total_bytes;
    buf_desc.mappedAtCreation = false;
    wgpu::Buffer staging_buf   = wgfx::device.createBuffer(buf_desc);

    wgpu::ImageCopyTexture src_copy{};
    src_copy.texture  = offscreen_tex;
    src_copy.mipLevel = 0;
    src_copy.origin   = { 0, 0, 0 };
    src_copy.aspect   = wgpu::TextureAspect::All;

    wgpu::ImageCopyBuffer dst_copy{};
    dst_copy.buffer              = staging_buf;
    dst_copy.layout.offset       = 0;
    dst_copy.layout.bytesPerRow  = pitch;
    dst_copy.layout.rowsPerImage = static_cast<uint32_t>(height);

    wgpu::Extent3D copy_size = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };
    encoder.copyTextureToBuffer(src_copy, dst_copy, copy_size);

    wgpu::CommandBuffer cb = encoder.finish();
    encoder.release();

    wgfx::queue.submit(1, &cb);
    cb.release();

    // 4. Map buffer and wait synchronously via device.poll
    struct MapCtx {
        bool done = false;
        wgpu::BufferMapAsyncStatus status = wgpu::BufferMapAsyncStatus::Unknown;
    };
    MapCtx map_ctx;
    auto map_handle = staging_buf.mapAsync(wgpu::MapMode::Read, 0, total_bytes,
        [&map_ctx](wgpu::BufferMapAsyncStatus status) {
            map_ctx.status = status;
            map_ctx.done   = true;
        });

    while (!map_ctx.done) {
        wgfx::device.poll(true);
    }

    if (map_ctx.status != wgpu::BufferMapAsyncStatus::Success) {
        staging_buf.release();
        offscreen_view.release();
        offscreen_tex.release();
        return {};
    }

    const uint8_t* raw = static_cast<const uint8_t*>(staging_buf.getConstMappedRange(0, total_bytes));

    // 5. Write 32-bit BMP (BGRA)
#pragma pack(push, 1)
    struct BmpHeader {
        uint16_t file_type{0x4D42}; // 'BM'
        uint32_t file_size{0};
        uint16_t reserved1{0};
        uint16_t reserved2{0};
        uint32_t offset_data{54};
        uint32_t header_size{40};
        int32_t  width{0};
        int32_t  height{0};
        uint16_t planes{1};
        uint16_t bit_count{32};
        uint32_t compression{0};
        uint32_t size_image{0};
        int32_t  x_pels_per_meter{2835};
        int32_t  y_pels_per_meter{2835};
        uint32_t clr_used{0};
        uint32_t clr_important{0};
    };
#pragma pack(pop)

    BmpHeader hdr;
    hdr.width      = width;
    hdr.height     = height;
    hdr.size_image = static_cast<uint32_t>(width * height * 4);
    hdr.file_size  = 54 + hdr.size_image;

    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        staging_buf.unmap();
        staging_buf.release();
        offscreen_view.release();
        offscreen_tex.release();
        return {};
    }
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    const bool is_bgra = (wgfx::surfaceFormat == wgpu::TextureFormat::BGRA8Unorm ||
                          wgfx::surfaceFormat == wgpu::TextureFormat::BGRA8UnormSrgb);

    std::vector<uint8_t> row_buf(width * 4);
    for (int y = height - 1; y >= 0; --y) {
        const uint8_t* src_row = raw + y * pitch;
        if (is_bgra) {
            std::memcpy(row_buf.data(), src_row, width * 4);
        } else {
            for (int x = 0; x < width; ++x) {
                row_buf[x * 4 + 0] = src_row[x * 4 + 2]; // B
                row_buf[x * 4 + 1] = src_row[x * 4 + 1]; // G
                row_buf[x * 4 + 2] = src_row[x * 4 + 0]; // R
                row_buf[x * 4 + 3] = src_row[x * 4 + 3]; // A
            }
        }
        out.write(reinterpret_cast<const char*>(row_buf.data()), width * 4);
    }
    out.close();

    staging_buf.unmap();
    staging_buf.release();
    offscreen_view.release();
    offscreen_tex.release();

    return out_path;
}

} // namespace viz
