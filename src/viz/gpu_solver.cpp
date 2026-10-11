/**
 * @file gpu_solver.cpp
 * @brief WebGPU compute-shader ISF fluid solver implementation.
 */

#include "viz/gpu_solver.hpp"
#include "isf/grid.hpp"
#include "isf/velocity.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace viz {

namespace {

inline wgpu::ShaderModule create_module(const std::string& source)
{
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.chain.next = nullptr;
    wgslDesc.chain.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
    wgslDesc.code = source.c_str();

    wgpu::ShaderModuleDescriptor desc{};
    desc.nextInChain = &wgslDesc.chain;
    return wgfx::device.createShaderModule(desc);
}

inline wgpu::Buffer create_buffer(size_t size, decltype(wgpu::BufferDescriptor::usage) usage)
{
    wgpu::BufferDescriptor desc{};
    desc.size = size;
    desc.usage = usage;
    desc.mappedAtCreation = false;
    return wgfx::device.createBuffer(desc);
}

inline wgpu::BindGroupLayoutEntry storage_entry(uint32_t binding, bool readOnly)
{
    wgpu::BindGroupLayoutEntry e{};
    e.binding = binding;
    e.visibility = wgpu::ShaderStage::Compute;
    e.buffer.type = readOnly ? wgpu::BufferBindingType::ReadOnlyStorage : wgpu::BufferBindingType::Storage;
    e.buffer.hasDynamicOffset = false;
    e.buffer.minBindingSize = 0;
    return e;
}

inline wgpu::BindGroupLayoutEntry uniform_entry(uint32_t binding)
{
    wgpu::BindGroupLayoutEntry e{};
    e.binding = binding;
    e.visibility = wgpu::ShaderStage::Compute;
    e.buffer.type = wgpu::BufferBindingType::Uniform;
    e.buffer.hasDynamicOffset = false;
    e.buffer.minBindingSize = 0;
    return e;
}

inline wgpu::BindGroupEntry bg_buf_entry(uint32_t binding, wgpu::Buffer buf, size_t size)
{
    wgpu::BindGroupEntry e{};
    e.binding = binding;
    e.buffer = buf;
    e.offset = 0;
    e.size = size;
    return e;
}

inline wgpu::ComputePipeline create_pipe(wgpu::ShaderModule module, wgpu::BindGroupLayout bgl, const char* entryPoint = "main")
{
    wgpu::PipelineLayoutDescriptor plDesc{};
    plDesc.bindGroupLayoutCount = 1;
    plDesc.bindGroupLayouts = (WGPUBindGroupLayout*)&bgl;
    wgpu::PipelineLayout pl = wgfx::device.createPipelineLayout(plDesc);

    wgpu::ComputePipelineDescriptor cpDesc{};
    cpDesc.layout = pl;
    cpDesc.compute.module = module;
    cpDesc.compute.entryPoint = entryPoint;
    wgpu::ComputePipeline pipe = wgfx::device.createComputePipeline(cpDesc);
    pl.release();
    return pipe;
}

} // namespace

GpuFluidSolver::GpuFluidSolver(int grid_n, double domain_size)
    : grid_n_(grid_n), domain_size_(domain_size)
{
    init_buffers();
    init_pipelines();
}

GpuFluidSolver::~GpuFluidSolver()
{
    destroy_resources();
}

void GpuFluidSolver::destroy_resources()
{
    auto safe_release = [](auto& obj) {
        if (obj) {
            obj.release();
            obj = nullptr;
        }
    };

    safe_release(bg_fft_psi1_);
    safe_release(bg_fft_psi2_);
    safe_release(bg_fft_scratch_);
    safe_release(bg_propagate_psi1_);
    safe_release(bg_propagate_psi2_);
    safe_release(bg_normalize_);
    safe_release(bg_divergence_);
    safe_release(bg_poisson_);
    safe_release(bg_phase_vel_);

    safe_release(pipe_fft_x_fwd_);
    safe_release(pipe_fft_x_inv_);
    safe_release(pipe_fft_y_fwd_);
    safe_release(pipe_fft_y_inv_);
    safe_release(pipe_fft_z_fwd_);
    safe_release(pipe_fft_z_inv_);

    safe_release(pipe_propagate_);
    safe_release(pipe_normalize_);
    safe_release(pipe_divergence_);
    safe_release(pipe_poisson_);
    safe_release(pipe_phase_vel_);
    safe_release(pipe_advect_);

    safe_release(bgl_fft_);
    safe_release(bgl_propagate_);
    safe_release(bgl_normalize_);
    safe_release(bgl_divergence_);
    safe_release(bgl_poisson_);
    safe_release(bgl_phase_vel_);
    safe_release(bgl_advect_);

    safe_release(b_psi1_);
    safe_release(b_psi2_);
    safe_release(b_scratch_);
    safe_release(b_vel_);
    safe_release(u_sim_param_);
    safe_release(u_adv_param_);
}

void GpuFluidSolver::resize(int new_grid_n)
{
    if (new_grid_n == grid_n_) return;
    destroy_resources();
    grid_n_ = new_grid_n;
    init_buffers();
    init_pipelines();
}

void GpuFluidSolver::init_buffers()
{
    const size_t total_cells = static_cast<size_t>(grid_n_) * grid_n_ * grid_n_;
    const size_t cplx_bytes  = total_cells * sizeof(float) * 2; // vec2<f32>
    const size_t vel_bytes   = total_cells * sizeof(float) * 4; // vec4<f32>

    const auto storage_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;

    b_psi1_    = create_buffer(cplx_bytes, storage_usage);
    b_psi2_    = create_buffer(cplx_bytes, storage_usage);
    b_scratch_ = create_buffer(cplx_bytes, storage_usage);
    b_vel_     = create_buffer(vel_bytes, storage_usage);

    const auto unif_usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    u_sim_param_ = create_buffer(64, unif_usage);
    u_adv_param_ = create_buffer(64, unif_usage);
}

void GpuFluidSolver::init_pipelines()
{
    const int N      = grid_n_;
    const int half_n = N / 2;
    int log2_n = 0;
    while ((1 << log2_n) < N) log2_n++;

    const size_t total_cells = static_cast<size_t>(N) * N * N;
    const size_t cplx_bytes  = total_cells * sizeof(float) * 2;
    const size_t vel_bytes   = total_cells * sizeof(float) * 4;

    // -------------------------------------------------------------
    // 1. Stockham Radix-2 FFT Shaders (Stateless, 6 Entry Points)
    // -------------------------------------------------------------
    std::string fft_wgsl = std::format(R"wgsl(
@group(0) @binding(0) var<storage, read_write> s_data: array<vec2<f32>>;

const N: u32 = {0}u;
const HALF_N: u32 = {1}u;
const LOG2_N: u32 = {2}u;
const PI: f32 = 3.141592653589793;

var<workgroup> s0: array<vec2<f32>, {0}>;
var<workgroup> s1: array<vec2<f32>, {0}>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {{
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}}

fn run_stockham(t: u32, idx0: u32, idx1: u32, is_inv: bool) {{
    s0[t]          = s_data[idx0];
    s0[t + HALF_N] = s_data[idx1];
    workgroupBarrier();

    let sign = select(-1.0, 1.0, is_inv);

    for (var s = 0u; s < LOG2_N; s = s + 1u) {{
        let L = 1u << s;
        let j = t / L;
        let k = t % L;

        let angle = sign * 2.0 * PI * f32(k) / f32(2u * L);
        let twiddle = vec2<f32>(cos(angle), sin(angle));

        if ((s & 1u) == 0u) {{
            let u = s0[j * L + k];
            let v = s0[j * L + k + HALF_N];
            let w_v = complex_mul(twiddle, v);
            s1[2u * j * L + k]     = u + w_v;
            s1[2u * j * L + k + L] = u - w_v;
        }} else {{
            let u = s1[j * L + k];
            let v = s1[j * L + k + HALF_N];
            let w_v = complex_mul(twiddle, v);
            s0[2u * j * L + k]     = u + w_v;
            s0[2u * j * L + k + L] = u - w_v;
        }}
        workgroupBarrier();
    }}

    var res0: vec2<f32>;
    var res1: vec2<f32>;
    if ((LOG2_N & 1u) != 0u) {{
        res0 = s1[t];
        res1 = s1[t + HALF_N];
    }} else {{
        res0 = s0[t];
        res1 = s0[t + HALF_N];
    }}

    if (is_inv) {{
        res0 = res0 / f32(N);
        res1 = res1 / f32(N);
    }}

    s_data[idx0] = res0;
    s_data[idx1] = res1;
}}

@compute @workgroup_size({1}, 1, 1)
fn fft_x_fwd(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    let y = wid.x;
    let z = wid.y;
    run_stockham(t, t + N * (y + N * z), (t + HALF_N) + N * (y + N * z), false);
}}

@compute @workgroup_size({1}, 1, 1)
fn fft_x_inv(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    let y = wid.x;
    let z = wid.y;
    run_stockham(t, t + N * (y + N * z), (t + HALF_N) + N * (y + N * z), true);
}}

@compute @workgroup_size({1}, 1, 1)
fn fft_y_fwd(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    let x = wid.x;
    let z = wid.y;
    run_stockham(t, x + N * (t + N * z), x + N * ((t + HALF_N) + N * z), false);
}}

@compute @workgroup_size({1}, 1, 1)
fn fft_y_inv(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    let x = wid.x;
    let z = wid.y;
    run_stockham(t, x + N * (t + N * z), x + N * ((t + HALF_N) + N * z), true);
}}

@compute @workgroup_size({1}, 1, 1)
fn fft_z_fwd(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    let x = wid.x;
    let y = wid.y;
    run_stockham(t, x + N * (y + N * t), x + N * (y + N * (t + HALF_N)), false);
}}

@compute @workgroup_size({1}, 1, 1)
fn fft_z_inv(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let t = lid.x;
    let x = wid.x;
    let y = wid.y;
    run_stockham(t, x + N * (y + N * t), x + N * (y + N * (t + HALF_N)), true);
}}
)wgsl", N, half_n, log2_n);

    wgpu::ShaderModule mod_fft = create_module(fft_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_fft_entries = {
        storage_entry(0, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_fft_desc{};
    bgl_fft_desc.entryCount = bgl_fft_entries.size();
    bgl_fft_desc.entries = bgl_fft_entries.data();
    bgl_fft_ = wgfx::device.createBindGroupLayout(bgl_fft_desc);

    pipe_fft_x_fwd_ = create_pipe(mod_fft, bgl_fft_, "fft_x_fwd");
    pipe_fft_x_inv_ = create_pipe(mod_fft, bgl_fft_, "fft_x_inv");
    pipe_fft_y_fwd_ = create_pipe(mod_fft, bgl_fft_, "fft_y_fwd");
    pipe_fft_y_inv_ = create_pipe(mod_fft, bgl_fft_, "fft_y_inv");
    pipe_fft_z_fwd_ = create_pipe(mod_fft, bgl_fft_, "fft_z_fwd");
    pipe_fft_z_inv_ = create_pipe(mod_fft, bgl_fft_, "fft_z_inv");
    mod_fft.release();

    auto make_fft_bg = [&](wgpu::Buffer target_buf) {
        std::vector<wgpu::BindGroupEntry> entries = {
            bg_buf_entry(0, target_buf, cplx_bytes)
        };
        wgpu::BindGroupDescriptor bgDesc{};
        bgDesc.layout = bgl_fft_;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        return wgfx::device.createBindGroup(bgDesc);
    };
    bg_fft_psi1_    = make_fft_bg(b_psi1_);
    bg_fft_psi2_    = make_fft_bg(b_psi2_);
    bg_fft_scratch_ = make_fft_bg(b_scratch_);

    // -------------------------------------------------------------
    // Common SimParams struct in WGSL
    // -------------------------------------------------------------
    const char* kCommonHeader = R"wgsl(
struct SimParams {
    nx: u32, ny: u32, nz: u32, _pad0: u32,
    lx: f32, ly: f32, lz: f32, dt: f32,
    hbar: f32, _pad1: f32, _pad2: f32, _pad3: f32,
};
const PI: f32 = 3.141592653589793;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
fn cell_index(x: u32, y: u32, z: u32, nx: u32, ny: u32) -> u32 {
    return x + nx * (y + ny * z);
}
fn wrap_c(c: i32, n: u32) -> u32 {
    let un = i32(n);
    return u32((c % un + un) % un);
}
fn wrap_pos(v: f32, period: f32) -> f32 {
    let r = v - floor(v / period) * period;
    return select(r, 0.0, r >= period);
}
)wgsl";

    // -------------------------------------------------------------
    // 2. Propagate Shader (Algorithm 2 in Fourier Space)
    // -------------------------------------------------------------
    std::string prop_wgsl = std::string(kCommonHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<storage, read_write> s_psi: array<vec2<f32>>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kx = gid.x;
    let ky = gid.y;
    let kz = gid.z;
    if (kx >= u_sim.nx || ky >= u_sim.ny || kz >= u_sim.nz) { return; }

    let sx = select(f32(kx), f32(kx) - f32(u_sim.nx), kx > u_sim.nx / 2u);
    let sy = select(f32(ky), f32(ky) - f32(u_sim.ny), ky > u_sim.ny / 2u);
    let sz = select(f32(kz), f32(kz) - f32(u_sim.nz), kz > u_sim.nz / 2u);

    let fx = 2.0 * PI / u_sim.lx;
    let fy = 2.0 * PI / u_sim.ly;
    let fz = 2.0 * PI / u_sim.lz;

    let lambda = -((sx * fx) * (sx * fx) + (sy * fy) * (sy * fy) + (sz * fz) * (sz * fz));
    let phi = u_sim.hbar * lambda * u_sim.dt * 0.5;
    let rot = vec2<f32>(cos(phi), sin(phi));

    let idx = cell_index(kx, ky, kz, u_sim.nx, u_sim.ny);
    s_psi[idx] = complex_mul(s_psi[idx], rot);
}
)wgsl";

    wgpu::ShaderModule mod_prop = create_module(prop_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_prop_entries = {
        uniform_entry(0),
        storage_entry(1, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_prop_desc{};
    bgl_prop_desc.entryCount = bgl_prop_entries.size();
    bgl_prop_desc.entries = bgl_prop_entries.data();
    bgl_propagate_ = wgfx::device.createBindGroupLayout(bgl_prop_desc);
    pipe_propagate_ = create_pipe(mod_prop, bgl_propagate_);
    mod_prop.release();

    auto make_prop_bg = [&](wgpu::Buffer target_buf) {
        std::vector<wgpu::BindGroupEntry> entries = {
            bg_buf_entry(0, u_sim_param_, 64),
            bg_buf_entry(1, target_buf, cplx_bytes)
        };
        wgpu::BindGroupDescriptor bgDesc{};
        bgDesc.layout = bgl_propagate_;
        bgDesc.entryCount = entries.size();
        bgDesc.entries = entries.data();
        return wgfx::device.createBindGroup(bgDesc);
    };
    bg_propagate_psi1_ = make_prop_bg(b_psi1_);
    bg_propagate_psi2_ = make_prop_bg(b_psi2_);

    // -------------------------------------------------------------
    // 3. Normalize Shader
    // -------------------------------------------------------------
    std::string norm_wgsl = std::string(kCommonHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<storage, read_write> s_psi1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_psi2: array<vec2<f32>>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    if (i >= u_sim.nx || j >= u_sim.ny || k >= u_sim.nz) { return; }

    let idx = cell_index(i, j, k, u_sim.nx, u_sim.ny);
    let p1 = s_psi1[idx];
    let p2 = s_psi2[idx];
    let m_sq = (p1.x * p1.x + p1.y * p1.y) + (p2.x * p2.x + p2.y * p2.y);
    if (m_sq > 0.0) {
        let inv_m = 1.0 / sqrt(m_sq);
        s_psi1[idx] = p1 * inv_m;
        s_psi2[idx] = p2 * inv_m;
    }
}
)wgsl";

    wgpu::ShaderModule mod_norm = create_module(norm_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_norm_entries = {
        uniform_entry(0),
        storage_entry(1, /*readOnly=*/false),
        storage_entry(2, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_norm_desc{};
    bgl_norm_desc.entryCount = bgl_norm_entries.size();
    bgl_norm_desc.entries = bgl_norm_entries.data();
    bgl_normalize_ = wgfx::device.createBindGroupLayout(bgl_norm_desc);
    pipe_normalize_ = create_pipe(mod_norm, bgl_normalize_);
    mod_norm.release();

    std::vector<wgpu::BindGroupEntry> bg_norm_entries = {
        bg_buf_entry(0, u_sim_param_, 64),
        bg_buf_entry(1, b_psi1_, cplx_bytes),
        bg_buf_entry(2, b_psi2_, cplx_bytes)
    };
    wgpu::BindGroupDescriptor bg_norm_desc{};
    bg_norm_desc.layout = bgl_normalize_;
    bg_norm_desc.entryCount = bg_norm_entries.size();
    bg_norm_desc.entries = bg_norm_entries.data();
    bg_normalize_ = wgfx::device.createBindGroup(bg_norm_desc);

    // -------------------------------------------------------------
    // 4. Divergence Shader (Algorithm 3, Step 1 & 2)
    // -------------------------------------------------------------
    std::string div_wgsl = std::string(kCommonHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<storage, read> s_psi1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> s_psi2: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> s_scratch: array<vec2<f32>>;

fn edge_arg(v: u32, w: u32) -> f32 {
    let p1v = s_psi1[v];
    let p2v = s_psi2[v];
    let p1w = s_psi1[w];
    let p2w = s_psi2[w];

    // conj(p1v)*p1w + conj(p2v)*p2w
    let re = (p1v.x * p1w.x + p1v.y * p1w.y) + (p2v.x * p2w.x + p2v.y * p2w.y);
    let im = (p1v.x * p1w.y - p1v.y * p1w.x) + (p2v.x * p2w.y - p2v.y * p2w.x);
    return atan2(im, re);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    if (i >= u_sim.nx || j >= u_sim.ny || k >= u_sim.nz) { return; }

    let v = cell_index(i, j, k, u_sim.nx, u_sim.ny);

    let xp = cell_index(wrap_c(i32(i) + 1, u_sim.nx), j, k, u_sim.nx, u_sim.ny);
    let xm = cell_index(wrap_c(i32(i) - 1, u_sim.nx), j, k, u_sim.nx, u_sim.ny);
    let yp = cell_index(i, wrap_c(i32(j) + 1, u_sim.ny), k, u_sim.nx, u_sim.ny);
    let ym = cell_index(i, wrap_c(i32(j) - 1, u_sim.ny), k, u_sim.nx, u_sim.ny);
    let zp = cell_index(i, j, wrap_c(i32(k) + 1, u_sim.nz), u_sim.nx, u_sim.ny);
    let zm = cell_index(i, j, wrap_c(i32(k) - 1, u_sim.nz), u_sim.nx, u_sim.ny);

    let eta_x_v  = edge_arg(v, xp);
    let eta_x_xm = edge_arg(xm, v);
    let eta_y_v  = edge_arg(v, yp);
    let eta_y_ym = edge_arg(ym, v);
    let eta_z_v  = edge_arg(v, zp);
    let eta_z_zm = edge_arg(zm, v);

    let dx = u_sim.lx / f32(u_sim.nx);
    let dy = u_sim.ly / f32(u_sim.ny);
    let dz = u_sim.lz / f32(u_sim.nz);

    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let inv_dz2 = 1.0 / (dz * dz);

    let xi = (eta_x_v - eta_x_xm) * inv_dx2
           + (eta_y_v - eta_y_ym) * inv_dy2
           + (eta_z_v - eta_z_zm) * inv_dz2;

    s_scratch[v] = vec2<f32>(xi, 0.0);
}
)wgsl";

    wgpu::ShaderModule mod_div = create_module(div_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_div_entries = {
        uniform_entry(0),
        storage_entry(1, /*readOnly=*/true),
        storage_entry(2, /*readOnly=*/true),
        storage_entry(3, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_div_desc{};
    bgl_div_desc.entryCount = bgl_div_entries.size();
    bgl_div_desc.entries = bgl_div_entries.data();
    bgl_divergence_ = wgfx::device.createBindGroupLayout(bgl_div_desc);
    pipe_divergence_ = create_pipe(mod_div, bgl_divergence_);
    mod_div.release();

    std::vector<wgpu::BindGroupEntry> bg_div_entries = {
        bg_buf_entry(0, u_sim_param_, 64),
        bg_buf_entry(1, b_psi1_, cplx_bytes),
        bg_buf_entry(2, b_psi2_, cplx_bytes),
        bg_buf_entry(3, b_scratch_, cplx_bytes)
    };
    wgpu::BindGroupDescriptor bg_div_desc{};
    bg_div_desc.layout = bgl_divergence_;
    bg_div_desc.entryCount = bg_div_entries.size();
    bg_div_desc.entries = bg_div_entries.data();
    bg_divergence_ = wgfx::device.createBindGroup(bg_div_desc);

    // -------------------------------------------------------------
    // 5. Poisson Solve Shader (Eq. 17 in Fourier Space)
    // -------------------------------------------------------------
    std::string pois_wgsl = std::string(kCommonHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<storage, read_write> s_scratch: array<vec2<f32>>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    if (i >= u_sim.nx || j >= u_sim.ny || k >= u_sim.nz) { return; }

    let dx = u_sim.lx / f32(u_sim.nx);
    let dy = u_sim.ly / f32(u_sim.ny);
    let dz = u_sim.lz / f32(u_sim.nz);

    let inv_dx2 = 1.0 / (dx * dx);
    let inv_dy2 = 1.0 / (dy * dy);
    let inv_dz2 = 1.0 / (dz * dz);

    let sx = sin(PI * f32(i) / f32(u_sim.nx));
    let sy = sin(PI * f32(j) / f32(u_sim.ny));
    let sz = sin(PI * f32(k) / f32(u_sim.nz));

    let lambda = -4.0 * inv_dx2 * sx * sx
                 -4.0 * inv_dy2 * sy * sy
                 -4.0 * inv_dz2 * sz * sz;

    let idx = cell_index(i, j, k, u_sim.nx, u_sim.ny);
    if (lambda != 0.0) {
        s_scratch[idx] = s_scratch[idx] / lambda;
    } else {
        s_scratch[idx] = vec2<f32>(0.0, 0.0);
    }
}
)wgsl";

    wgpu::ShaderModule mod_pois = create_module(pois_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_pois_entries = {
        uniform_entry(0),
        storage_entry(1, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_pois_desc{};
    bgl_pois_desc.entryCount = bgl_pois_entries.size();
    bgl_pois_desc.entries = bgl_pois_entries.data();
    bgl_poisson_ = wgfx::device.createBindGroupLayout(bgl_pois_desc);
    pipe_poisson_ = create_pipe(mod_pois, bgl_poisson_);
    mod_pois.release();

    std::vector<wgpu::BindGroupEntry> bg_pois_entries = {
        bg_buf_entry(0, u_sim_param_, 64),
        bg_buf_entry(1, b_scratch_, cplx_bytes)
    };
    wgpu::BindGroupDescriptor bg_pois_desc{};
    bg_pois_desc.layout = bgl_poisson_;
    bg_pois_desc.entryCount = bg_pois_entries.size();
    bg_pois_desc.entries = bg_pois_entries.data();
    bg_poisson_ = wgfx::device.createBindGroup(bg_pois_desc);

    // -------------------------------------------------------------
    // 6. Phase Correction + Velocity Field Shader
    // -------------------------------------------------------------
    std::string pv_wgsl = std::string(kCommonHeader) + R"wgsl(
@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<storage, read_write> s_psi1: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_psi2: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> s_scratch: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> s_vel: array<vec4<f32>>;

fn edge_arg(v: u32, w: u32) -> f32 {
    let p1v = s_psi1[v];
    let p2v = s_psi2[v];
    let p1w = s_psi1[w];
    let p2w = s_psi2[w];

    let re = (p1v.x * p1w.x + p1v.y * p1w.y) + (p2v.x * p2w.x + p2v.y * p2w.y);
    let im = (p1v.x * p1w.y - p1v.y * p1w.x) + (p2v.x * p2w.y - p2v.y * p2w.x);
    return atan2(im, re);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let k = gid.z;
    if (i >= u_sim.nx || j >= u_sim.ny || k >= u_sim.nz) { return; }

    let v  = cell_index(i, j, k, u_sim.nx, u_sim.ny);
    let xp = cell_index(wrap_c(i32(i) + 1, u_sim.nx), j, k, u_sim.nx, u_sim.ny);
    let xm = cell_index(wrap_c(i32(i) - 1, u_sim.nx), j, k, u_sim.nx, u_sim.ny);
    let yp = cell_index(i, wrap_c(i32(j) + 1, u_sim.ny), k, u_sim.nx, u_sim.ny);
    let ym = cell_index(i, wrap_c(i32(j) - 1, u_sim.ny), k, u_sim.nx, u_sim.ny);
    let zp = cell_index(i, j, wrap_c(i32(k) + 1, u_sim.nz), u_sim.nx, u_sim.ny);
    let zm = cell_index(i, j, wrap_c(i32(k) - 1, u_sim.nz), u_sim.nx, u_sim.ny);

    // 1. Recover vertex velocity from cached uncorrected edge angles and q
    let eta_x_v  = edge_arg(v, xp);
    let eta_x_xm = edge_arg(xm, v);
    let eta_y_v  = edge_arg(v, yp);
    let eta_y_ym = edge_arg(ym, v);
    let eta_z_v  = edge_arg(v, zp);
    let eta_z_zm = edge_arg(zm, v);

    let q_xp = s_scratch[xp].x;
    let q_xm = s_scratch[xm].x;
    let q_yp = s_scratch[yp].x;
    let q_ym = s_scratch[ym].x;
    let q_zp = s_scratch[zp].x;
    let q_zm = s_scratch[zm].x;

    let dx = u_sim.lx / f32(u_sim.nx);
    let dy = u_sim.ly / f32(u_sim.ny);
    let dz = u_sim.lz / f32(u_sim.nz);

    let fx = u_sim.hbar / (2.0 * dx);
    let fy = u_sim.hbar / (2.0 * dy);
    let fz = u_sim.hbar / (2.0 * dz);

    let vx = fx * (eta_x_v + eta_x_xm - q_xp + q_xm);
    let vy = fy * (eta_y_v + eta_y_ym - q_yp + q_ym);
    let vz = fz * (eta_z_v + eta_z_zm - q_zp + q_zm);

    s_vel[v] = vec4<f32>(vx, vy, vz, 0.0);

    // 2. Pointwise phase correction ψ ← exp(-i · q) · ψ
    let q_v = s_scratch[v].x;
    let phi = -q_v;
    let rot = vec2<f32>(cos(phi), sin(phi));

    s_psi1[v] = complex_mul(s_psi1[v], rot);
    s_psi2[v] = complex_mul(s_psi2[v], rot);
}
)wgsl";

    wgpu::ShaderModule mod_pv = create_module(pv_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_pv_entries = {
        uniform_entry(0),
        storage_entry(1, /*readOnly=*/false),
        storage_entry(2, /*readOnly=*/false),
        storage_entry(3, /*readOnly=*/true),
        storage_entry(4, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_pv_desc{};
    bgl_pv_desc.entryCount = bgl_pv_entries.size();
    bgl_pv_desc.entries = bgl_pv_entries.data();
    bgl_phase_vel_ = wgfx::device.createBindGroupLayout(bgl_pv_desc);
    pipe_phase_vel_ = create_pipe(mod_pv, bgl_phase_vel_);
    mod_pv.release();

    std::vector<wgpu::BindGroupEntry> bg_pv_entries = {
        bg_buf_entry(0, u_sim_param_, 64),
        bg_buf_entry(1, b_psi1_, cplx_bytes),
        bg_buf_entry(2, b_psi2_, cplx_bytes),
        bg_buf_entry(3, b_scratch_, cplx_bytes),
        bg_buf_entry(4, b_vel_, vel_bytes)
    };
    wgpu::BindGroupDescriptor bg_pv_desc{};
    bg_pv_desc.layout = bgl_phase_vel_;
    bg_pv_desc.entryCount = bg_pv_entries.size();
    bg_pv_desc.entries = bg_pv_entries.data();
    bg_phase_vel_ = wgfx::device.createBindGroup(bg_pv_desc);

    // -------------------------------------------------------------
    // 7. Particle Advection Shader (RK2 Midpoint on GPU)
    // -------------------------------------------------------------
    std::string adv_wgsl = std::string(kCommonHeader) + R"wgsl(
struct AdvectParams {
    dt: f32,
    num_particles: u32,
    _pad0: u32,
    _pad1: u32,
};

struct ParticleData {
    pos: vec4<f32>,
    col: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_sim: SimParams;
@group(0) @binding(1) var<uniform> u_adv: AdvectParams;
@group(0) @binding(2) var<storage, read> s_vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> s_particles: array<ParticleData>;

fn sample_vel_trilinear(pos: vec3<f32>) -> vec3<f32> {
    let lx = u_sim.lx;
    let ly = u_sim.ly;
    let lz = u_sim.lz;

    // Periodic wrap into [0, L)
    let wx = wrap_pos(pos.x, lx);
    let wy = wrap_pos(pos.y, ly);
    let wz = wrap_pos(pos.z, lz);

    let dx = lx / f32(u_sim.nx);
    let dy = ly / f32(u_sim.ny);
    let dz = lz / f32(u_sim.nz);

    let fx = wx / dx;
    let fy = wy / dy;
    let fz = wz / dz;

    let i0 = u32(floor(fx)) % u_sim.nx;
    let j0 = u32(floor(fy)) % u_sim.ny;
    let k0 = u32(floor(fz)) % u_sim.nz;

    let i1 = (i0 + 1u) % u_sim.nx;
    let j1 = (j0 + 1u) % u_sim.ny;
    let k1 = (k0 + 1u) % u_sim.nz;

    let tx = fx - floor(fx);
    let ty = fy - floor(fy);
    let tz = fz - floor(fz);

    let c000 = s_vel[cell_index(i0, j0, k0, u_sim.nx, u_sim.ny)].xyz;
    let c100 = s_vel[cell_index(i1, j0, k0, u_sim.nx, u_sim.ny)].xyz;
    let c010 = s_vel[cell_index(i0, j1, k0, u_sim.nx, u_sim.ny)].xyz;
    let c110 = s_vel[cell_index(i1, j1, k0, u_sim.nx, u_sim.ny)].xyz;
    let c001 = s_vel[cell_index(i0, j0, k1, u_sim.nx, u_sim.ny)].xyz;
    let c101 = s_vel[cell_index(i1, j0, k1, u_sim.nx, u_sim.ny)].xyz;
    let c011 = s_vel[cell_index(i0, j1, k1, u_sim.nx, u_sim.ny)].xyz;
    let c111 = s_vel[cell_index(i1, j1, k1, u_sim.nx, u_sim.ny)].xyz;

    let c00 = mix(c000, c100, tx);
    let c10 = mix(c010, c110, tx);
    let c01 = mix(c001, c101, tx);
    let c11 = mix(c011, c111, tx);

    let c0 = mix(c00, c10, ty);
    let c1 = mix(c01, c11, ty);

    return mix(c0, c1, tz);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p_idx = gid.x;
    if (p_idx >= u_adv.num_particles) { return; }

    var p = s_particles[p_idx];
    let pos0 = p.pos.xyz;

    // RK2 midpoint step
    let v0 = sample_vel_trilinear(pos0);
    let pos_mid = pos0 + 0.5 * u_adv.dt * v0;
    let v1 = sample_vel_trilinear(pos_mid);
    var pos1 = pos0 + u_adv.dt * v1;

    // Wrap into periodic box [0, L)
    pos1.x = wrap_pos(pos1.x, u_sim.lx);
    pos1.y = wrap_pos(pos1.y, u_sim.ly);
    pos1.z = wrap_pos(pos1.z, u_sim.lz);

    p.pos = vec4<f32>(pos1, 1.0);
    s_particles[p_idx] = p;
}
)wgsl";

    wgpu::ShaderModule mod_adv = create_module(adv_wgsl);
    std::vector<wgpu::BindGroupLayoutEntry> bgl_adv_entries = {
        uniform_entry(0),
        uniform_entry(1),
        storage_entry(2, /*readOnly=*/true),
        storage_entry(3, /*readOnly=*/false)
    };
    wgpu::BindGroupLayoutDescriptor bgl_adv_desc{};
    bgl_adv_desc.entryCount = bgl_adv_entries.size();
    bgl_adv_desc.entries = bgl_adv_entries.data();
    bgl_advect_ = wgfx::device.createBindGroupLayout(bgl_adv_desc);
    pipe_advect_ = create_pipe(mod_adv, bgl_advect_);
    mod_adv.release();
}

void GpuFluidSolver::upload_grid(const isf::Grid3D& g)
{
    const int N = grid_n_;
    const size_t total_cells = static_cast<size_t>(N) * N * N;
    std::vector<float> p1_scratch(total_cells * 2);
    std::vector<float> p2_scratch(total_cells * 2);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(total_cells); ++i) {
        const auto c1 = g.psi1(static_cast<std::size_t>(i));
        const auto c2 = g.psi2(static_cast<std::size_t>(i));
        p1_scratch[2 * i + 0] = static_cast<float>(c1.real());
        p1_scratch[2 * i + 1] = static_cast<float>(c1.imag());
        p2_scratch[2 * i + 0] = static_cast<float>(c2.real());
        p2_scratch[2 * i + 1] = static_cast<float>(c2.imag());
    }

    wgfx::queue.writeBuffer(b_psi1_, 0, p1_scratch.data(), p1_scratch.size() * sizeof(float));
    wgfx::queue.writeBuffer(b_psi2_, 0, p2_scratch.data(), p2_scratch.size() * sizeof(float));
}

void GpuFluidSolver::upload_velocity(const isf::VelocityField& vf)
{
    const size_t total_cells = static_cast<size_t>(grid_n_) * grid_n_ * grid_n_;
    if (vf.size() != total_cells) return;
    std::vector<float> vel_scratch(total_cells * 4);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(total_cells); ++i) {
        vel_scratch[4 * i + 0] = static_cast<float>(vf.vx[i]);
        vel_scratch[4 * i + 1] = static_cast<float>(vf.vy[i]);
        vel_scratch[4 * i + 2] = static_cast<float>(vf.vz[i]);
        vel_scratch[4 * i + 3] = 0.0f;
    }
    wgfx::queue.writeBuffer(b_vel_, 0, vel_scratch.data(), vel_scratch.size() * sizeof(float));
}

void GpuFluidSolver::execute_fft_3d(wgpu::CommandEncoder& enc, wgpu::BindGroup bg, bool inverse)
{
    const uint32_t N = static_cast<uint32_t>(grid_n_);
    auto dispatch_pass = [&](wgpu::ComputePipeline pipe) {
        wgpu::ComputePassDescriptor cpDesc{};
        wgpu::ComputePassEncoder cpass = enc.beginComputePass(cpDesc);
        cpass.setPipeline(pipe);
        cpass.setBindGroup(0, bg, 0, nullptr);
        cpass.dispatchWorkgroups(N, N, 1);
        cpass.end();
        cpass.release();
    };

    if (!inverse) {
        dispatch_pass(pipe_fft_x_fwd_);
        dispatch_pass(pipe_fft_y_fwd_);
        dispatch_pass(pipe_fft_z_fwd_);
    } else {
        dispatch_pass(pipe_fft_x_inv_);
        dispatch_pass(pipe_fft_y_inv_);
        dispatch_pass(pipe_fft_z_inv_);
    }
}

void GpuFluidSolver::step(double dt, double hbar)
{
    const uint32_t N = static_cast<uint32_t>(grid_n_);
    const uint32_t num_wg3 = (N + 3) / 4;

    struct SimParams {
        uint32_t nx, ny, nz, pad0;
        float    lx, ly, lz, dt;
        float    hbar, pad1, pad2, pad3;
    };
    SimParams sim_params{
        N, N, N, 0,
        static_cast<float>(domain_size_),
        static_cast<float>(domain_size_),
        static_cast<float>(domain_size_),
        static_cast<float>(dt),
        static_cast<float>(hbar),
        0.0f, 0.0f, 0.0f
    };
    wgfx::queue.writeBuffer(u_sim_param_, 0, &sim_params, sizeof(sim_params));

    wgpu::CommandEncoderDescriptor encDesc{};
    encDesc.label = "ISF GPU Fluid Step Encoder";
    wgpu::CommandEncoder enc = wgfx::device.createCommandEncoder(encDesc);

    auto dispatch_3d = [&](wgpu::ComputePipeline pipe, wgpu::BindGroup bg) {
        wgpu::ComputePassDescriptor cpDesc{};
        wgpu::ComputePassEncoder cpass = enc.beginComputePass(cpDesc);
        cpass.setPipeline(pipe);
        cpass.setBindGroup(0, bg, 0, nullptr);
        cpass.dispatchWorkgroups(num_wg3, num_wg3, num_wg3);
        cpass.end();
        cpass.release();
    };

    // -------------------------------------------------------------
    // Step 1: Free-particle Schrödinger Step (Alg. 2)
    // -------------------------------------------------------------
    // ψ₁: forward FFT -> propagator phase -> inverse FFT
    execute_fft_3d(enc, bg_fft_psi1_, /*inverse=*/false);
    dispatch_3d(pipe_propagate_, bg_propagate_psi1_);
    execute_fft_3d(enc, bg_fft_psi1_, /*inverse=*/true);

    // ψ₂: forward FFT -> propagator phase -> inverse FFT
    execute_fft_3d(enc, bg_fft_psi2_, /*inverse=*/false);
    dispatch_3d(pipe_propagate_, bg_propagate_psi2_);
    execute_fft_3d(enc, bg_fft_psi2_, /*inverse=*/true);

    // -------------------------------------------------------------
    // Step 2: Normalize ψ (|ψ| ← 1)
    // -------------------------------------------------------------
    dispatch_3d(pipe_normalize_, bg_normalize_);

    // -------------------------------------------------------------
    // Step 3: Pressure Projection (Alg. 3)
    // -------------------------------------------------------------
    // Extract divergence into b_scratch_
    dispatch_3d(pipe_divergence_, bg_divergence_);

    // Poisson solve: forward FFT on ξ -> divide eigenvalues -> inverse FFT
    execute_fft_3d(enc, bg_fft_scratch_, /*inverse=*/false);
    dispatch_3d(pipe_poisson_, bg_poisson_);
    execute_fft_3d(enc, bg_fft_scratch_, /*inverse=*/true);

    // Step 4: Phase correction + velocity recovery
    dispatch_3d(pipe_phase_vel_, bg_phase_vel_);

    wgpu::CommandBuffer commands = enc.finish();
    enc.release();
    wgfx::queue.submit(1, &commands);
    commands.release();
}

void GpuFluidSolver::advect_particles(wgpu::Buffer particle_buffer, std::size_t particle_count, double dt)
{
    if (!particle_buffer || particle_count == 0) return;

    struct AdvectParams {
        float    dt;
        uint32_t num_particles;
        uint32_t pad0;
        uint32_t pad1;
    };
    AdvectParams adv_params{
        static_cast<float>(dt),
        static_cast<uint32_t>(particle_count),
        0, 0
    };
    wgfx::queue.writeBuffer(u_adv_param_, 0, &adv_params, sizeof(adv_params));

    const size_t vel_bytes = static_cast<size_t>(grid_n_) * grid_n_ * grid_n_ * sizeof(float) * 4;
    const size_t part_bytes = particle_count * sizeof(float) * 8; // 32 bytes per GpuParticle

    std::vector<wgpu::BindGroupEntry> entries = {
        bg_buf_entry(0, u_sim_param_, 64),
        bg_buf_entry(1, u_adv_param_, 64),
        bg_buf_entry(2, b_vel_, vel_bytes),
        bg_buf_entry(3, particle_buffer, part_bytes)
    };
    wgpu::BindGroupDescriptor bgDesc{};
    bgDesc.layout = bgl_advect_;
    bgDesc.entryCount = entries.size();
    bgDesc.entries = entries.data();
    wgpu::BindGroup bg_adv = wgfx::device.createBindGroup(bgDesc);

    wgpu::CommandEncoderDescriptor encDesc{};
    encDesc.label = "ISF GPU Particle Advect Encoder";
    wgpu::CommandEncoder enc = wgfx::device.createCommandEncoder(encDesc);

    wgpu::ComputePassDescriptor cpDesc{};
    wgpu::ComputePassEncoder cpass = enc.beginComputePass(cpDesc);
    cpass.setPipeline(pipe_advect_);
    cpass.setBindGroup(0, bg_adv, 0, nullptr);
    const uint32_t num_wg = static_cast<uint32_t>((particle_count + 255) / 256);
    cpass.dispatchWorkgroups(num_wg, 1, 1);
    cpass.end();
    cpass.release();

    wgpu::CommandBuffer commands = enc.finish();
    enc.release();
    wgfx::queue.submit(1, &commands);
    commands.release();
    bg_adv.release();
}

} // namespace viz
