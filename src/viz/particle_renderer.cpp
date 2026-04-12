/**
 * @file particle_renderer.cpp
 * @brief Implementation of the point-sprite particle renderer.
 *
 * Shader pipeline:
 *
 *   Vertex — transforms a_position by u_view_proj and sets a
 *   perspective-attenuated gl_PointSize (`u_point_size / clip.w`).
 *   Forwards a_color through to the fragment stage as v_color.
 *
 *   Fragment — discards outside the unit disk of gl_PointCoord and
 *   emits a soft radial falloff coloured by v_color, with additive
 *   alpha. Combined with additive blending this produces a pleasing
 *   volumetric-smoke look on a dark background.
 *
 * Per-particle colour is uploaded as a separate `vbo_col_` so the
 * advection code (which has no notion of colour) and the renderer's
 * colour buffer remain decoupled.
 */

#include "viz/particle_renderer.hpp"

#include "viz/particles.hpp"
#include "viz/shader.hpp"

#include <glm/gtc/type_ptr.hpp>

namespace viz {

namespace {

constexpr const char* kVertexShader = R"glsl(
#version 410 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4  u_view_proj;
uniform float u_point_size;

out vec3 v_color;

void main() {
    vec4 clip = u_view_proj * vec4(a_position, 1.0);
    gl_Position  = clip;
    // Attenuate with perspective depth so near particles look larger.
    gl_PointSize = u_point_size / max(clip.w, 0.1);
    v_color      = a_color;
}
)glsl";

constexpr const char* kFragmentShader = R"glsl(
#version 410 core
in  vec3 v_color;
out vec4 out_color;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = length(d);
    if (r > 0.5) discard;
    // Gaussian-ish radial falloff.
    float a = smoothstep(0.5, 0.0, r);
    out_color = vec4(v_color, a * 0.22);
}
)glsl";

} // namespace

ParticleRenderer::ParticleRenderer()
{
    program_   = compile_shader_program(kVertexShader, kFragmentShader);
    loc_mvp_   = glGetUniformLocation(program_, "u_view_proj");
    loc_psize_ = glGetUniformLocation(program_, "u_point_size");

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_pos_);
    glGenBuffers(1, &vbo_col_);

    glBindVertexArray(vao_);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
    glVertexAttribPointer(/*index=*/0, /*size=*/3, GL_FLOAT, GL_FALSE,
                          /*stride=*/3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);
    glVertexAttribPointer(/*index=*/1, /*size=*/3, GL_FLOAT, GL_FALSE,
                          /*stride=*/3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

ParticleRenderer::~ParticleRenderer()
{
    if (vbo_col_) glDeleteBuffers(1, &vbo_col_);
    if (vbo_pos_) glDeleteBuffers(1, &vbo_pos_);
    if (vao_)     glDeleteVertexArrays(1, &vao_);
    if (program_) glDeleteProgram(program_);
}

void ParticleRenderer::upload(const ParticleSystem& ps)
{
    const std::size_t n = ps.size();

    // Pack positions: SoA → interleaved float3.
    scratch_pos_.resize(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        scratch_pos_[3 * i + 0] = static_cast<float>(ps.x[i]);
        scratch_pos_[3 * i + 1] = static_cast<float>(ps.y[i]);
        scratch_pos_[3 * i + 2] = static_cast<float>(ps.z[i]);
    }

    // Pack colours: SoA → interleaved float3, defaulting to soft white
    // when the particle system has no per-particle colour.
    scratch_col_.resize(3 * n);
    if (ps.has_colors()) {
        for (std::size_t i = 0; i < n; ++i) {
            scratch_col_[3 * i + 0] = ps.r[i];
            scratch_col_[3 * i + 1] = ps.g[i];
            scratch_col_[3 * i + 2] = ps.b[i];
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            scratch_col_[3 * i + 0] = 0.85f;
            scratch_col_[3 * i + 1] = 0.90f;
            scratch_col_[3 * i + 2] = 1.00f;
        }
    }
    count_ = static_cast<GLsizei>(n);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(scratch_pos_.size() * sizeof(float)),
                 scratch_pos_.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(scratch_col_.size() * sizeof(float)),
                 scratch_col_.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ParticleRenderer::draw(const glm::mat4& view_proj, float point_size)
{
    glUseProgram(program_);
    glUniformMatrix4fv(loc_mvp_, 1, GL_FALSE, glm::value_ptr(view_proj));
    glUniform1f(loc_psize_, point_size);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, count_);
    glBindVertexArray(0);

    glUseProgram(0);
}

} // namespace viz
