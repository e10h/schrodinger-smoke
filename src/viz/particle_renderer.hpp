/**
 * @file particle_renderer.hpp
 * @brief OpenGL renderer for the tracer-particle cloud.
 *
 * Draws a ParticleSystem as additively-blended soft point sprites.
 * Ownership is RAII:
 *   - ctor: compiles the shader program, creates VAO and VBO,
 *   - dtor: deletes all GL objects.
 *
 * `upload()` copies the current particle positions into the VBO
 * (GL_DYNAMIC_DRAW). `draw()` binds the program, sets uniforms,
 * and issues one `glDrawArrays(GL_POINTS, …)`.
 */

#pragma once

#include "viz/gl_includes.hpp"

#include <glm/mat4x4.hpp>

#include <vector>

namespace viz {

struct ParticleSystem;

/**
 * @brief Small RAII class that draws a ParticleSystem as point sprites.
 */
class ParticleRenderer {
public:
    /**
     * @brief Construct and initialise GL resources. A valid OpenGL
     *        context must be current when this runs.
     */
    ParticleRenderer();

    /** @brief Release all owned GL resources. */
    ~ParticleRenderer();

    ParticleRenderer(const ParticleRenderer&)            = delete;
    ParticleRenderer& operator=(const ParticleRenderer&) = delete;

    /**
     * @brief Upload the current particle positions to the VBO.
     *
     * Repacks the SoA position arrays into an interleaved float3 layout
     * in a reusable host-side scratch buffer, then copies it to the
     * GPU with glBufferData (orphan-and-replace for simplicity).
     */
    void upload(const ParticleSystem& ps);

    /**
     * @brief Draw the most recently uploaded particle cloud.
     *
     * @param view_proj  Combined view · projection matrix.
     * @param point_size Base point-sprite size in screen pixels,
     *                   attenuated by perspective depth in the shader.
     */
    void draw(const glm::mat4& view_proj, float point_size);

private:
    GLuint vao_         = 0;
    GLuint vbo_pos_     = 0;
    GLuint vbo_col_     = 0;
    GLuint program_     = 0;
    GLint  loc_mvp_     = -1;
    GLint  loc_psize_   = -1;
    GLsizei count_      = 0;

    std::vector<float> scratch_pos_;  ///< host-side interleave buffer (xyz)
    std::vector<float> scratch_col_;  ///< host-side interleave buffer (rgb)
};

} // namespace viz
