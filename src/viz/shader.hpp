/**
 * @file shader.hpp
 * @brief Tiny helper for compiling and linking a GLSL vertex+fragment
 *        shader pair into an OpenGL program object.
 *
 * This is intentionally minimal: no hot-reload, no file I/O, no
 * geometry/tessellation stages. The program source strings are
 * embedded at compile time, which is exactly what we want for a
 * small viewer.
 */

#pragma once

#include "viz/gl_includes.hpp"

#include <string>

namespace viz {

/**
 * @brief Compile a GLSL vertex + fragment shader pair and link them.
 *
 * On any compilation or linking error the info log is written to
 * stderr and the returned id is 0. Callers should check for 0 and
 * treat it as a fatal configuration error — this path is only taken
 * when a developer typo lands in the embedded shader source.
 *
 * @param vertex_source   GLSL source of the vertex shader.
 * @param fragment_source GLSL source of the fragment shader.
 * @return Linked program id, or 0 on failure.
 */
[[nodiscard]] GLuint
compile_shader_program(const char* vertex_source,
                       const char* fragment_source);

} // namespace viz
