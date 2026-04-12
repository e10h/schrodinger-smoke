/**
 * @file shader.cpp
 * @brief Implementation of the vertex/fragment shader compiler.
 */

#include "viz/shader.hpp"

#include <cstdio>
#include <vector>

namespace viz {

namespace {

/**
 * @brief Compile a single GLSL stage (vertex or fragment).
 * @return Shader object id, or 0 on failure.
 */
GLuint compile_stage(GLenum stage, const char* source)
{
    const GLuint shader = glCreateShader(stage);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<std::size_t>(len) + 1, '\0');
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::fprintf(stderr, "Shader compile error (stage %u):\n%s\n",
                     static_cast<unsigned>(stage), log.data());
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

} // namespace

GLuint compile_shader_program(const char* vertex_source,
                              const char* fragment_source)
{
    const GLuint vs = compile_stage(GL_VERTEX_SHADER,   vertex_source);
    if (!vs) return 0;
    const GLuint fs = compile_stage(GL_FRAGMENT_SHADER, fragment_source);
    if (!fs) { glDeleteShader(vs); return 0; }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    // Detach + delete shaders immediately: the program object owns them now.
    glDetachShader(program, vs);
    glDetachShader(program, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(static_cast<std::size_t>(len) + 1, '\0');
        glGetProgramInfoLog(program, len, nullptr, log.data());
        std::fprintf(stderr, "Program link error:\n%s\n", log.data());
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

} // namespace viz
