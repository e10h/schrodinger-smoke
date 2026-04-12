/**
 * @file gl_includes.hpp
 * @brief Single point of truth for OpenGL headers.
 *
 * macOS ships OpenGL up to 4.1 core in its `OpenGL.framework`; no GL
 * loader library is needed. On other platforms we would include a
 * loader like glad here; keeping the include in one place makes that
 * switch trivial later.
 */

#pragma once

#if defined(__APPLE__)
#   define GL_SILENCE_DEPRECATION 1
#   include <OpenGL/gl3.h>
#else
#   error "Only macOS is supported right now; add a GL loader for your platform."
#endif
