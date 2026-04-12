cmake_minimum_required(VERSION 3.20)
project(schroedinger
    VERSION 0.1.0
    DESCRIPTION "Schroedinger's Smoke — incompressible fluid simulation via the Schrödinger equation"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if the user didn't pick a build type. Without this
# CMake's "no build type" mode compiles everything with -O0, which makes
# the solver ~10x slower than necessary.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the build type." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# Generate compile_commands.json for clangd / IDEs.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

    # Aggressive native-arch tuning for the solver hot path. -ffast-math
    # is intentionally NOT enabled here: pocketfft and the pressure
    # projection rely on IEEE-754 semantics (NaN handling, signed zero,
    # subnormals at the FFT boundaries), and the speed-up vs. plain -O3
    # on this workload is small.
    add_compile_options($<$<CONFIG:Release>:-O3>
                        $<$<CONFIG:Release>:-march=native>
                        $<$<CONFIG:Release>:-DNDEBUG>)
endif()

# ------------------------------------------------------------------
# Third-party dependencies
# ------------------------------------------------------------------

# pocketfft — header-only, vendored in third_party/.
add_library(pocketfft INTERFACE)
target_include_directories(pocketfft INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party)

# Catch2 — test framework, fetched on first configure.
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4)
FetchContent_MakeAvailable(Catch2)

# OpenMP — used to parallelise the per-vertex inner loops in the
# solver. CMake's first-class FindOpenMP works with the modern
# Homebrew libomp + Apple Clang combination once the right hint
# variables are set.
if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT          /opt/homebrew/opt/libomp)
        set(OpenMP_CXX_FLAGS     "-Xclang -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY   /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
endif()
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found — solver will run single-threaded")
endif()

# GLFW — windowing and input. Expected to be installed system-wide
# (e.g. via Homebrew). We use find_package rather than FetchContent
# to keep build times reasonable.
find_package(glfw3 QUIET)

# GLM — header-only math library. Prefer the system package, fall
# back to FetchContent only if the user has no system install.
find_package(glm QUIET)

# Dear ImGui — no upstream CMake, so we fetch the sources and build
# a small static library from the subset of files we need.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.8)
FetchContent_MakeAvailable(imgui)

if(glfw3_FOUND AND glm_FOUND)
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw)
endif()

# ------------------------------------------------------------------
# ISF core library
#
# Sources are added incrementally via target_sources() as each TDD
# cycle introduces a new component.
# ------------------------------------------------------------------

add_library(isf STATIC
    src/isf/grid.cpp
    src/isf/fft.cpp
    src/isf/schrodinger.cpp
    src/isf/normalize.cpp
    src/isf/pressure.cpp
    src/isf/constraint.cpp
    src/isf/init.cpp
    src/isf/velocity.cpp
    src/isf/solver.cpp
)
target_include_directories(isf PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(isf PUBLIC pocketfft)
if(OpenMP_CXX_FOUND)
    target_link_libraries(isf PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(isf PUBLIC ISF_HAVE_OPENMP=1)
endif()

# ------------------------------------------------------------------
# Visualization / output library
#
# Pure CPU for now: tracer particles and advection. The OpenGL viewer
# in Phase 2 will add its own window/renderer sources alongside these.
# ------------------------------------------------------------------

add_library(viz STATIC
    src/viz/particles.cpp
    src/viz/camera.cpp
)
target_include_directories(viz PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(viz PUBLIC isf)
if(glm_FOUND)
    target_link_libraries(viz PUBLIC glm::glm)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(viz PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------------------------------------------
# Interactive viewer application
#
# Only built if the system provides GLFW + GLM. Otherwise the core
# library and tests still build cleanly.
# ------------------------------------------------------------------

if(glfw3_FOUND AND glm_FOUND)
    add_executable(schroedinger_viewer
        src/main.cpp
        src/viz/window.cpp
        src/viz/shader.cpp
        src/viz/particle_renderer.cpp
        src/viz/gif_recorder.cpp
    )
    target_link_libraries(schroedinger_viewer PRIVATE
        viz isf glfw glm::glm imgui "-framework OpenGL")
    set_target_properties(schroedinger_viewer PROPERTIES
        CXX_STANDARD 20)
endif()

# ------------------------------------------------------------------
# CPU benchmark — headless, no GL, no ImGui. Used to track
# performance regressions and improvements as we optimize.
# ------------------------------------------------------------------

add_executable(bench_solver src/bench_solver.cpp)
target_link_libraries(bench_solver PRIVATE viz isf)

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

enable_testing()
add_subdirectory(tests)
