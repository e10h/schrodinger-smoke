cmake_minimum_required(VERSION 3.20)
project(schroedinger
    VERSION 0.1.0
    DESCRIPTION "Schroedinger's Smoke — incompressible fluid simulation via the Schrödinger equation"
    LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the build type." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
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

# Catch2 — test framework
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4)
FetchContent_MakeAvailable(Catch2)

# OpenMP
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

# GLM — header-only math library
find_package(glm QUIET)
if(NOT glm_FOUND AND NOT TARGET glm::glm)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG 1.0.1)
    FetchContent_MakeAvailable(glm)
    set(glm_FOUND TRUE)
endif()

# SDL3 (system or Homebrew)
find_package(SDL3 QUIET)
if(NOT SDL3_FOUND)
    if(APPLE AND EXISTS "/opt/homebrew/opt/sdl3")
        set(SDL3_ROOT /opt/homebrew/opt/sdl3)
        find_package(SDL3 REQUIRED)
    else()
        message(FATAL_ERROR "SDL3 not found. Please install SDL3 via Homebrew ('brew install sdl3').")
    endif()
endif()

# WebGPU graphics library (wgfx)
add_subdirectory(third_party/wgfx)

# Dear ImGui
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9)
FetchContent_MakeAvailable(imgui)

set(IMGUI_SRC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    src/viz/imgui_impl_wgpu.cpp
)

add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${CMAKE_SOURCE_DIR}/third_party/wgfx/src
)
target_link_libraries(imgui PUBLIC SDL3::SDL3 webgpu)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_WEBGPU_BACKEND_WGPU)

if(APPLE)
    set_source_files_properties(
        src/viz/imgui_impl_wgpu.cpp
        PROPERTIES COMPILE_FLAGS "-x objective-c++"
    )
endif()

# ------------------------------------------------------------------
# ISF core library
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
# Interactive viewer application (WebGPU + SDL3)
# ------------------------------------------------------------------

add_executable(schroedinger_viewer
    src/main.cpp
    src/viz/window.cpp
    src/viz/particle_renderer.cpp
    src/viz/gpu_solver.cpp
    src/viz/gif_recorder.cpp
)
target_include_directories(schroedinger_viewer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/wgfx/src
    ${CMAKE_SOURCE_DIR}/third_party/wgfx/deps/sdl2webgpu
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(schroedinger_viewer PRIVATE
    viz isf wgfx sdl2webgpu SDL3::SDL3 glm::glm imgui webgpu
)
set_target_properties(schroedinger_viewer PROPERTIES
    CXX_STANDARD 20
)

target_copy_webgpu_binaries(schroedinger_viewer)

# ------------------------------------------------------------------
# CPU benchmark
# ------------------------------------------------------------------

add_executable(bench_solver src/bench_solver.cpp)
target_link_libraries(bench_solver PRIVATE viz isf)

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

enable_testing()
add_subdirectory(tests)
