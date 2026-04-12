/**
 * @file window.cpp
 * @brief GLFW window + ImGui backend implementation.
 */

#include "viz/window.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include <cstdio>
#include <stdexcept>

namespace viz {

namespace {

/// @brief GLFW fatal-error callback — prints what we can and keeps going.
void glfw_error_callback(int error, const char* description)
{
    std::fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

} // namespace

Window::Window(int width, int height, const char* title)
{
    glfwSetErrorCallback(&glfw_error_callback);
    if (!glfwInit()) {
        throw std::runtime_error("glfwInit failed");
    }

    // Request OpenGL 4.1 core profile — the highest macOS supports.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }

    glfwSetWindowUserPointer(window_, this);
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);   // vsync

    glfwSetScrollCallback(window_, &Window::scroll_callback);
    glfwGetCursorPos(window_, &prev_mouse_x_, &prev_mouse_y_);

    // Dear ImGui init.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window_, /*install_callbacks=*/true);
    ImGui_ImplOpenGL3_Init("#version 410");

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);   // additive blending
    glDisable(GL_DEPTH_TEST);            // particles don't need depth test
}

Window::~Window()
{
    if (window_) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

bool Window::should_close() const noexcept
{
    return glfwWindowShouldClose(window_);
}

int Window::width() const noexcept
{
    int w = 0, h = 0;
    glfwGetWindowSize(window_, &w, &h);
    return w;
}

int Window::height() const noexcept
{
    int w = 0, h = 0;
    glfwGetWindowSize(window_, &w, &h);
    return h;
}

void Window::framebuffer_size(int& w, int& h) const noexcept
{
    glfwGetFramebufferSize(window_, &w, &h);
}

void Window::poll_events()
{
    glfwPollEvents();

    // Mouse delta.
    double mx = 0.0, my = 0.0;
    glfwGetCursorPos(window_, &mx, &my);
    frame_input_.mouse_dx = static_cast<float>(mx - prev_mouse_x_);
    frame_input_.mouse_dy = static_cast<float>(my - prev_mouse_y_);
    prev_mouse_x_ = mx;
    prev_mouse_y_ = my;

    frame_input_.left_mouse_down =
        (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);

    frame_input_.scroll_dy = scroll_accum_;
    scroll_accum_          = 0.0f;

    frame_input_.imgui_captures = ImGui::GetIO().WantCaptureMouse;
}

void Window::begin_frame()
{
    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window_, &fbw, &fbh);
    glViewport(0, 0, fbw, fbh);
    glClearColor(0.02f, 0.03f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Window::end_frame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
}

void Window::scroll_callback(GLFWwindow* w, double /*xoff*/, double yoff)
{
    // IMPORTANT: do NOT call ImGui_ImplGlfw_ScrollCallback from here.
    //
    // We installed this callback BEFORE ImGui_ImplGlfw_InitForOpenGL,
    // so ImGui chains us as its "previous" callback. The actual call
    // path on every scroll event is therefore:
    //
    //   GLFW → ImGui's scroll cb → (chain) → this function
    //
    // Forwarding back to ImGui_ImplGlfw_ScrollCallback would re-enter
    // ImGui, which would re-invoke its previous callback (us), and so
    // on — infinite recursion → stack overflow → SIGSEGV. ImGui has
    // already processed the event by the time we get here, so all we
    // need to do is forward the wheel delta to the camera as long as
    // ImGui isn't claiming the mouse for one of its widgets.
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return;

    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    if (self) {
        self->scroll_accum_ += static_cast<float>(yoff);
    }
}

} // namespace viz
