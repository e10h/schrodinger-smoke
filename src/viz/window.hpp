/**
 * @file window.hpp
 * @brief GLFW window + OpenGL context + Dear ImGui setup.
 *
 * Window is an RAII wrapper that does everything needed to get an
 * OpenGL 4.1 core-profile window running on macOS with Dear ImGui
 * overlays and a minimal mouse-input abstraction for camera control.
 *
 *     Window win(1280, 720, "…");
 *     while (!win.should_close()) {
 *         win.poll_events();
 *         win.begin_frame();       // glClear + ImGui::NewFrame
 *         // … draw, ImGui widgets, …
 *         win.end_frame();         // ImGui::Render + SwapBuffers
 *     }
 *
 * The class is non-copyable; at most one Window should exist at a
 * time because it owns the shared GLFW process-level state.
 */

#pragma once

#include "viz/gl_includes.hpp"

struct GLFWwindow;

namespace viz {

/**
 * @brief GLFW window + GL context + Dear ImGui backend wrapper.
 */
class Window {
public:
    /** @brief Per-frame mouse input deltas. */
    struct Input {
        float mouse_dx        = 0.0f;  ///< delta X this frame (pixels)
        float mouse_dy        = 0.0f;  ///< delta Y this frame
        float scroll_dy       = 0.0f;  ///< accumulated scroll since last poll
        bool  left_mouse_down = false; ///< true while LMB is pressed
        bool  imgui_captures  = false; ///< ImGui wants exclusive mouse
    };

    /**
     * @brief Construct and create the GLFW window + GL context.
     * @throws std::runtime_error if GLFW or GL initialization fails.
     */
    Window(int width, int height, const char* title);

    /** @brief Destroy the ImGui backends and the GLFW window. */
    ~Window();

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool should_close() const noexcept;
    [[nodiscard]] int  width()  const noexcept;
    [[nodiscard]] int  height() const noexcept;

    /** @brief Get the latest framebuffer size (may differ from window size
     *         on Retina displays). */
    void framebuffer_size(int& w, int& h) const noexcept;

    /** @brief Poll window events and refresh the Input delta snapshot. */
    void poll_events();

    /**
     * @brief Start a new frame: clear color/depth buffers and begin
     *        a Dear ImGui frame.
     */
    void begin_frame();

    /**
     * @brief Finish a frame: render ImGui draw data and swap buffers.
     */
    void end_frame();

    /** @brief Per-frame input delta produced by the last poll_events(). */
    [[nodiscard]] const Input& input() const noexcept { return frame_input_; }

    /** @brief Raw GLFWwindow handle (escape hatch). */
    [[nodiscard]] GLFWwindow* native() const noexcept { return window_; }

private:
    GLFWwindow* window_        = nullptr;
    double      prev_mouse_x_  = 0.0;
    double      prev_mouse_y_  = 0.0;
    float       scroll_accum_  = 0.0f;
    Input       frame_input_{};

    // Static GLFW callback shim — reads the Window* from the user-pointer.
    static void scroll_callback(GLFWwindow* w, double xoff, double yoff);
};

} // namespace viz
