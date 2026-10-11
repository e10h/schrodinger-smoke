/**
 * @file window.hpp
 * @brief SDL3 window + WebGPU surface + Dear ImGui setup.
 *
 * Window is an RAII wrapper around SDL3 and WebGPU (via wgfx) providing
 * frame lifecycle, window events, Dear ImGui overlays, and camera input.
 */

#pragma once

#include <SDL3/SDL.h>
#include <wgfx.h>

namespace viz {

/**
 * @brief SDL3 window + WebGPU surface + Dear ImGui backend wrapper.
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

    Window(int width, int height, const char* title);
    ~Window();

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool should_close() const noexcept { return should_close_; }
    [[nodiscard]] int  width()  const noexcept;
    [[nodiscard]] int  height() const noexcept;

    /** @brief Get the latest framebuffer size (Retina-aware). */
    void framebuffer_size(int& w, int& h) const noexcept;

    /** @brief Poll window events and refresh the Input delta snapshot. */
    void poll_events();

    /**
     * @brief Start a new frame: touch the color attachment, start the
     *        command encoder, and begin Dear ImGui frame.
     */
    void begin_frame();

    /**
     * @brief Finish a frame: render ImGui in its own pass and submit via wgfx::frame().
     */
    void end_frame();

    /** @brief Per-frame input delta produced by the last poll_events(). */
    [[nodiscard]] const Input& input() const noexcept { return frame_input_; }

    /** @brief Raw SDL_Window handle. */
    [[nodiscard]] SDL_Window* native() const noexcept { return window_; }

    /** @brief Access to main scene render pass and surface color texture. */
    [[nodiscard]] wgfx::RenderPass* render_pass() const noexcept { return scene_pass_; }
    [[nodiscard]] wgfx::ColorTexture* color_target() const noexcept { return color_tex_; }

private:
    SDL_Window*         window_       = nullptr;
    bool                should_close_ = false;
    float               prev_mouse_x_ = 0.0f;
    float               prev_mouse_y_ = 0.0f;
    float               scroll_accum_ = 0.0f;
    Input               frame_input_{};

    wgfx::ColorTexture* color_tex_    = nullptr;
    wgfx::RenderPass*   scene_pass_   = nullptr;
    wgfx::RenderPass*   ui_pass_      = nullptr;
};

} // namespace viz
