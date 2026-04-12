/**
 * @file gif_recorder.hpp
 * @brief Minimal GIF recorder for the viewer — capture the simulation
 *        to an animated GIF without the ImGui overlay.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace viz {

/**
 * @brief Captures rendered frames to an animated GIF.
 *
 * Usage: call `capture_frame()` every frame between the scene draw
 * and the ImGui overlay.  The recorder internally skips frames (every
 * 3rd by default), reads back the GL framebuffer, and accumulates
 * GIF frames via msf_gif.  On `stop()` the GIF is flushed to disk.
 */
class GifRecorder {
public:
    GifRecorder() = default;
    ~GifRecorder();

    GifRecorder(const GifRecorder&) = delete;
    GifRecorder& operator=(const GifRecorder&) = delete;

    /// @brief True while a recording session is active.
    [[nodiscard]] bool is_recording() const noexcept { return recording_; }

    /**
     * @brief Begin a new recording session.
     *
     * @param framebuffer_width  GL framebuffer width in pixels (Retina-aware).
     * @param framebuffer_height GL framebuffer height in pixels.
     */
    void start(int framebuffer_width, int framebuffer_height);

    /**
     * @brief Offer the current frame for capture.
     *
     * Call after `renderer.draw()` but before `ImGui::Render()` so the
     * GIF shows the simulation without the UI overlay.  Internally
     * skips frames according to the frame-skip setting.
     */
    void capture_frame();

    /**
     * @brief Stop recording and write the GIF to disk.
     * @return Path of the written file (e.g. "recording_001.gif"),
     *         or empty string on failure.
     */
    std::string stop();

    /// @brief Draw the ImGui recorder controls (Record/Stop button).
    void imgui_panel();

private:
    bool recording_    = false;
    int  frame_counter_ = 0;
    int  file_counter_  = 0;
    int  gif_frames_    = 0;

    int  fb_width_  = 0;
    int  fb_height_ = 0;
    int  out_width_ = 320;
    int  out_height_= 180;
    int  frame_skip_= 3;       ///< record every Nth rendered frame

    std::vector<uint8_t> fb_pixels_;

    void* gif_ = nullptr;  // opaque MsfGifState*

    std::string last_file_;

    void readback_and_encode();
};

} // namespace viz
