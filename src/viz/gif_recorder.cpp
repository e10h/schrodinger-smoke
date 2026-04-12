/**
 * @file gif_recorder.cpp
 * @brief GIF recorder implementation using msf_gif.
 */

#define MSF_GIF_IMPL
#include "msf_gif.h"

#include "viz/gif_recorder.hpp"
#include "viz/gl_includes.hpp"

#include <imgui.h>

#include <cstdio>
#include <cstring>

namespace viz {

namespace {
MsfGifState* gif_ptr(void* p) { return static_cast<MsfGifState*>(p); }
} // namespace

GifRecorder::~GifRecorder()
{
    if (recording_) {
        MsfGifResult result = msf_gif_end(gif_ptr(gif_));
        if (result.data) std::free(result.data);
        delete gif_ptr(gif_);
    }
}

void GifRecorder::start(int framebuffer_width, int framebuffer_height)
{
    if (recording_) return;

    fb_width_  = framebuffer_width;
    fb_height_ = framebuffer_height;
    out_width_  = fb_width_;
    out_height_ = fb_height_;

    fb_pixels_.resize(static_cast<std::size_t>(fb_width_) * fb_height_ * 4);

    auto* state = new MsfGifState{};
    msf_gif_begin(state, out_width_, out_height_);
    gif_ = state;

    frame_counter_ = 0;
    gif_frames_    = 0;
    recording_     = true;
}

void GifRecorder::capture_frame()
{
    if (!recording_) return;
    if (++frame_counter_ % frame_skip_ != 0) return;
    readback_and_encode();
}

std::string GifRecorder::stop()
{
    if (!recording_) return {};

    MsfGifResult result = msf_gif_end(gif_ptr(gif_));
    delete gif_ptr(gif_);
    gif_ = nullptr;
    recording_ = false;

    if (!result.data) return {};

    // Write to disk.
    char name[64];
    std::snprintf(name, sizeof(name), "recording_%03d.gif", ++file_counter_);
    FILE* f = std::fopen(name, "wb");
    if (f) {
        std::fwrite(result.data, result.dataSize, 1, f);
        std::fclose(f);
    }
    std::free(result.data);

    last_file_ = name;
    std::printf("GIF saved: %s  (%d frames, %zu bytes)\n",
                name, gif_frames_,
                static_cast<std::size_t>(result.dataSize));
    return last_file_;
}

void GifRecorder::imgui_panel()
{
    if (!recording_) {
        if (ImGui::Button("Record GIF")) {
            // Defer start — the caller passes framebuffer size.
            // We'll mark a "want start" flag and let capture_frame
            // handle it, but it's simpler to just grab the viewport.
            GLint vp[4];
            glGetIntegerv(GL_VIEWPORT, vp);
            start(vp[2], vp[3]);
        }
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImVec4(0.8f, 0.15f, 0.15f, 1.0f));
        if (ImGui::Button("Stop")) {
            stop();
        }
        ImGui::PopStyleColor();
        ImGui::SameLine();
        ImGui::Text("%d frames", gif_frames_);
    }
    if (!last_file_.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", last_file_.c_str());
    }
}

// ----------------------------------------------------------------
// Internal
// ----------------------------------------------------------------

void GifRecorder::readback_and_encode()
{
    // Read the GL framebuffer (bottom-up RGBA).
    glReadPixels(0, 0, fb_width_, fb_height_,
                 GL_RGBA, GL_UNSIGNED_BYTE, fb_pixels_.data());

    // Flip vertically in-place (glReadPixels is bottom-up, GIF is top-down).
    const std::size_t row_bytes = static_cast<std::size_t>(fb_width_) * 4;
    for (int y = 0; y < fb_height_ / 2; ++y) {
        uint8_t* top = fb_pixels_.data() + y * row_bytes;
        uint8_t* bot = fb_pixels_.data() + (fb_height_ - 1 - y) * row_bytes;
        for (std::size_t x = 0; x < row_bytes; ++x)
            std::swap(top[x], bot[x]);
    }

    // GIF delay: at frame_skip_=3 and 60 fps → 20 gif-fps → 5 centisec.
    const int centisec = std::max(1, frame_skip_ * 100 / 60);
    msf_gif_frame(gif_ptr(gif_), fb_pixels_.data(), centisec, 16,
                  fb_width_ * 4);
    ++gif_frames_;
}

} // namespace viz
