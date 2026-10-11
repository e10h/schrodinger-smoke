/**
 * @file gif_recorder.cpp
 * @brief GIF recorder stub for the WebGPU viewer.
 */

#include "viz/gif_recorder.hpp"
#include <imgui.h>

namespace viz {

GifRecorder::~GifRecorder() = default;

void GifRecorder::start(int /*framebuffer_width*/, int /*framebuffer_height*/)
{
    recording_ = false;
}

void GifRecorder::capture_frame()
{
}

std::string GifRecorder::stop()
{
    recording_ = false;
    return {};
}

void GifRecorder::imgui_panel()
{
    ImGui::TextDisabled("GIF recorder: WebGPU readback disabled");
}

void GifRecorder::readback_and_encode()
{
}

} // namespace viz
