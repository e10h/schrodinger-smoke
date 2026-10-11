/**
 * @file window.cpp
 * @brief SDL3 window + WebGPU (wgfx) + ImGui backend implementation.
 */

#define WGPU_IMPLEMENTATION
#include <wgfx.h>

#include "viz/window.hpp"

#include <imgui.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_wgpu.h>

#include <cstdio>
#include <stdexcept>

namespace viz {

Window::Window(int width, int height, const char* title)
{
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
    }

    window_ = SDL_CreateWindow(title, width, height, SDL_WINDOW_RESIZABLE);
    if (!window_) {
        SDL_Quit();
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
    }

    // Initialize WebGPU surface via wgfx
    wgfx::init(wgfx::getSurface(window_));

    // Create swapchain color texture and render passes
    color_tex_  = new wgfx::ColorTexture();
    scene_pass_ = new wgfx::RenderPass();
    ui_pass_    = new wgfx::RenderPass();

    scene_pass_->addTarget(color_tex_);
    ui_pass_->addTarget(color_tex_);

    scene_pass_->setClear({ 0.02, 0.03, 0.06, 1.0 });
    scene_pass_->shouldClear = true;
    ui_pass_->shouldClear    = false;

    // Initialize Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplSDL3_InitForOther(window_);

    ImGui_ImplWGPU_InitInfo init_info = {};
    init_info.Device = (WGPUDevice)wgfx::device;
    init_info.NumFramesInFlight = 2;
    init_info.RenderTargetFormat = (WGPUTextureFormat)wgfx::surfaceFormat;
    init_info.DepthStencilFormat = WGPUTextureFormat_Undefined;
    ImGui_ImplWGPU_Init(&init_info);

    SDL_GetMouseState(&prev_mouse_x_, &prev_mouse_y_);
}

Window::~Window()
{
    ImGui_ImplWGPU_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    delete scene_pass_;
    delete ui_pass_;
    delete color_tex_;

    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

int Window::width() const noexcept
{
    int w = 0, h = 0;
    SDL_GetWindowSize(window_, &w, &h);
    return w;
}

int Window::height() const noexcept
{
    int w = 0, h = 0;
    SDL_GetWindowSize(window_, &w, &h);
    return h;
}

void Window::framebuffer_size(int& w, int& h) const noexcept
{
    SDL_GetWindowSizeInPixels(window_, &w, &h);
    if (w <= 0) w = 1;
    if (h <= 0) h = 1;
}

void Window::poll_events()
{
    scroll_accum_ = 0.0f;

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);

        switch (event.type) {
        case SDL_EVENT_QUIT:
            should_close_ = true;
            break;

        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_RESIZED:
        case SDL_EVENT_WINDOW_EXPOSED:
            wgfx::initSurface();
            break;

        case SDL_EVENT_KEY_DOWN:
            if (event.key.key == SDLK_ESCAPE) {
                should_close_ = true;
            }
            break;

        case SDL_EVENT_MOUSE_WHEEL:
            if (!ImGui::GetIO().WantCaptureMouse) {
                scroll_accum_ += event.wheel.y;
            }
            break;
        }
    }

    float mx = 0.0f, my = 0.0f;
    const SDL_MouseButtonFlags buttons = SDL_GetMouseState(&mx, &my);

    frame_input_.mouse_dx        = mx - prev_mouse_x_;
    frame_input_.mouse_dy        = my - prev_mouse_y_;
    prev_mouse_x_                = mx;
    prev_mouse_y_                = my;
    frame_input_.left_mouse_down = (buttons & SDL_BUTTON_LMASK) != 0;
    frame_input_.scroll_dy       = scroll_accum_;
    frame_input_.imgui_captures  = ImGui::GetIO().WantCaptureMouse;
}

void Window::begin_frame()
{
    wgfx::touch(color_tex_);
    wgfx::start();

    ImGui_ImplWGPU_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void Window::end_frame()
{
    ImGui::Render();

    ui_pass_->prepare();
    ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(),
                                  (WGPURenderPassEncoder)ui_pass_->renderPass);
    ui_pass_->end();

    wgfx::frame();
}

} // namespace viz
