/**
 * @file camera.cpp
 * @brief Implementation of the orbit camera.
 *
 * Conventions:
 *   - World up is +Y.
 *   - yaw = 0, pitch = 0 places the eye at target + (0, 0, distance),
 *     i.e. on the +Z axis looking toward −Z (matches GLM defaults).
 *   - yaw rotates around the world +Y axis (azimuth).
 *   - pitch rotates around the camera's right axis (elevation),
 *     clamped to avoid the pole flip.
 */

#include "viz/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace viz {

namespace {

constexpr float kMinDistance = 0.01f;
constexpr float kPitchLimit  = 1.55334f;  // ≈ π/2 − 0.02

} // namespace

OrbitCamera::OrbitCamera(glm::vec3 target, float distance)
    : target_(target),
      distance_(std::max(distance, kMinDistance)),
      yaw_(0.0f),
      pitch_(0.0f)
{}

glm::vec3 OrbitCamera::position() const noexcept
{
    // Spherical → Cartesian offset from target.
    // At (yaw=0, pitch=0) the offset is (0, 0, distance), giving +Z.
    const float cp = std::cos(pitch_);
    const float sp = std::sin(pitch_);
    const float cy = std::cos(yaw_);
    const float sy = std::sin(yaw_);

    const glm::vec3 offset{
        distance_ * cp * sy,
        distance_ * sp,
        distance_ * cp * cy,
    };
    return target_ + offset;
}

void OrbitCamera::orbit(float d_yaw, float d_pitch) noexcept
{
    yaw_   += d_yaw;
    pitch_ = std::clamp(pitch_ + d_pitch, -kPitchLimit, kPitchLimit);
}

void OrbitCamera::zoom(float factor) noexcept
{
    distance_ = std::max(distance_ * factor, kMinDistance);
}

glm::mat4 OrbitCamera::view_matrix() const noexcept
{
    return glm::lookAt(position(), target_, glm::vec3{ 0.0f, 1.0f, 0.0f });
}

glm::mat4 OrbitCamera::projection_matrix(float aspect) const noexcept
{
    return glm::perspective(fov_y_, aspect, near_z_, far_z_);
}

} // namespace viz
