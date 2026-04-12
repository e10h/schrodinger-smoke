/**
 * @file camera.hpp
 * @brief Arcball-style orbit camera for the interactive viewer.
 *
 * OrbitCamera describes a right-handed "look at a target from some
 * distance" camera parameterised by:
 *
 *   - target         (world-space focal point)
 *   - distance       (distance from eye to target)
 *   - yaw, pitch     (spherical angles around the target)
 *   - up axis        (world +Y throughout)
 *
 * The class is pure math — no OpenGL state is touched. It is
 * consumed both by the viewer (for live view/projection matrices)
 * and by unit tests (which verify the matrix contracts against GLM's
 * lookAt and perspective).
 */

#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace viz {

/**
 * @brief Orbit camera centred on a target, with a mutable distance and
 *        yaw/pitch orientation.
 */
class OrbitCamera {
public:
    /**
     * @brief Construct a camera looking at `target` from `distance`.
     *
     * Initial yaw and pitch are both zero, which places the eye at
     * `target + (0, 0, distance)` — i.e. on the +Z axis looking
     * toward −Z.
     *
     * @param target   World-space focal point.
     * @param distance Distance from the eye to the target.
     */
    OrbitCamera(glm::vec3 target, float distance);

    /// @brief Current focal point.
    [[nodiscard]] glm::vec3 target() const noexcept { return target_; }

    /// @brief Current distance from eye to target.
    [[nodiscard]] float distance() const noexcept { return distance_; }

    /// @brief World-space eye position derived from (target, distance, yaw, pitch).
    [[nodiscard]] glm::vec3 position() const noexcept;

    /**
     * @brief Orbit the camera around the target.
     *
     * @param d_yaw   Delta azimuth in radians.
     * @param d_pitch Delta elevation in radians; clamped to keep the
     *                view from flipping over the poles.
     */
    void orbit(float d_yaw, float d_pitch) noexcept;

    /**
     * @brief Scale the distance by `factor`.
     *
     * Values < 1 zoom in, values > 1 zoom out. The distance is clamped
     * to a small positive minimum.
     */
    void zoom(float factor) noexcept;

    /** @brief Move the target (pan). */
    void set_target(glm::vec3 target) noexcept { target_ = target; }

    /** @brief Right-handed view matrix (`glm::lookAt`). */
    [[nodiscard]] glm::mat4 view_matrix() const noexcept;

    /**
     * @brief Perspective projection matrix with the current vertical
     *        field of view and the given aspect ratio.
     *
     * @param aspect viewport_width / viewport_height
     */
    [[nodiscard]] glm::mat4 projection_matrix(float aspect) const noexcept;

    /// @brief Set the vertical field of view in radians.
    void set_fov_y(float radians) noexcept { fov_y_ = radians; }

    /// @brief Set the clipping-plane distances.
    void set_clip(float near_z, float far_z) noexcept {
        near_z_ = near_z;
        far_z_  = far_z;
    }

private:
    glm::vec3 target_;
    float     distance_;
    float     yaw_;
    float     pitch_;
    float     fov_y_  = 1.0472f;   ///< ≈ 60°
    float     near_z_ = 0.01f;
    float     far_z_  = 100.0f;
};

} // namespace viz
