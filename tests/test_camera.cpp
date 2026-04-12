/**
 * @file test_camera.cpp
 * @brief Unit tests for the arcball orbit camera used by the viewer.
 *
 * The camera is a simple "look at a target from a given distance and
 * orientation" controller. The testable surface is the math:
 *
 *   - view_matrix()        — right-handed look-at.
 *   - projection_matrix()  — perspective, configurable aspect.
 *   - position()           — world-space eye position.
 *   - orbit(dyaw, dpitch)  — rotates around the target.
 *   - zoom(factor)         — scales the distance.
 *
 * No OpenGL state is touched here, so all of this is plain GLM math
 * and cleanly testable.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "viz/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <numbers>

using viz::OrbitCamera;
using Catch::Matchers::WithinAbs;

namespace {

/// @brief Shorthand: are two glm::vec3 close?
void require_vec3_close(const glm::vec3& a, const glm::vec3& b,
                        float tol = 1e-5f)
{
    REQUIRE_THAT(a.x, WithinAbs(b.x, tol));
    REQUIRE_THAT(a.y, WithinAbs(b.y, tol));
    REQUIRE_THAT(a.z, WithinAbs(b.z, tol));
}

} // namespace

TEST_CASE("OrbitCamera starts in a known pose", "[camera]")
{
    const glm::vec3 target{ 1.0f, 2.0f, 3.0f };
    OrbitCamera cam(target, /*distance=*/5.0f);

    REQUIRE_THAT(cam.distance(), WithinAbs(5.0f, 1e-6f));
    require_vec3_close(cam.target(), target);

    // Default pose: looking along −Z, y up. Position should be target + 5·ẑ.
    require_vec3_close(cam.position(), target + glm::vec3{ 0.0f, 0.0f, 5.0f });
}

TEST_CASE("OrbitCamera::view_matrix maps the eye to the origin",
          "[camera][view]")
{
    OrbitCamera cam(glm::vec3{ 0.0f }, 4.0f);
    const glm::mat4 view = cam.view_matrix();

    // View matrix applied to the eye position yields (0,0,0, 1).
    const glm::vec4 eye4 = view * glm::vec4(cam.position(), 1.0f);
    REQUIRE_THAT(eye4.x, WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(eye4.y, WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(eye4.z, WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("OrbitCamera::view_matrix maps the target to -distance along -Z",
          "[camera][view]")
{
    const glm::vec3 target{ 2.0f, -1.0f, 0.5f };
    const float     dist = 4.0f;
    OrbitCamera cam(target, dist);
    const glm::mat4 view = cam.view_matrix();

    const glm::vec4 t = view * glm::vec4(target, 1.0f);
    REQUIRE_THAT(t.x, WithinAbs(0.0f,  1e-5f));
    REQUIRE_THAT(t.y, WithinAbs(0.0f,  1e-5f));
    REQUIRE_THAT(t.z, WithinAbs(-dist, 1e-5f));
}

TEST_CASE("OrbitCamera::zoom scales the distance but preserves the target",
          "[camera][zoom]")
{
    OrbitCamera cam(glm::vec3{ 0.0f }, 4.0f);
    cam.zoom(0.5f);       // halve the distance

    REQUIRE_THAT(cam.distance(), WithinAbs(2.0f, 1e-6f));
    require_vec3_close(cam.target(), glm::vec3{ 0.0f });
}

TEST_CASE("OrbitCamera::orbit(π, 0) swings to the opposite side",
          "[camera][orbit]")
{
    OrbitCamera cam(glm::vec3{ 0.0f }, 4.0f);
    cam.orbit(std::numbers::pi_v<float>, 0.0f);

    // After a π yaw we should be at approximately target + (0, 0, −4).
    require_vec3_close(cam.position(), glm::vec3{ 0.0f, 0.0f, -4.0f }, 1e-5f);
    // Distance is preserved.
    REQUIRE_THAT(cam.distance(), WithinAbs(4.0f, 1e-6f));
}

TEST_CASE("OrbitCamera::projection_matrix encodes the aspect ratio",
          "[camera][projection]")
{
    OrbitCamera cam(glm::vec3{ 0.0f }, 4.0f);
    const glm::mat4 p1 = cam.projection_matrix(1.0f);  // square
    const glm::mat4 p2 = cam.projection_matrix(2.0f);  // 2:1

    // A 2:1 aspect ratio compresses the x-direction FOV; the [0][0]
    // element (x-scale) should halve.
    REQUIRE_THAT(p2[0][0], WithinAbs(p1[0][0] * 0.5f, 1e-6f));
}
