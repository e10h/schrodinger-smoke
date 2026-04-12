/**
 * @file velocity.hpp
 * @brief Recovery of a vertex-centered velocity field from ψ.
 *
 * The ISF paper stores velocity as an edge 1-form η on directed edges
 * (Eq. 4):
 *
 *     η_vw  =  ℏ · arg⟨ψ_v, ψ_w⟩_ℂ
 *
 * For particle advection and visualization we prefer a 3-vector
 * sample at each vertex.  The natural vertex-centered reconstruction
 * averages the two incident edges per axis:
 *
 *     v_x(v) = ½ · (η_x(v) + η_x(v − ê_x)) / dx
 *
 * and analogously for y, z.
 *
 * The "fast" overload avoids recomputing edge args from ψ by reusing
 * the pre-correction edge 1-form η̃ and pressure potential q̃ that
 * `pressure_project` already computed.  The post-correction edge
 * velocity is simply  η̃' = η̃ − ∇q̃  (exact in the arg form), so
 * the vertex velocity becomes:
 *
 *     v_x(v) = ℏ/(2dx) · [η̃(v,v+1) + η̃(v−1,v)
 *                          − q̃(v+1) + q̃(v−1)]
 */

#pragma once

#include <cstddef>
#include <vector>

namespace isf {

class Grid3D;

/**
 * @brief Vertex-centered velocity sample (three scalar fields) in the
 *        same row-major (x-fastest) layout as the grid.
 */
struct VelocityField {
    std::vector<double> vx;   ///< x-component, one value per vertex
    std::vector<double> vy;   ///< y-component
    std::vector<double> vz;   ///< z-component

    /// @brief Number of vertices (== `vx.size()`).
    [[nodiscard]] std::size_t size() const noexcept { return vx.size(); }
};

/**
 * @brief Reconstruct a vertex-centered velocity field from ψ.
 *
 * Convenience overload that allocates a fresh VelocityField each call.
 * Use the in-place overload below in tight loops to amortise away the
 * per-call ~6 MB·3-component allocation.
 *
 * @param g    Grid holding ψ.
 * @param hbar ℏ used to convert scaled arg-values back into physical
 *             velocity units (η = ℏ · arg⟨ψ_v, ψ_w⟩, v = η / dx).
 * @return Velocity field with `g.size()` samples per component.
 */
[[nodiscard]] VelocityField
recover_velocity(const Grid3D& g, double hbar);

/**
 * @brief Reconstruct the vertex-centered velocity field into a
 *        caller-provided buffer.
 *
 * Functionally identical to the value-returning overload but reuses
 * `out`'s storage across calls. Lazily resizes `out` if its component
 * vectors don't match `g.size()`.
 *
 * @param g    Grid holding ψ.
 * @param hbar ℏ.
 * @param out  Velocity field to fill in place.
 */
void recover_velocity(const Grid3D& g, double hbar, VelocityField& out);

struct PressureWorkspace;   // forward (defined in pressure.hpp)

/**
 * @brief Fast velocity recovery from cached pressure-workspace data.
 *
 * Instead of recomputing arg⟨ψ_v, ψ_w⟩ at every edge (the hot path
 * in the standard `recover_velocity`), this overload derives the
 * post-correction edge 1-form algebraically:
 *
 *     η̃'(v,w) = η̃(v,w) − [q̃(w) − q̃(v)]
 *
 * using the pre-correction η̃ and Poisson solution q̃ that
 * `pressure_project` left in `ws`.  This is exact for the arg form
 * and eliminates the entire edge-arg recomputation (~25 % of frame
 * time at 128³).
 *
 * @param g    Grid providing lattice geometry.
 * @param hbar ℏ.
 * @param ws   PressureWorkspace populated by the preceding
 *             `pressure_project` call on the same grid.
 * @param out  Velocity field to fill in place.
 */
void recover_velocity_from_workspace(const Grid3D& g, double hbar,
                                     const PressureWorkspace& ws,
                                     VelocityField& out);

} // namespace isf
