/**
 * @file particles.hpp
 * @brief Passive tracer particles advected through an ISF velocity field.
 *
 * The particle system is a structure-of-arrays position buffer (one
 * `std::vector<double>` per axis) so the layout maps one-to-one onto
 * three GPU SSBOs when we later port to the compute-shader solver.
 *
 * All positions are treated as living on the periodic torus of the
 * grid: after each advection step positions are wrapped back into
 * [0, L_α) along the corresponding axis. Trilinear sampling of a
 * grid-defined scalar field also wraps, so queries at any world
 * position are valid.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace isf {
class Grid3D;
struct VelocityField;
} // namespace isf

namespace viz {

/**
 * @brief Structure-of-arrays particle cloud.
 *
 * Keeping x, y, z in separate contiguous buffers matches the layout
 * we'll want on the GPU (three SSBOs) and is also slightly friendlier
 * to auto-vectorisation on the CPU path.
 *
 * Optional per-particle RGB colour: when `r/g/b` are populated they
 * are uploaded to a second VBO and used by the renderer; when empty
 * the renderer falls back to a single uniform colour. This lets the
 * advection unit tests stay colour-blind while the demo can ship
 * red/blue ink for the colliding-spheres scene.
 */
struct ParticleSystem {
    std::vector<double> x;  ///< world x-positions
    std::vector<double> y;  ///< world y-positions
    std::vector<double> z;  ///< world z-positions

    std::vector<float>  r;  ///< per-particle red   (0–1) — optional
    std::vector<float>  g;  ///< per-particle green (0–1) — optional
    std::vector<float>  b;  ///< per-particle blue  (0–1) — optional

    /// @brief Number of particles currently stored.
    [[nodiscard]] std::size_t size() const noexcept { return x.size(); }

    /// @brief True if a colour buffer of matching size is present.
    [[nodiscard]] bool has_colors() const noexcept {
        return r.size() == x.size() && r.size() > 0;
    }
};

/**
 * @brief Trilinearly sample a vertex-centered scalar field at a world
 *        position.
 *
 * The field is interpreted as a row-major (x-fastest) array of length
 * `g.size()`, with value `field[g.index(i,j,k)]` located at the world
 * position `(i·dx, j·dy, k·dz)`. Out-of-range world coordinates wrap
 * periodically with the grid periods `(L_x, L_y, L_z)`.
 *
 * Exact at vertex positions, linear in each axis otherwise.
 *
 * @param g     Grid providing the lattice geometry.
 * @param field Vertex-centered scalar field, size == `g.size()`.
 * @param x     World x-coordinate.
 * @param y     World y-coordinate.
 * @param z     World z-coordinate.
 * @return Interpolated value at (x, y, z).
 */
[[nodiscard]] double
sample_trilinear(const isf::Grid3D& g,
                 const std::vector<double>& field,
                 double x, double y, double z);

/**
 * @brief Advect every particle in `ps` by one RK2 (midpoint) step.
 *
 * RK2 midpoint method:
 *
 *     v₀ = v(x_n)
 *     x_mid = x_n + (dt/2) · v₀
 *     v₁ = v(x_mid)
 *     x_{n+1} = x_n + dt · v₁
 *
 * Each velocity sample uses periodic trilinear interpolation of the
 * three VelocityField components. Positions are wrapped into
 * [0, L_α) after the step.
 *
 * @param ps  Particle cloud to advect (mutated in place).
 * @param g   Grid providing the lattice geometry for vf's layout.
 * @param vf  Vertex-centered velocity field (same layout as g).
 * @param dt  Time step.
 */
void advect_particles_rk2(ParticleSystem& ps,
                          const isf::Grid3D& g,
                          const isf::VelocityField& vf,
                          double dt);

} // namespace viz
