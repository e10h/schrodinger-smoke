/**
 * @file particles.cpp
 * @brief Implementation of the CPU tracer-particle system.
 */

#include "viz/particles.hpp"

#include "isf/grid.hpp"
#include "isf/velocity.hpp"

#include <cmath>
#include <cstddef>

namespace viz {

namespace {

/**
 * @brief Wrap a floating-point value into [0, period).
 *
 * `std::fmod` returns a value with the sign of the dividend, which
 * leaves negatives negative; we shift those back up so the result
 * lies strictly in [0, period) for any finite input.
 */
inline double wrap_periodic(double v, double period) noexcept
{
    double r = std::fmod(v, period);
    if (r < 0.0) r += period;
    // Guard against `r == period` that can arise from fmod round-off on
    // negative values (fmod(-0.0, period) can yield exactly period on some
    // libcs after the +period correction above).
    if (r >= period) r -= period;
    return r;
}

/**
 * @brief Trilinear interpolation core: take a world position, a scalar
 *        field, and compute the 8-corner weighted sum.
 *
 * The caller already has the grid; this helper just implements the
 * per-axis wrap + fractional decomposition + 8-corner blend.
 */
double trilinear_core(const isf::Grid3D& g,
                      const std::vector<double>& field,
                      double x, double y, double z)
{
    // Map world coordinate to continuous lattice coordinate, wrapped
    // into [0, N) so that fractional parts are well-defined for any
    // finite input including negatives.
    const double u = wrap_periodic(x / g.dx(), static_cast<double>(g.nx()));
    const double v = wrap_periodic(y / g.dy(), static_cast<double>(g.ny()));
    const double w = wrap_periodic(z / g.dz(), static_cast<double>(g.nz()));

    const int i0 = static_cast<int>(std::floor(u));
    const int j0 = static_cast<int>(std::floor(v));
    const int k0 = static_cast<int>(std::floor(w));

    const double fx = u - i0;
    const double fy = v - j0;
    const double fz = w - k0;

    // Neighbours on the upper side, with periodic wrap.
    const int i1 = (i0 + 1 >= g.nx()) ? 0 : i0 + 1;
    const int j1 = (j0 + 1 >= g.ny()) ? 0 : j0 + 1;
    const int k1 = (k0 + 1 >= g.nz()) ? 0 : k0 + 1;

    const auto at = [&](int i, int j, int k) {
        return field[g.index(i, j, k)];
    };

    // Classic 8-corner trilinear blend.
    const double c000 = at(i0, j0, k0);
    const double c100 = at(i1, j0, k0);
    const double c010 = at(i0, j1, k0);
    const double c110 = at(i1, j1, k0);
    const double c001 = at(i0, j0, k1);
    const double c101 = at(i1, j0, k1);
    const double c011 = at(i0, j1, k1);
    const double c111 = at(i1, j1, k1);

    const double c00 = c000 * (1.0 - fx) + c100 * fx;
    const double c01 = c001 * (1.0 - fx) + c101 * fx;
    const double c10 = c010 * (1.0 - fx) + c110 * fx;
    const double c11 = c011 * (1.0 - fx) + c111 * fx;

    const double c0 = c00 * (1.0 - fy) + c10 * fy;
    const double c1 = c01 * (1.0 - fy) + c11 * fy;

    return c0 * (1.0 - fz) + c1 * fz;
}

/**
 * @brief Sample all three velocity components at a world position.
 */
inline void sample_velocity(const isf::Grid3D& g,
                            const isf::VelocityField& vf,
                            double x, double y, double z,
                            double& vx, double& vy, double& vz)
{
    vx = trilinear_core(g, vf.vx, x, y, z);
    vy = trilinear_core(g, vf.vy, x, y, z);
    vz = trilinear_core(g, vf.vz, x, y, z);
}

} // namespace

double sample_trilinear(const isf::Grid3D& g,
                        const std::vector<double>& field,
                        double x, double y, double z)
{
    return trilinear_core(g, field, x, y, z);
}

void advect_particles_rk2(ParticleSystem& ps,
                          const isf::Grid3D& g,
                          const isf::VelocityField& vf,
                          double dt)
{
    const double lx = g.lx();
    const double ly = g.ly();
    const double lz = g.lz();
    const double half_dt = 0.5 * dt;

    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(ps.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t p = 0; p < n; ++p) {
        const std::size_t i = static_cast<std::size_t>(p);

        // k₁ = v(x_n)
        double v0x, v0y, v0z;
        sample_velocity(g, vf, ps.x[i], ps.y[i], ps.z[i],
                        v0x, v0y, v0z);

        // x_mid = x_n + (dt/2) · k₁
        const double mx = ps.x[i] + half_dt * v0x;
        const double my = ps.y[i] + half_dt * v0y;
        const double mz = ps.z[i] + half_dt * v0z;

        // k₂ = v(x_mid)
        double v1x, v1y, v1z;
        sample_velocity(g, vf, mx, my, mz, v1x, v1y, v1z);

        // x_{n+1} = x_n + dt · k₂, then wrap into the periodic box.
        ps.x[i] = wrap_periodic(ps.x[i] + dt * v1x, lx);
        ps.y[i] = wrap_periodic(ps.y[i] + dt * v1y, ly);
        ps.z[i] = wrap_periodic(ps.z[i] + dt * v1z, lz);
    }
}

} // namespace viz
