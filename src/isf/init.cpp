/**
 * @file init.cpp
 * @brief Implementations of the Grid3D initial-condition builders.
 */

#include "isf/init.hpp"

#include "isf/constraint.hpp"
#include "isf/grid.hpp"
#include "isf/normalize.hpp"
#include "isf/pressure.hpp"

#include <cmath>
#include <complex>
#include <numbers>

namespace isf {

void init_plane_wave(Grid3D& g, double kx, double ky, double kz)
{
    const int    nx = g.nx();
    const int    ny = g.ny();
    const int    nz = g.nz();
    const double dx = g.dx();
    const double dy = g.dy();
    const double dz = g.dz();

    for (int k = 0; k < nz; ++k) {
        const double z = k * dz;
        for (int j = 0; j < ny; ++j) {
            const double y = j * dy;
            for (int i = 0; i < nx; ++i) {
                const double x = i * dx;
                const double phase = kx * x + ky * y + kz * z;
                g.psi1(i, j, k) = { std::cos(phase), std::sin(phase) };
                g.psi2(i, j, k) = { 0.0, 0.0 };
            }
        }
    }
}

void init_vortex_ring_z(Grid3D& g,
                        double cx, double cy, double cz,
                        double R, double r,
                        double epsilon)
{
    const int    nx = g.nx();
    const int    ny = g.ny();
    const int    nz = g.nz();
    const double dx = g.dx();
    const double dy = g.dy();
    const double dz = g.dz();
    const double pi = std::numbers::pi_v<double>;

    for (int k = 0; k < nz; ++k) {
        const double z_rel = k * dz - cz;
        for (int j = 0; j < ny; ++j) {
            const double y_rel = j * dy - cy;
            for (int i = 0; i < nx; ++i) {
                const double x_rel = i * dx - cx;
                const double rxy = std::sqrt(x_rel * x_rel + y_rel * y_rel);

                // Slab = disk of radius R at height cz, half-thickness r.
                double theta = 0.0;
                if (rxy < R && std::abs(z_rel) < r) {
                    theta = pi * (1.0 + z_rel / r);
                }

                // ψ₁ starts at 1, ψ₂ starts at ε (guard against zeros).
                g.psi1(i, j, k) = { std::cos(theta), std::sin(theta) };
                g.psi2(i, j, k) = { epsilon, 0.0 };
            }
        }
    }

    // Enforce |ψ|² = 1 and div-free velocity field.
    normalize(g);
    pressure_project(g);
}

void init_two_velocity_balls(Grid3D& g,
                             double cx1, double cy1, double cz1,
                             double cx2, double cy2, double cz2,
                             double R,
                             double vx1, double vy1, double vz1,
                             double vx2, double vy2, double vz2,
                             double hbar,
                             int    iters,
                             double epsilon)
{
    // Step 1: clean initial guess (1, ε), normalized.
    {
        const double inv = 1.0 / std::sqrt(1.0 + epsilon * epsilon);
        for (std::size_t n = 0; n < g.size(); ++n) {
            g.psi1(n) = { 1.0     * inv, 0.0 };
            g.psi2(n) = { epsilon * inv, 0.0 };
        }
    }

    // Region predicates: spheres around (c·1) and (c·2).
    const double R2 = R * R;
    auto in_ball1 = [cx1, cy1, cz1, R2, &g](int i, int j, int k) {
        const double x = i * g.dx() - cx1;
        const double y = j * g.dy() - cy1;
        const double z = k * g.dz() - cz1;
        return x * x + y * y + z * z < R2;
    };
    auto in_ball2 = [cx2, cy2, cz2, R2, &g](int i, int j, int k) {
        const double x = i * g.dx() - cx2;
        const double y = j * g.dy() - cy2;
        const double z = k * g.dz() - cz2;
        return x * x + y * y + z * z < R2;
    };

    // Velocity → wave-vector conversion (v = ℏk).
    const double k1x = vx1 / hbar, k1y = vy1 / hbar, k1z = vz1 / hbar;
    const double k2x = vx2 / hbar, k2y = vy2 / hbar, k2z = vz2 / hbar;

    // Multi-region Algorithm 4: apply each region's constraint then
    // project once. Iterate to the fixed point.
    PressureWorkspace ws;
    for (int it = 0; it < iters; ++it) {
        apply_velocity_constraint(g, in_ball1, k1x, k1y, k1z);
        apply_velocity_constraint(g, in_ball2, k2x, k2y, k2z);
        pressure_project(g, ws);
    }
}

} // namespace isf
