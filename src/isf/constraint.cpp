/**
 * @file constraint.cpp
 * @brief Implementation of Algorithm 4 (velocity-constraint projection).
 */

#include "isf/constraint.hpp"

#include "isf/grid.hpp"
#include "isf/pressure.hpp"

#include <cmath>
#include <complex>

namespace isf {

void apply_velocity_constraint(Grid3D& g,
                               const RegionMask& region,
                               double kx, double ky, double kz)
{
    const int    nx = g.nx();
    const int    ny = g.ny();
    const int    nz = g.nz();
    const double dx = g.dx();
    const double dy = g.dy();
    const double dz = g.dz();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (!region(i, j, k)) continue;

                const double x = i * dx;
                const double y = j * dy;
                const double z = k * dz;
                const double phase = kx * x + ky * y + kz * z;

                const std::complex<double> pw{ std::cos(phase),
                                               std::sin(phase) };

                const auto p1 = g.psi1(i, j, k);
                const auto p2 = g.psi2(i, j, k);
                const double m1 = std::abs(p1);
                const double m2 = std::abs(p2);

                g.psi1(i, j, k) = pw * m1;
                g.psi2(i, j, k) = pw * m2;
            }
        }
    }
}

void constraint_project(Grid3D& g,
                        const RegionMask& region,
                        double kx, double ky, double kz,
                        PressureWorkspace& ws)
{
    apply_velocity_constraint(g, region, kx, ky, kz);
    pressure_project(g, ws);
}

} // namespace isf
