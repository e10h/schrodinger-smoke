/**
 * @file velocity.cpp
 * @brief Implementation of vertex-centered velocity recovery.
 */

#include "isf/velocity.hpp"

#include "isf/grid.hpp"
#include "isf/pressure.hpp"

#include <cmath>
#include <complex>

namespace isf {

namespace {

/// @brief arg⟨ψ_v, ψ_w⟩_ℂ for two vertices v, w — Eq. 4 without ℏ.
inline double edge_arg(const Grid3D& g,
                       std::size_t v, std::size_t w) noexcept
{
    const auto inner = std::conj(g.psi1(v)) * g.psi1(w)
                     + std::conj(g.psi2(v)) * g.psi2(w);
    return std::arg(inner);
}

} // namespace

void recover_velocity(const Grid3D& g, double hbar, VelocityField& out)
{
    const int nx = g.nx();
    const int ny = g.ny();
    const int nz = g.nz();
    const double dx = g.dx();
    const double dy = g.dy();
    const double dz = g.dz();
    const std::size_t N = g.size();

    if (out.vx.size() != N) out.vx.assign(N, 0.0);
    if (out.vy.size() != N) out.vy.assign(N, 0.0);
    if (out.vz.size() != N) out.vz.assign(N, 0.0);

    // Factors that convert (edge-arg average) to (physical velocity).
    // We average two edges spanning 2·dα and divide by 2·dα, i.e.
    // multiply by ℏ / (2·dα). The effective per-side factor is then
    // ℏ/(2·dα), applied to the summed arg pair.
    const double fx = hbar / (2.0 * dx);
    const double fy = hbar / (2.0 * dy);
    const double fz = hbar / (2.0 * dz);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t v = g.index(i, j, k);

                // +axis neighbours (periodic wrap).
                const std::size_t xp = g.wrap(i + 1, j,     k    );
                const std::size_t yp = g.wrap(i,     j + 1, k    );
                const std::size_t zp = g.wrap(i,     j,     k + 1);

                // -axis neighbours for the "incoming" edge.
                const std::size_t xm = g.wrap(i - 1, j,     k    );
                const std::size_t ym = g.wrap(i,     j - 1, k    );
                const std::size_t zm = g.wrap(i,     j,     k - 1);

                // η_x(v)        = ℏ · arg⟨ψ_v, ψ_{v+ê_x}⟩
                // η_x(v − ê_x)  = ℏ · arg⟨ψ_{v−ê_x}, ψ_v⟩
                // v_x(v)        = ½ · (η_x(v) + η_x(v−ê_x)) / dx
                //                = fx · (arg(...) + arg(...))
                out.vx[v] = fx * (edge_arg(g, v,  xp) + edge_arg(g, xm, v));
                out.vy[v] = fy * (edge_arg(g, v,  yp) + edge_arg(g, ym, v));
                out.vz[v] = fz * (edge_arg(g, v,  zp) + edge_arg(g, zm, v));
            }
        }
    }
}

VelocityField recover_velocity(const Grid3D& g, double hbar)
{
    VelocityField out;
    recover_velocity(g, hbar, out);
    return out;
}

void recover_velocity_from_workspace(const Grid3D& g, double hbar,
                                     const PressureWorkspace& ws,
                                     VelocityField& out)
{
    const int nx = g.nx();
    const int ny = g.ny();
    const int nz = g.nz();
    const double dx = g.dx();
    const double dy = g.dy();
    const double dz = g.dz();
    const std::size_t N = g.size();

    if (out.vx.size() != N) out.vx.assign(N, 0.0);
    if (out.vy.size() != N) out.vy.assign(N, 0.0);
    if (out.vz.size() != N) out.vz.assign(N, 0.0);

    const double fx = hbar / (2.0 * dx);
    const double fy = hbar / (2.0 * dy);
    const double fz = hbar / (2.0 * dz);

    // Post-correction edge 1-form:
    //   η̃'(v, v+1) = η̃(v, v+1) − [q̃(v+1) − q̃(v)]
    //
    // Vertex velocity (average of two incident corrected edges):
    //   v_x(v) = ℏ/(2dx) · [η̃'(v,v+1) + η̃'(v−1,v)]
    //          = ℏ/(2dx) · [η̃_x(v) + η̃_x(v−1) − q(v+1) + q(v−1)]
    //
    // No inner products needed — pure arithmetic on cached workspace.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t v  = g.index(i, j, k);
                const std::size_t xm = g.wrap(i - 1, j,     k    );
                const std::size_t xp = g.wrap(i + 1, j,     k    );
                const std::size_t ym = g.wrap(i,     j - 1, k    );
                const std::size_t yp = g.wrap(i,     j + 1, k    );
                const std::size_t zm = g.wrap(i,     j,     k - 1);
                const std::size_t zp = g.wrap(i,     j,     k + 1);

                out.vx[v] = fx * (ws.eta_x[v] + ws.eta_x[xm]
                                  - ws.q[xp] + ws.q[xm]);
                out.vy[v] = fy * (ws.eta_y[v] + ws.eta_y[ym]
                                  - ws.q[yp] + ws.q[ym]);
                out.vz[v] = fz * (ws.eta_z[v] + ws.eta_z[zm]
                                  - ws.q[zp] + ws.q[zm]);
            }
        }
    }
}

} // namespace isf
