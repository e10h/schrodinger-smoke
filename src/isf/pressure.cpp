/**
 * @file pressure.cpp
 * @brief Implementation of Algorithm 3 (pressure projection).
 *
 * Edge orientation convention: η̃ along the x-axis at the tail vertex
 * (i, j, k) is the edge going to (i+1, j, k), with periodic wrap on the
 * upper boundary. Same pattern for y and z.
 *
 * Divergence at a vertex (Eq. 5) specialises on a regular Cartesian grid
 * to the backward finite-difference of the signed flux:
 *
 *   ξ̃(i,j,k) = [η̃_x(i,j,k) − η̃_x(i−1,j,k)] / dx²
 *            + [η̃_y(i,j,k) − η̃_y(i,j−1,k)] / dy²
 *            + [η̃_z(i,j,k) − η̃_z(i,j,k−1)] / dz²
 *
 * because A_vw/l_vw divided by V_v equals 1/dx² for an x-edge (and so on).
 *
 * The discrete Laplacian built by `div(grad·)` of the *centred*
 * differences `(q(i+1) − q(i−1))/…` has FFT eigenvalues
 *
 *   λ̄_k = −(4/dx²)·sin²(π k_x/N_x)
 *         −(4/dy²)·sin²(π k_y/N_y)
 *         −(4/dz²)·sin²(π k_z/N_z)                      (Eq. 17)
 *
 * which is exactly what we must invert to recover q̃ such that the phase
 * correction e^(−iq̃)·ψ zeroes ξ̃' to machine precision. (Using the
 * continuous eigenvalues from Eq. 18 instead would leave an O((k·dx)⁴)
 * residual — tolerable in practice but it would ruin the unit test, so
 * we stick with Eq. 17.)
 */

#include "isf/pressure.hpp"

#include "isf/fft.hpp"
#include "isf/grid.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>

namespace isf {

namespace {

/**
 * @brief Compute the scaled edge velocity 1-form η̃ on all axis-aligned
 *        edges: η̃_vw = arg⟨ψ_v, ψ_w⟩_ℂ (Eq. 4, without the ℏ factor).
 *
 * Output arrays are sized `g.size()`; `eta_x[v]` stores the x-edge
 * whose tail is vertex v, with periodic wrap on the upper boundary.
 */
void compute_scaled_edge_velocity(const Grid3D& g,
                                  std::vector<double>& eta_x,
                                  std::vector<double>& eta_y,
                                  std::vector<double>& eta_z)
{
    const int nx = g.nx();
    const int ny = g.ny();
    const int nz = g.nz();

    if (eta_x.size() != g.size()) eta_x.assign(g.size(), 0.0);
    if (eta_y.size() != g.size()) eta_y.assign(g.size(), 0.0);
    if (eta_z.size() != g.size()) eta_z.assign(g.size(), 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t v  = g.index(i, j, k);

                // Neighbours in +x/+y/+z with periodic wrap.
                const std::size_t nxi = g.wrap(i + 1, j,     k    );
                const std::size_t nyi = g.wrap(i,     j + 1, k    );
                const std::size_t nzi = g.wrap(i,     j,     k + 1);

                const auto p1v = g.psi1(v);
                const auto p2v = g.psi2(v);

                // ⟨ψ_v, ψ_w⟩_ℂ = conj(ψ₁_v) · ψ₁_w + conj(ψ₂_v) · ψ₂_w.
                const auto inner = [&](std::size_t w) {
                    return std::conj(p1v) * g.psi1(w)
                         + std::conj(p2v) * g.psi2(w);
                };

                eta_x[v] = std::arg(inner(nxi));
                eta_y[v] = std::arg(inner(nyi));
                eta_z[v] = std::arg(inner(nzi));
            }
        }
    }
}

/**
 * @brief Compute vertex divergence ξ̃ from per-edge η̃ arrays via Eq. 5.
 *
 * On a regular Cartesian grid the combinatorial weights collapse to
 * the backward finite-difference with 1/dx² scaling on each axis.
 */
void compute_divergence_from_edges(const Grid3D& g,
                                   const std::vector<double>& eta_x,
                                   const std::vector<double>& eta_y,
                                   const std::vector<double>& eta_z,
                                   std::vector<double>& xi)
{
    const int nx = g.nx();
    const int ny = g.ny();
    const int nz = g.nz();

    const double inv_dx2 = 1.0 / (g.dx() * g.dx());
    const double inv_dy2 = 1.0 / (g.dy() * g.dy());
    const double inv_dz2 = 1.0 / (g.dz() * g.dz());

    if (xi.size() != g.size()) xi.assign(g.size(), 0.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t v   = g.index(i, j, k);
                // "Incoming" neighbours along each axis (periodic wrap).
                const std::size_t pxi = g.wrap(i - 1, j,     k    );
                const std::size_t pyi = g.wrap(i,     j - 1, k    );
                const std::size_t pzi = g.wrap(i,     j,     k - 1);

                xi[v] = (eta_x[v] - eta_x[pxi]) * inv_dx2
                      + (eta_y[v] - eta_y[pyi]) * inv_dy2
                      + (eta_z[v] - eta_z[pzi]) * inv_dz2;
            }
        }
    }
}

/**
 * @brief Solve Δq̃ = ξ̃ on the periodic grid via FFT.
 *
 * Uses the *discrete* Laplacian eigenvalues of Eq. 17 so the solve
 * exactly inverts the finite-difference operator used above. The DC
 * mode is set to zero (the Poisson operator is singular on constants).
 *
 * Input `xi` is real; output `q` is real. The complex FFT scratch
 * buffer and per-axis eigenvalue tables are passed in by reference so
 * they can be reused across calls without reallocation.
 */
void solve_poisson_fft(const Grid3D& g,
                       const std::vector<double>& xi,
                       std::vector<double>& q,
                       std::vector<std::complex<double>>& buf,
                       std::vector<double>& sx,
                       std::vector<double>& sy,
                       std::vector<double>& sz)
{
    const int nx = g.nx();
    const int ny = g.ny();
    const int nz = g.nz();
    const std::size_t N = g.size();

    if (buf.size() != N) buf.assign(N, { 0.0, 0.0 });
    if (q.size()   != N) q.assign(N, 0.0);

    // Pack real ξ̃ into the complex scratch buffer (imag = 0).
    for (std::size_t n = 0; n < N; ++n) {
        buf[n] = { xi[n], 0.0 };
    }

    fft3_forward(buf.data(), nx, ny, nz);

    // Eigenvalues of the discrete 7-point Laplacian (Eq. 17).
    const double pi = std::numbers::pi_v<double>;
    const double inv_dx2 = 1.0 / (g.dx() * g.dx());
    const double inv_dy2 = 1.0 / (g.dy() * g.dy());
    const double inv_dz2 = 1.0 / (g.dz() * g.dz());

    if (static_cast<int>(sx.size()) != nx) sx.resize(nx);
    if (static_cast<int>(sy.size()) != ny) sy.resize(ny);
    if (static_cast<int>(sz.size()) != nz) sz.resize(nz);

    for (int i = 0; i < nx; ++i) {
        const double s = std::sin(pi * i / nx);
        sx[i] = -4.0 * inv_dx2 * s * s;
    }
    for (int j = 0; j < ny; ++j) {
        const double s = std::sin(pi * j / ny);
        sy[j] = -4.0 * inv_dy2 * s * s;
    }
    for (int k = 0; k < nz; ++k) {
        const double s = std::sin(pi * k / nz);
        sz[k] = -4.0 * inv_dz2 * s * s;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = g.index(i, j, k);
                const double lambda = sx[i] + sy[j] + sz[k];
                if (lambda != 0.0) {
                    buf[idx] /= lambda;
                } else {
                    buf[idx] = { 0.0, 0.0 };
                }
            }
        }
    }

    fft3_inverse(buf.data(), nx, ny, nz);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t n = 0; n < static_cast<std::ptrdiff_t>(N); ++n) {
        q[static_cast<std::size_t>(n)] =
            buf[static_cast<std::size_t>(n)].real();     // imag part is round-off
    }
}

} // namespace

std::vector<double> compute_scaled_divergence(const Grid3D& g)
{
    std::vector<double> eta_x, eta_y, eta_z;
    compute_scaled_edge_velocity(g, eta_x, eta_y, eta_z);

    std::vector<double> xi;
    compute_divergence_from_edges(g, eta_x, eta_y, eta_z, xi);
    return xi;
}

void pressure_project(Grid3D& g, PressureWorkspace& ws)
{
    // Step 1 + 2: η̃ on edges, then ξ̃ at vertices.
    compute_scaled_edge_velocity(g, ws.eta_x, ws.eta_y, ws.eta_z);
    compute_divergence_from_edges(g, ws.eta_x, ws.eta_y, ws.eta_z, ws.xi);

    // Step 3: Poisson solve Δq̃ = ξ̃ (in scaled units).
    solve_poisson_fft(g, ws.xi, ws.q, ws.fft_buf, ws.sx, ws.sy, ws.sz);

    // Step 4: pointwise phase correction ψ ← e^(−iq̃) · ψ.
    const std::ptrdiff_t Nv = static_cast<std::ptrdiff_t>(g.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t n = 0; n < Nv; ++n) {
        const std::size_t i = static_cast<std::size_t>(n);
        const double phi = -ws.q[i];
        const std::complex<double> rot{ std::cos(phi), std::sin(phi) };
        g.psi1(i) *= rot;
        g.psi2(i) *= rot;
    }
}

void pressure_project(Grid3D& g)
{
    // Convenience overload — allocates a fresh workspace each call.
    // Used by tests and any client that doesn't care about per-call
    // allocation cost.
    PressureWorkspace ws;
    pressure_project(g, ws);
}

} // namespace isf
