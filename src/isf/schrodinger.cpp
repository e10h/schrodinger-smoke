/**
 * @file schrodinger.cpp
 * @brief Implementation of Algorithm 2 (free-particle Schrödinger step).
 *
 * Each component of ψ is transformed to the Fourier basis, multiplied
 * pointwise by the analytic propagator
 *
 *     exp(i · ℏ · λ_k · dt / 2)
 *
 * and transformed back. The Fourier-bin index convention follows
 * numpy / pocketfft:
 *
 *     bin 0         ↔ k =  0                (DC)
 *     bin 1         ↔ k = +1
 *     …
 *     bin ⌊N/2⌋     ↔ k = ±N/2              (Nyquist)
 *     bin ⌊N/2⌋+1   ↔ k = −(N/2 − 1)
 *     …
 *     bin N−1       ↔ k = −1
 *
 * Only k² matters here, so the Nyquist sign ambiguity is harmless.
 */

#include "isf/schrodinger.hpp"

#include "isf/fft.hpp"
#include "isf/grid.hpp"

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace isf {

namespace {

/**
 * @brief Map a non-negative bin index in [0, N) to the signed frequency
 *        k ∈ [−⌊N/2⌋, ⌈N/2⌉).
 */
constexpr int signed_freq(int bin, int n) noexcept
{
    return (bin <= n / 2) ? bin : bin - n;
}

/**
 * @brief Multiply ψ̂ (already in Fourier space) pointwise by
 *        exp(i · ℏ · λ_k · dt / 2).
 *
 * λ_k = −(2π)² · (k_x²/L_x² + k_y²/L_y² + k_z²/L_z²)  (Eq. 18).
 *
 * @param hat  Fourier-transformed component, row-major (x fastest).
 * @param nx,ny,nz  Grid dimensions.
 * @param lx,ly,lz  Physical domain extents.
 * @param dt        Time step.
 * @param hbar      ℏ.
 */
void apply_propagator(std::complex<double>* hat,
                      int nx, int ny, int nz,
                      double lx, double ly, double lz,
                      double dt, double hbar)
{
    const double two_pi = 2.0 * std::numbers::pi_v<double>;
    const double fx = two_pi / lx;
    const double fy = two_pi / ly;
    const double fz = two_pi / lz;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int kz = 0; kz < nz; ++kz) {
        for (int ky = 0; ky < ny; ++ky) {
            const int    sz_i  = signed_freq(kz, nz);
            const double kz_sq = (sz_i * fz) * (sz_i * fz);
            const int    sy_i  = signed_freq(ky, ny);
            const double ky_sq = (sy_i * fy) * (sy_i * fy);
            for (int kx = 0; kx < nx; ++kx) {
                const int    sx_i  = signed_freq(kx, nx);
                const double kx_sq = (sx_i * fx) * (sx_i * fx);

                // λ_k = −(k_x² + k_y² + k_z²)  with k_i in physical units.
                const double lambda = -(kx_sq + ky_sq + kz_sq);

                // Propagator phase  φ = ℏ · λ · dt / 2
                const double phi = hbar * lambda * dt / 2.0;
                const std::complex<double> rot{ std::cos(phi), std::sin(phi) };

                const std::size_t idx =
                    static_cast<std::size_t>(kx) +
                    static_cast<std::size_t>(nx) *
                        (static_cast<std::size_t>(ky) +
                         static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(kz));
                hat[idx] *= rot;
            }
        }
    }
}

} // namespace

void schrodinger_step(Grid3D& g, double dt, double hbar)
{
    const int nx = g.nx();
    const int ny = g.ny();
    const int nz = g.nz();
    const double lx = g.lx();
    const double ly = g.ly();
    const double lz = g.lz();

    // Component ψ₁.
    fft3_forward(g.psi1_data(), nx, ny, nz);
    apply_propagator(g.psi1_data(), nx, ny, nz, lx, ly, lz, dt, hbar);
    fft3_inverse(g.psi1_data(), nx, ny, nz);

    // Component ψ₂.
    fft3_forward(g.psi2_data(), nx, ny, nz);
    apply_propagator(g.psi2_data(), nx, ny, nz, lx, ly, lz, dt, hbar);
    fft3_inverse(g.psi2_data(), nx, ny, nz);
}

} // namespace isf
