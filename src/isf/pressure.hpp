/**
 * @file pressure.hpp
 * @brief Divergence-free constraint projection (Algorithm 3).
 *
 * Implements Algorithm 3 of Chern et al. 2016. The fluid is constrained
 * to have vanishing divergence of the velocity 1-form
 *
 *     η_vw = ℏ · arg⟨ψ_v, ψ_w⟩_ℂ                          (Eq. 4)
 *
 * via a phase correction on ψ. The algorithm operates in ℏ-scaled units
 * (η̃ = η/ℏ, q̃ = q/ℏ) so the phase correction is simply e^(−iq̃)·ψ —
 * ℏ never appears inside this step.
 *
 * All differential operators are discrete Exterior Calculus on a regular
 * periodic Cartesian grid:
 *
 *   - η̃ lives on directed edges, stored at the tail vertex in the
 *     "positive-axis" convention from Fig. 2 of the paper.
 *   - ξ̃ lives on vertices and is computed by the vertex-divergence
 *     formula (Eq. 5).
 *   - The Poisson problem Δq̃ = ξ̃ is solved by FFT using the exact
 *     *discrete* Laplacian eigenvalues (Eq. 17) so the projected ξ̃'
 *     is zero to round-off.
 */

#pragma once

#include <complex>
#include <vector>

namespace isf {

class Grid3D;

/**
 * @brief Reusable scratch buffers for pressure_project().
 *
 * Allocating ξ̃, η̃ on edges, q̃, and the FFT scratch buffer on every
 * call costs ~12 MB of malloc/free per frame at 64³. By holding the
 * workspace as a long-lived object owned by the caller (the viewer's
 * main loop, or the benchmark), each `pressure_project` call only
 * does in-place arithmetic on the existing storage.
 *
 * The workspace lazily resizes itself on first use and any subsequent
 * call where the grid size has changed; clients normally don't have
 * to touch it directly.
 */
struct PressureWorkspace {
    std::vector<double> eta_x;            ///< edge x-velocity 1-form
    std::vector<double> eta_y;            ///< edge y-velocity 1-form
    std::vector<double> eta_z;            ///< edge z-velocity 1-form
    std::vector<double> xi;               ///< vertex divergence
    std::vector<double> q;                ///< pressure potential
    std::vector<std::complex<double>>
                        fft_buf;          ///< complex FFT scratch
    std::vector<double> sx;               ///< per-axis Laplacian
    std::vector<double> sy;               ///< eigenvalue contributions
    std::vector<double> sz;
};

/**
 * @brief Compute the scaled divergence ξ̃ at every vertex of the grid.
 *
 * Useful both internally (Alg. 3 step 2) and as a diagnostic for tests.
 * The values returned are in scaled units (η̃ = arg⟨ψ_v, ψ_w⟩ without
 * the ℏ factor).
 *
 * @param g The grid holding ψ.
 * @return  Vector of length `g.size()` with ξ̃ stored in the same
 *          row-major (x-fastest) layout as ψ.
 */
[[nodiscard]] std::vector<double>
compute_scaled_divergence(const Grid3D& g);

/**
 * @brief In-place pressure projection of ψ (Algorithm 3).
 *
 * Enforces that the velocity 1-form recovered from ψ is divergence-free.
 * Mutates both ψ₁ and ψ₂ via a pointwise phase rotation by e^(−iq̃),
 * where q̃ is the solution of the discrete Poisson equation
 * Δq̃ = div η̃.
 *
 * Convenience overload — allocates a fresh PressureWorkspace internally
 * each call. Use the workspace overload below in tight loops to avoid
 * the per-call allocations.
 *
 * @param g The grid whose ψ is projected.
 */
void pressure_project(Grid3D& g);

/**
 * @brief Pressure projection using a caller-provided workspace.
 *
 * Functionally identical to the no-workspace version but reuses the
 * scratch buffers across calls. Lazily resizes the workspace if the
 * grid size has changed.
 *
 * @param g  The grid whose ψ is projected.
 * @param ws Reusable scratch buffers; persisted across calls.
 */
void pressure_project(Grid3D& g, PressureWorkspace& ws);

} // namespace isf
