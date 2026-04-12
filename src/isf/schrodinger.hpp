/**
 * @file schrodinger.hpp
 * @brief Free-particle Schrödinger evolution step (Alg. 2).
 *
 * Implements Algorithm 2 of
 *
 *   Chern, Knöppel, Pinkall, Schröder, Weißmann.
 *   "Schrödinger's Smoke." ACM Trans. Graph. 35(4), SIGGRAPH 2016.
 *
 * The step integrates
 *
 *     i ℏ ∂ₜψ = −(ℏ²/2) Δψ
 *
 * exactly in Fourier space over a time interval `dt`. Each component of
 * ψ = (ψ₁, ψ₂) is propagated independently, as the free Schrödinger
 * equation is diagonal on the two components.
 *
 * In the Fourier basis the Laplacian is diagonal with eigenvalues (Eq. 18)
 *
 *     λ_k = −(2π)² · (k_x²/L_x² + k_y²/L_y² + k_z²/L_z²),
 *
 * so the analytic propagator is a pointwise multiplication by
 *
 *     exp(i · ℏ · λ_k · dt / 2).
 *
 * Because λ_k is real-valued the propagator lies on the unit circle, and
 * the step is unitary (conserves the discrete L² norm to round-off).
 */

#pragma once

namespace isf {

class Grid3D;

/**
 * @brief In-place Schrödinger evolution of both ψ components by one dt.
 *
 * Corresponds exactly to Algorithm 2 in Chern et al. 2016.
 *
 * @param g    Grid holding ψ₁ and ψ₂; both components are updated in place.
 * @param dt   Time step.
 * @param hbar Planck-like constant ℏ. In ISF this is the only physical knob;
 *             smaller values yield thinner vortex filaments.
 */
void schrodinger_step(Grid3D& g, double dt, double hbar);

} // namespace isf
