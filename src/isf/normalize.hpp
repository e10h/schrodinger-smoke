/**
 * @file normalize.hpp
 * @brief Pointwise normalization of ψ = (ψ₁, ψ₂) to unit modulus.
 *
 * Corresponds to line 3 of Algorithm 1 in Chern et al. 2016:
 *
 *     ψ ← ψ / |ψ|         where   |ψ|² = |ψ₁|² + |ψ₂|².
 *
 * This enforces the pointwise constraint |ψ|² = 1 that keeps ψ on the
 * manifold 𝓜 ⊂ C^∞(M, ℂ²) on which ISF is defined. The paper guards
 * against degenerate zeros by initializing ψ₂ to a small ε instead of
 * 0; we simply skip the division when the norm underflows so we never
 * introduce NaNs.
 */

#pragma once

namespace isf {

class Grid3D;

/**
 * @brief Normalize ψ pointwise so that |ψ₁|² + |ψ₂|² = 1 at every vertex.
 *
 * In-place; leaves the phase (direction on S³) of ψ unchanged.
 *
 * @param g Grid whose ψ components are updated.
 */
void normalize(Grid3D& g);

} // namespace isf
