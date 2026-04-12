/**
 * @file constraint.hpp
 * @brief Velocity-constraint projection (Algorithm 4).
 *
 * Implements Algorithm 4 of Chern et al. 2016, which prescribes a
 * fixed velocity inside a region Ω ⊂ M while keeping the global ψ
 * divergence-free. Two pieces:
 *
 *   - `apply_velocity_constraint(g, region, k)` — pure pointwise
 *     update inside `region`. Replaces ψ's S³ direction with the
 *     plane-wave phase ⟨k,x⟩, leaving the per-component magnitudes
 *     |ψ₁|, |ψ₂| untouched.
 *
 *   - `constraint_project(g, region, k, ws)` — Algorithm 4 itself:
 *     `apply_velocity_constraint` followed by `pressure_project`.
 *
 * Iterating `constraint_project` 5–10 times converges so the velocity
 * inside the region matches ℏk while the field outside accommodates
 * the global div-free constraint.
 *
 * For multi-region setups (e.g. two balls of dye pointing at each
 * other), call `apply_velocity_constraint` once per region with the
 * matching k, then call `pressure_project` once. Repeat that pair as
 * the iteration. The function `init_two_velocity_balls` in init.hpp
 * does exactly this for the canonical colliding-spheres demo.
 */

#pragma once

#include <functional>

namespace isf {

class Grid3D;
struct PressureWorkspace;

/**
 * @brief Region predicate type: (i, j, k) → bool.
 *
 * `true` means "this vertex is inside the region", `false` "outside".
 */
using RegionMask = std::function<bool(int, int, int)>;

/**
 * @brief Pointwise: replace ψ inside `region` with a plane-wave-direction
 *        copy of itself.
 *
 * For every vertex v ∈ region, set
 *
 *     ψ₁(v) ← |ψ₁(v)| · e^(i⟨k,x_v⟩)
 *     ψ₂(v) ← |ψ₂(v)| · e^(i⟨k,x_v⟩)
 *
 * The per-component magnitudes are preserved, so `|ψ|² = |ψ₁|² + |ψ₂|²`
 * stays the same — if it was 1 going in, it's still 1 going out.
 *
 * @param g       Grid holding ψ; mutated in place.
 * @param region  Predicate selecting which vertices to constrain.
 * @param kx,ky,kz Target wave vector. The induced velocity inside the
 *                 region is approximately ℏk after one full iteration
 *                 of `constraint_project`.
 */
void apply_velocity_constraint(Grid3D& g,
                               const RegionMask& region,
                               double kx, double ky, double kz);

/**
 * @brief One iteration of Algorithm 4 (constraint projection).
 *
 * Equivalent to:
 *
 *     apply_velocity_constraint(g, region, kx, ky, kz);
 *     pressure_project(g, ws);
 *
 * Iterate 5–10 times until the velocity inside the region matches
 * the target ℏk to the desired tolerance.
 *
 * @param g        Grid holding ψ.
 * @param region   Predicate selecting the constrained region.
 * @param kx,ky,kz Target wave vector.
 * @param ws       Reusable PressureWorkspace.
 */
void constraint_project(Grid3D& g,
                        const RegionMask& region,
                        double kx, double ky, double kz,
                        PressureWorkspace& ws);

} // namespace isf
