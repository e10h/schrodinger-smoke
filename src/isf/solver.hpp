/**
 * @file solver.hpp
 * @brief The basic ISF time step — Algorithm 1.
 *
 * Composition of the three primitive operators that make up one
 * incompressible Schrödinger flow step:
 *
 *     ψ ← Schrödinger(ψ, dt, ℏ)    — Alg. 2, free-particle evolution
 *     ψ ← Normalize(ψ)             — pointwise |ψ| ← 1
 *     ψ ← PressureProject(ψ)       — Alg. 3, divergence-free projection
 *
 * Call `solve_step` once per time step in the main loop.
 */

#pragma once

namespace isf {

class Grid3D;

/**
 * @brief Advance ψ by one basic ISF step (Algorithm 1).
 *
 * @param g    Grid holding ψ; mutated in place.
 * @param dt   Time step.
 * @param hbar Planck-like constant ℏ. Controls vortex-filament quantization
 *             strength (2πℏ) and is the only physical knob in basic ISF.
 */
void solve_step(Grid3D& g, double dt, double hbar);

} // namespace isf
