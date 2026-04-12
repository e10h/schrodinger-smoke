/**
 * @file init.hpp
 * @brief Initial-condition builders for ISF.
 *
 * Each builder writes ψ = (ψ₁, ψ₂) directly into a Grid3D. Clients are
 * expected to apply pressure projection after initialization if the
 * condition is not already divergence-free (plane waves and the
 * paper's slab-based vortex ring both are, but constructing
 * multi-region initial data generally is not — see Alg. 4 in Sec. 3.2).
 */

#pragma once

namespace isf {

class Grid3D;

/**
 * @brief Initialize ψ to the plane-wave state ψ = (exp(i⟨k,x⟩), 0)ᵀ.
 *
 * Under ISF this corresponds to a uniform velocity field v = ℏk
 * (Sec. 3.2 of Chern et al. 2016). For the plane wave to be exactly
 * periodic on the discrete grid, each k component must be an integer
 * multiple of 2π/L along the corresponding axis.
 *
 * The result is automatically divergence-free (uniform flow) and has
 * |ψ|² = 1 pointwise, so no normalization or projection is required.
 *
 * @param g  Grid to fill.
 * @param kx x-component of the wave vector.
 * @param ky y-component of the wave vector.
 * @param kz z-component of the wave vector.
 */
void init_plane_wave(Grid3D& g, double kx, double ky, double kz);

/**
 * @brief Initialize a single circular vortex ring in the plane z = cz.
 *
 * Implements the "slab" construction of Section 3.1 of Chern et al.
 * 2016. Σ is the disk of radius R at height cz whose boundary is the
 * ring γ. Inside the disk-shaped slab
 *
 *     { (x,y,z) : (x−cx)² + (y−cy)² < R²  and  |z−cz| < r }
 *
 * the phase field is set to
 *
 *     θ(x,y,z) = π · (1 + (z − cz) / r)
 *
 * and outside the slab θ = 0. We start from ψ₁ = 1, ψ₂ = ε, then
 * multiply ψ₁ ← ψ₁ · exp(iθ), normalize pointwise to |ψ|² = 1, and
 * pressure-project (Alg. 3) so the result is immediately a valid ISF
 * state.
 *
 * The 2π phase jump across the slab boundary manifests as a thin
 * vortex filament along γ carrying strength 2πℏ — a single quantized
 * vortex ring.
 *
 * @param g        Grid to fill.
 * @param cx,cy,cz Center of the ring in physical coordinates.
 * @param R        Ring radius.
 * @param r        Slab half-thickness (sets filament radius; must be > 0
 *                 and larger than dz for the slab to be resolvable).
 * @param epsilon  Small "guard" value for ψ₂ to avoid isolated zeros of ψ;
 *                 0.01 is the paper's recommended default (Sec. 3).
 */
void init_vortex_ring_z(Grid3D& g,
                        double cx, double cy, double cz,
                        double R, double r,
                        double epsilon = 0.01);

/**
 * @brief Initialize two ball-shaped regions with prescribed velocities.
 *
 * The canonical "two ink regions colliding" demo from Sec. 3.2 / Fig. 6
 * of Chern et al. 2016 (their Bunny + Teapot example with simpler
 * geometry). Each ball is a sphere centred at (cα, cα, cα) with radius
 * R. Velocity v₁ is enforced inside ball 1, v₂ inside ball 2.
 *
 * The construction is the iterated multi-region Algorithm 4:
 *
 *   1. ψ ← (1, ε)ᵀ everywhere; normalize.
 *   2. Repeat `iters` times:
 *        a. apply_velocity_constraint inside ball 1 with k₁ = v₁/ℏ
 *        b. apply_velocity_constraint inside ball 2 with k₂ = v₂/ℏ
 *        c. pressure_project the whole grid.
 *
 * 5–10 iterations converge to a stable fixed point. The flow inside
 * each ball runs in the prescribed direction; its magnitude follows
 * the geometry-dependent fixed-point scaling discussed in
 * constraint.hpp.
 *
 * @param g    Grid to fill.
 * @param cx1,cy1,cz1 Centre of ball 1.
 * @param cx2,cy2,cz2 Centre of ball 2.
 * @param R    Radius shared by both balls.
 * @param vx1,vy1,vz1 Target velocity vector inside ball 1.
 * @param vx2,vy2,vz2 Target velocity vector inside ball 2.
 * @param hbar ℏ used to convert v into k = v/ℏ.
 * @param iters Number of constraint-projection iterations (default 8).
 * @param epsilon Small ψ₂ guard for the initial guess (default 0.01).
 */
void init_two_velocity_balls(Grid3D& g,
                             double cx1, double cy1, double cz1,
                             double cx2, double cy2, double cz2,
                             double R,
                             double vx1, double vy1, double vz1,
                             double vx2, double vy2, double vz2,
                             double hbar,
                             int    iters   = 8,
                             double epsilon = 0.01);

} // namespace isf
