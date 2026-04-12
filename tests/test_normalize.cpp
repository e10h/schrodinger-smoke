/**
 * @file test_normalize.cpp
 * @brief Unit tests for pointwise ψ normalization (line 3 of Alg. 1).
 *
 * Step 3 of the basic ISF loop enforces the incompressibility-adjacent
 * constraint |ψ|² = |ψ₁|² + |ψ₂|² = 1 at every lattice vertex. This is a
 * pure pointwise operation: ψ ← ψ / |ψ|.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"
#include "isf/normalize.hpp"

#include <cmath>
#include <complex>

using isf::Grid3D;
using isf::normalize;
using Catch::Matchers::WithinAbs;

TEST_CASE("normalize yields |ψ|² = 1 at every vertex", "[normalize]")
{
    Grid3D g(4, 4, 4, 1.0, 1.0, 1.0);

    // Non-unit, non-trivial ψ.
    for (int k = 0; k < g.nz(); ++k) {
        for (int j = 0; j < g.ny(); ++j) {
            for (int i = 0; i < g.nx(); ++i) {
                g.psi1(i, j, k) = { static_cast<double>(i + 1),
                                     static_cast<double>(j) };
                g.psi2(i, j, k) = { static_cast<double>(k),
                                    -static_cast<double>(j + 1) };
            }
        }
    }

    normalize(g);

    for (std::size_t n = 0; n < g.size(); ++n) {
        const double norm_sq = std::norm(g.psi1(n)) + std::norm(g.psi2(n));
        REQUIRE_THAT(norm_sq, WithinAbs(1.0, 1e-14));
    }
}

TEST_CASE("normalize preserves the direction of ψ", "[normalize]")
{
    Grid3D g(2, 2, 2, 1.0, 1.0, 1.0);

    // Set a single vertex with known components.
    g.psi1(0, 0, 0) = { 3.0, 0.0 };
    g.psi2(0, 0, 0) = { 0.0, 4.0 };
    // |ψ| = sqrt(9 + 16) = 5, so normalized values are (0.6, 0) and (0, 0.8).

    // Give all other vertices a unit value so they are untouched.
    for (std::size_t n = 1; n < g.size(); ++n) {
        g.psi1(n) = { 1.0, 0.0 };
        g.psi2(n) = { 0.0, 0.0 };
    }

    normalize(g);

    REQUIRE_THAT(g.psi1(0, 0, 0).real(), WithinAbs(0.6, 1e-15));
    REQUIRE_THAT(g.psi1(0, 0, 0).imag(), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(g.psi2(0, 0, 0).real(), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(g.psi2(0, 0, 0).imag(), WithinAbs(0.8, 1e-15));
}
