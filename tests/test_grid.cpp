/**
 * @file test_grid.cpp
 * @brief Unit tests for isf::Grid3D — the ψ storage and indexing grid.
 *
 * The Grid3D type owns the two-component wave function ψ = (ψ₁, ψ₂) sampled
 * at vertices of a regular 3D lattice. These tests pin down the API the rest
 * of the solver will rely on: construction, dimensions, linear indexing,
 * periodic wrapping, and per-vertex accessors.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "isf/grid.hpp"

#include <complex>

using isf::Grid3D;
using Catch::Matchers::WithinAbs;

TEST_CASE("Grid3D stores the requested dimensions and physical size",
          "[grid]")
{
    Grid3D g(4, 5, 6, 1.0, 2.0, 3.0);

    REQUIRE(g.nx() == 4);
    REQUIRE(g.ny() == 5);
    REQUIRE(g.nz() == 6);

    REQUIRE_THAT(g.lx(), WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(g.ly(), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(g.lz(), WithinAbs(3.0, 1e-15));

    // Periodic grid: N vertices per axis with spacing L/N.
    REQUIRE_THAT(g.dx(), WithinAbs(1.0 / 4.0, 1e-15));
    REQUIRE_THAT(g.dy(), WithinAbs(2.0 / 5.0, 1e-15));
    REQUIRE_THAT(g.dz(), WithinAbs(3.0 / 6.0, 1e-15));

    REQUIRE(g.size() == static_cast<std::size_t>(4 * 5 * 6));
}

TEST_CASE("Grid3D uses row-major (x-fastest) linear indexing", "[grid]")
{
    Grid3D g(4, 5, 6, 1.0, 1.0, 1.0);

    // Convention: idx = i + Nx*(j + Ny*k)
    REQUIRE(g.index(0, 0, 0) == 0);
    REQUIRE(g.index(1, 0, 0) == 1);
    REQUIRE(g.index(0, 1, 0) == 4);
    REQUIRE(g.index(0, 0, 1) == 4 * 5);
    REQUIRE(g.index(3, 4, 5) == 3 + 4 * 4 + 5 * 4 * 5);
}

TEST_CASE("Grid3D::wrap handles positive and negative out-of-range indices",
          "[grid]")
{
    Grid3D g(4, 5, 6, 1.0, 1.0, 1.0);

    // In-range: identical to index().
    REQUIRE(g.wrap(2, 3, 4) == g.index(2, 3, 4));

    // Positive overflow wraps back to 0.
    REQUIRE(g.wrap(4, 0, 0) == g.index(0, 0, 0));
    REQUIRE(g.wrap(0, 5, 0) == g.index(0, 0, 0));
    REQUIRE(g.wrap(0, 0, 6) == g.index(0, 0, 0));

    // Negative underflow wraps to N-1.
    REQUIRE(g.wrap(-1, 0, 0) == g.index(3, 0, 0));
    REQUIRE(g.wrap(0, -1, 0) == g.index(0, 4, 0));
    REQUIRE(g.wrap(0, 0, -1) == g.index(0, 0, 5));

    // Multi-period wrap.
    REQUIRE(g.wrap(4 + 4 + 2, 0, 0) == g.index(2, 0, 0));
    REQUIRE(g.wrap(-4 - 1, 0, 0) == g.index(3, 0, 0));
}

TEST_CASE("Grid3D exposes ψ₁ and ψ₂ accessors that round-trip", "[grid]")
{
    Grid3D g(2, 2, 2, 1.0, 1.0, 1.0);

    // Default-constructed ψ is zero.
    REQUIRE(g.psi1(0) == std::complex<double>(0.0, 0.0));
    REQUIRE(g.psi2(0) == std::complex<double>(0.0, 0.0));

    // Set via (i,j,k) overload, read via linear index.
    g.psi1(1, 0, 1) = {1.5, -2.5};
    g.psi2(1, 0, 1) = {3.0, 4.0};

    const std::size_t idx = g.index(1, 0, 1);
    REQUIRE(g.psi1(idx) == std::complex<double>(1.5, -2.5));
    REQUIRE(g.psi2(idx) == std::complex<double>(3.0, 4.0));

    // Raw data pointers for FFT kernels.
    REQUIRE(g.psi1_data() != nullptr);
    REQUIRE(g.psi2_data() != nullptr);
    REQUIRE(g.psi1_data()[idx] == std::complex<double>(1.5, -2.5));
}
