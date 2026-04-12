/**
 * @file grid.hpp
 * @brief Regular 3D lattice storage for the two-component wave function ψ.
 *
 * The Incompressible Schrödinger Flow algorithm of Chern et al. (SIGGRAPH
 * 2016) evolves a wave function
 *
 *     ψ = (ψ₁, ψ₂) : M → ℂ²
 *
 * sampled at the vertices V = {0,…,Nx−1} × {0,…,Ny−1} × {0,…,Nz−1} of a
 * regular 3D lattice over a rectangular domain of physical size
 * (Lx, Ly, Lz). With periodic boundary conditions — the default — the
 * number of vertices equals the number of edges per axis, so the grid
 * spacing is `dx = Lx / Nx` (and likewise for y, z).
 */

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace isf {

/**
 * @brief Regular 3D lattice storing ψ₁ and ψ₂ at vertices.
 *
 * Storage is row-major with the x-axis varying fastest so that FFTs along x
 * access contiguous memory:
 *
 *     index(i, j, k) = i + Nx * (j + Ny * k)
 *
 * Both ψ components share this layout but live in separate contiguous
 * buffers so each can be handed directly to a complex-to-complex FFT.
 *
 * All wrap-around operations treat the grid as periodic with period Nx,
 * Ny, Nz along the three axes respectively.
 */
class Grid3D {
public:
    /**
     * @brief Construct a grid with the given vertex counts and physical size.
     *
     * ψ₁ and ψ₂ are zero-initialized.
     *
     * @param nx Number of vertices along x (must be ≥ 1).
     * @param ny Number of vertices along y (must be ≥ 1).
     * @param nz Number of vertices along z (must be ≥ 1).
     * @param lx Physical extent along x.
     * @param ly Physical extent along y.
     * @param lz Physical extent along z.
     */
    Grid3D(int nx, int ny, int nz, double lx, double ly, double lz);

    /// @name Lattice dimensions
    /// @{
    [[nodiscard]] int nx() const noexcept { return nx_; }
    [[nodiscard]] int ny() const noexcept { return ny_; }
    [[nodiscard]] int nz() const noexcept { return nz_; }
    /// @}

    /// @name Physical domain extents
    /// @{
    [[nodiscard]] double lx() const noexcept { return lx_; }
    [[nodiscard]] double ly() const noexcept { return ly_; }
    [[nodiscard]] double lz() const noexcept { return lz_; }
    /// @}

    /// @name Grid spacings (Lᵢ / Nᵢ for a periodic grid)
    /// @{
    [[nodiscard]] double dx() const noexcept { return lx_ / nx_; }
    [[nodiscard]] double dy() const noexcept { return ly_ / ny_; }
    [[nodiscard]] double dz() const noexcept { return lz_ / nz_; }
    /// @}

    /** @brief Total vertex count Nx·Ny·Nz. */
    [[nodiscard]] std::size_t size() const noexcept {
        return static_cast<std::size_t>(nx_) *
               static_cast<std::size_t>(ny_) *
               static_cast<std::size_t>(nz_);
    }

    /**
     * @brief Linear index for in-range integer coordinates.
     *
     * No bounds checking and no wrapping: callers are responsible for
     * ensuring 0 ≤ i < Nx, 0 ≤ j < Ny, 0 ≤ k < Nz.
     */
    [[nodiscard]] std::size_t index(int i, int j, int k) const noexcept {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(nx_) *
                   (static_cast<std::size_t>(j) +
                    static_cast<std::size_t>(ny_) *
                        static_cast<std::size_t>(k));
    }

    /**
     * @brief Linear index for possibly out-of-range coordinates,
     *        wrapped periodically.
     *
     * Any integer triple (i, j, k) is mapped into the fundamental domain
     * [0, Nx) × [0, Ny) × [0, Nz) using a true modulo (negative arguments
     * wrap to the positive side).
     */
    [[nodiscard]] std::size_t wrap(int i, int j, int k) const noexcept {
        const int iw = pmod(i, nx_);
        const int jw = pmod(j, ny_);
        const int kw = pmod(k, nz_);
        return index(iw, jw, kw);
    }

    /// @name ψ₁ / ψ₂ accessors
    /// @{

    /** @brief Mutable reference to ψ₁ at a linear index. */
    [[nodiscard]] std::complex<double>& psi1(std::size_t idx) noexcept {
        return psi1_[idx];
    }
    /** @brief Const reference to ψ₁ at a linear index. */
    [[nodiscard]] const std::complex<double>&
    psi1(std::size_t idx) const noexcept {
        return psi1_[idx];
    }

    /** @brief Mutable reference to ψ₁ at (i, j, k). */
    [[nodiscard]] std::complex<double>& psi1(int i, int j, int k) noexcept {
        return psi1_[index(i, j, k)];
    }
    /** @brief Const reference to ψ₁ at (i, j, k). */
    [[nodiscard]] const std::complex<double>&
    psi1(int i, int j, int k) const noexcept {
        return psi1_[index(i, j, k)];
    }

    /** @brief Mutable reference to ψ₂ at a linear index. */
    [[nodiscard]] std::complex<double>& psi2(std::size_t idx) noexcept {
        return psi2_[idx];
    }
    /** @brief Const reference to ψ₂ at a linear index. */
    [[nodiscard]] const std::complex<double>&
    psi2(std::size_t idx) const noexcept {
        return psi2_[idx];
    }

    /** @brief Mutable reference to ψ₂ at (i, j, k). */
    [[nodiscard]] std::complex<double>& psi2(int i, int j, int k) noexcept {
        return psi2_[index(i, j, k)];
    }
    /** @brief Const reference to ψ₂ at (i, j, k). */
    [[nodiscard]] const std::complex<double>&
    psi2(int i, int j, int k) const noexcept {
        return psi2_[index(i, j, k)];
    }

    /** @brief Contiguous pointer to ψ₁ storage (row-major, x fastest). */
    [[nodiscard]] std::complex<double>* psi1_data() noexcept {
        return psi1_.data();
    }
    [[nodiscard]] const std::complex<double>* psi1_data() const noexcept {
        return psi1_.data();
    }

    /** @brief Contiguous pointer to ψ₂ storage (row-major, x fastest). */
    [[nodiscard]] std::complex<double>* psi2_data() noexcept {
        return psi2_.data();
    }
    [[nodiscard]] const std::complex<double>* psi2_data() const noexcept {
        return psi2_.data();
    }

    /// @}

private:
    /**
     * @brief True modulo that always returns a value in [0, n).
     *
     * Unlike the built-in `%` operator, `pmod(-1, 4) == 3`.
     */
    static constexpr int pmod(int x, int n) noexcept {
        const int r = x % n;
        return r < 0 ? r + n : r;
    }

    int nx_;
    int ny_;
    int nz_;
    double lx_;
    double ly_;
    double lz_;
    std::vector<std::complex<double>> psi1_;
    std::vector<std::complex<double>> psi2_;
};

} // namespace isf
