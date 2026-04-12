/**
 * @file fft.hpp
 * @brief Thin wrapper around pocketfft for in-place 3D complex FFTs.
 *
 * The wrapper hides pocketfft's shape/stride/axes plumbing behind a
 * four-argument call that takes a contiguous row-major (x-fastest)
 * ℂ buffer and the three lattice dimensions. This keeps the rest of
 * the solver (Schrödinger step, pressure projection) backend-agnostic
 * so the CPU FFT can be swapped for a compute-shader FFT in Phase 2
 * with only this file changing.
 *
 * Sign and normalization conventions match numpy / pocketfft:
 *
 *   forward:  X_m = Σ_x  x(n) · exp(-2πi · m · n / N)
 *   inverse:  x_n = (1/N) · Σ_m  X(m) · exp(+2πi · m · n / N)
 *
 * With these conventions `fft3_forward` followed by `fft3_inverse` is
 * the identity (up to floating-point round-off).
 */

#pragma once

#include <complex>

namespace isf {

/**
 * @brief In-place forward 3D complex FFT over a row-major grid.
 *
 * The buffer is treated as a three-dimensional array of shape
 * (Nx, Ny, Nz) laid out so that x varies fastest (matching
 * Grid3D::index). Normalization factor is 1.
 *
 * @param data Pointer to Nx·Ny·Nz contiguous complex doubles.
 * @param nx   Number of samples along x (fastest axis).
 * @param ny   Number of samples along y.
 * @param nz   Number of samples along z (slowest axis).
 */
void fft3_forward(std::complex<double>* data, int nx, int ny, int nz);

/**
 * @brief In-place inverse 3D complex FFT over a row-major grid.
 *
 * Applies the (1/N)-scaled inverse so that `fft3_forward` followed by
 * `fft3_inverse` is the identity.
 *
 * @param data Pointer to Nx·Ny·Nz contiguous complex doubles.
 * @param nx   Number of samples along x (fastest axis).
 * @param ny   Number of samples along y.
 * @param nz   Number of samples along z (slowest axis).
 */
void fft3_inverse(std::complex<double>* data, int nx, int ny, int nz);

} // namespace isf
