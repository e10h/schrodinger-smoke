/**
 * @file fft.cpp
 * @brief Implementation of the isf::fft3_* wrappers over pocketfft.
 *
 * pocketfft's c2c() takes shape and stride vectors in numpy-style
 * slowest-first order, with strides expressed in bytes. For our
 * row-major (x-fastest) layout that means:
 *
 *   shape   = { Nz, Ny, Nx }
 *   stride  = { sizeof(ℂ) · Nx · Ny,   sizeof(ℂ) · Nx,   sizeof(ℂ) }
 *   axes    = { 0, 1, 2 }   // transform all three dimensions
 *
 * The backend is single-threaded here; multithreading can be enabled
 * later by passing a non-default `nthreads` argument to c2c().
 */

#include "isf/fft.hpp"

#include "pocketfft_hdronly.h"

#include <cstddef>
#include <thread>

namespace isf {

namespace {

/**
 * @brief Number of threads to ask pocketfft for.
 *
 * On Apple Silicon `hardware_concurrency()` reports the sum of
 * performance and efficiency cores. Threading the FFT across all of
 * them is fine for our 64³–128³ workloads; the cross-cluster cost is
 * smaller than the per-axis 1D FFTs we save.
 */
inline std::size_t fft_thread_count()
{
    const auto n = std::thread::hardware_concurrency();
    return n > 0 ? static_cast<std::size_t>(n) : 1;
}

} // namespace

namespace {

/**
 * @brief Fill numpy-style shape/stride arrays for a row-major
 *        (x-fastest) 3D buffer of complex doubles.
 */
void make_shape_and_stride(int nx, int ny, int nz,
                           pocketfft::shape_t& shape,
                           pocketfft::stride_t& stride)
{
    using Complex = std::complex<double>;
    shape  = { static_cast<std::size_t>(nz),
               static_cast<std::size_t>(ny),
               static_cast<std::size_t>(nx) };
    stride = { static_cast<std::ptrdiff_t>(sizeof(Complex) *
                   static_cast<std::size_t>(nx) *
                   static_cast<std::size_t>(ny)),
               static_cast<std::ptrdiff_t>(sizeof(Complex) *
                   static_cast<std::size_t>(nx)),
               static_cast<std::ptrdiff_t>(sizeof(Complex)) };
}

} // namespace

void fft3_forward(std::complex<double>* data, int nx, int ny, int nz)
{
    pocketfft::shape_t  shape;
    pocketfft::stride_t stride;
    make_shape_and_stride(nx, ny, nz, shape, stride);

    const pocketfft::shape_t axes{0, 1, 2};
    pocketfft::c2c(shape, stride, stride, axes,
                   pocketfft::FORWARD, data, data, 1.0,
                   fft_thread_count());
}

void fft3_inverse(std::complex<double>* data, int nx, int ny, int nz)
{
    pocketfft::shape_t  shape;
    pocketfft::stride_t stride;
    make_shape_and_stride(nx, ny, nz, shape, stride);

    const double inv_n =
        1.0 / (static_cast<double>(nx) *
               static_cast<double>(ny) *
               static_cast<double>(nz));
    const pocketfft::shape_t axes{0, 1, 2};
    pocketfft::c2c(shape, stride, stride, axes,
                   pocketfft::BACKWARD, data, data, inv_n,
                   fft_thread_count());
}

} // namespace isf
