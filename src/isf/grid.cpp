/**
 * @file grid.cpp
 * @brief Out-of-line constructor for isf::Grid3D.
 *
 * Kept in a .cpp so the allocation of the ψ buffers does not inline into
 * every translation unit that just needs the indexing helpers.
 */

#include "isf/grid.hpp"

namespace isf {

Grid3D::Grid3D(int nx, int ny, int nz, double lx, double ly, double lz)
    : nx_(nx),
      ny_(ny),
      nz_(nz),
      lx_(lx),
      ly_(ly),
      lz_(lz),
      psi1_(static_cast<std::size_t>(nx) *
            static_cast<std::size_t>(ny) *
            static_cast<std::size_t>(nz),
            std::complex<double>(0.0, 0.0)),
      psi2_(static_cast<std::size_t>(nx) *
            static_cast<std::size_t>(ny) *
            static_cast<std::size_t>(nz),
            std::complex<double>(0.0, 0.0)) {}

} // namespace isf
