/**
 * @file normalize.cpp
 * @brief Implementation of the pointwise ψ normalization step.
 */

#include "isf/normalize.hpp"

#include "isf/grid.hpp"

#include <cmath>
#include <complex>

namespace isf {

void normalize(Grid3D& g)
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(g.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t idx = 0; idx < n; ++idx) {
        const std::size_t i = static_cast<std::size_t>(idx);
        const double mod_sq = std::norm(g.psi1(i)) + std::norm(g.psi2(i));
        if (mod_sq > 0.0) {
            const double inv = 1.0 / std::sqrt(mod_sq);
            g.psi1(i) *= inv;
            g.psi2(i) *= inv;
        }
    }
}

} // namespace isf
