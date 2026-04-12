/**
 * @file solver.cpp
 * @brief Implementation of the basic ISF step (Algorithm 1).
 */

#include "isf/solver.hpp"

#include "isf/grid.hpp"
#include "isf/normalize.hpp"
#include "isf/pressure.hpp"
#include "isf/schrodinger.hpp"

namespace isf {

void solve_step(Grid3D& g, double dt, double hbar)
{
    schrodinger_step(g, dt, hbar);  // Alg. 2
    normalize(g);                   // |ψ| ← 1
    pressure_project(g);            // Alg. 3
}

} // namespace isf
