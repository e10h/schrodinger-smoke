# Schrödinger's Smoke

A real-time C++20 implementation of *Incompressible Schrödinger Flow* (ISF)
from

> Chern, Knöppel, Pinkall, Schröder, Weißmann.
> **Schrödinger's Smoke.** *ACM Trans. Graph.* 35(4), SIGGRAPH 2016.
> [DOI: 10.1145/2897824.2925868](https://doi.org/10.1145/2897824.2925868)

The fluid state is a two-component complex wave function ψ : M → C² on a
periodic 3D lattice.  Velocity is recovered from the edge 1-form
η = ℏ · arg⟨ψ_v, ψ_w⟩ (Eq. 4).  Vorticity naturally concentrates into
quantized filaments of strength 2πℏ, giving leapfrogging rings,
reconnection events, and turbulent mixing — without Lagrangian tracking.

## Demos

- **Vortex ring** — single quantized filament via the slab construction
  (Sec. 3.1).  The ring propagates, sheds secondary filaments, and
  reconnects over time.
- **Colliding spheres** — two regions of dye initialized with opposing
  velocities (Sec. 3.2, Algorithm 4).  The inks collide, mix, and
  produce turbulent vortex structures.

## Building

Requires CMake ≥ 3.20, a C++20 compiler, GLFW 3, and GLM.
OpenMP is optional but recommended.

```sh
cmake -S . -B build
cmake --build build -j
```

Run the tests:

```sh
ctest --test-dir build --output-on-failure
```

Launch the viewer:

```sh
./build/schroedinger_viewer
```

## Solver

Implements Algorithms 1–4 of the paper:

| Step | Function | Method |
|------|----------|--------|
| Schrödinger integration | `schrodinger_step` | FFT spectral propagator (Alg. 2) |
| Normalization | `normalize` | Pointwise \|ψ\|² = 1 |
| Pressure projection | `pressure_project` | Discrete Poisson via FFT (Alg. 3, Eq. 17) |
| Velocity constraint | `apply_velocity_constraint` | Region phase replacement (Alg. 4) |
| Velocity recovery | `recover_velocity` | Edge arg averaging to vertices |

The pressure workspace caches the edge 1-form η and Poisson potential q,
allowing `recover_velocity_from_workspace` to derive the post-correction
velocity as η' = η − ∇q without recomputing inner products (~25% speedup
at 128³).

## Viewer

- OpenGL 4.1 core profile, GLFW, Dear ImGui
- Additive point-sprite particle renderer (150k tracers)
- Orbit camera with mouse control
- Runtime controls for ℏ, dt, resolution (48³–160³), particle count
- Built-in GIF recorder

## Performance

Benchmarked on Apple Silicon (single socket):

| Resolution | ms / frame | FPS |
|------------|-----------|-----|
| 64³        | 9 ms      | 109 |
| 96³        | 22 ms     | 45  |
| 128³       | 57 ms     | 18  |
| 160³       | 93 ms     | 11  |

## Layout

```
src/isf/       core solver (grid, FFT, Schrödinger, pressure, constraints, velocity)
src/viz/       viewer (window, camera, particle renderer, GIF recorder)
src/           main.cpp, bench_solver.cpp
tests/         37 Catch2 unit + integration tests
third_party/   pocketfft, msf_gif
```

## Third-party libraries

- [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) — Martin Reinecke (BSD-3)
- [msf_gif](https://github.com/notnullnotvoid/msf_gif) — Miles Fogle (MIT / Public Domain)
- [Dear ImGui](https://github.com/ocornut/imgui) — Omar Cornut (MIT), fetched by CMake
- [Catch2](https://github.com/catchorg/Catch2) — Catch2 authors (BSL-1.0), fetched by CMake

## License

Free for personal, academic, and research use. No warranty. Commercial use requires written permission from the author.
