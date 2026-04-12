include(Catch)

# Each TDD cycle adds its test executable here.

add_executable(test_grid test_grid.cpp)
target_link_libraries(test_grid PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_grid)

add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_fft)

add_executable(test_schrodinger test_schrodinger.cpp)
target_link_libraries(test_schrodinger PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_schrodinger)

add_executable(test_normalize test_normalize.cpp)
target_link_libraries(test_normalize PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_normalize)

add_executable(test_pressure test_pressure.cpp)
target_link_libraries(test_pressure PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_pressure)

add_executable(test_init_plane_wave test_init_plane_wave.cpp)
target_link_libraries(test_init_plane_wave PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_init_plane_wave)

add_executable(test_solver test_solver.cpp)
target_link_libraries(test_solver PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_solver)

add_executable(test_particles test_particles.cpp)
target_link_libraries(test_particles PRIVATE viz Catch2::Catch2WithMain)
catch_discover_tests(test_particles)

if(TARGET glm::glm)
    add_executable(test_camera test_camera.cpp)
    target_link_libraries(test_camera PRIVATE viz glm::glm Catch2::Catch2WithMain)
    catch_discover_tests(test_camera)
endif()

add_executable(test_constraint test_constraint.cpp)
target_link_libraries(test_constraint PRIVATE isf Catch2::Catch2WithMain)
catch_discover_tests(test_constraint)
