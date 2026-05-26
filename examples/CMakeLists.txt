# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_executable(game_of_life game_of_life.cpp)
target_link_libraries(game_of_life PUBLIC DDC::core)
add_test(NAME Examples.GameOfLife COMMAND game_of_life)

add_executable(heat_equation heat_equation.cpp)
target_link_libraries(heat_equation PUBLIC DDC::core)
add_test(NAME Examples.HeatEquation COMMAND heat_equation)

add_executable(uniform_heat_equation uniform_heat_equation.cpp)
target_link_libraries(uniform_heat_equation PUBLIC DDC::core)
add_test(NAME Examples.UniformHeatEquation COMMAND uniform_heat_equation)

add_executable(non_uniform_heat_equation non_uniform_heat_equation.cpp)
target_link_libraries(non_uniform_heat_equation PUBLIC DDC::core)
add_test(NAME Examples.NonUniformHeatEquation COMMAND non_uniform_heat_equation)

if("${DDC_BUILD_PDI_WRAPPER}")
    add_executable(pdi_event pdi_event.cpp)
    target_link_libraries(pdi_event PUBLIC DDC::core DDC::pdi)
    add_test(NAME Examples.PdiEvent COMMAND pdi_event)
endif()

if("${DDC_BUILD_KERNELS_FFT}")
    add_executable(heat_equation_spectral heat_equation_spectral.cpp)
    target_link_libraries(heat_equation_spectral PUBLIC DDC::core DDC::fft)
    add_test(NAME Examples.HeatEquationSpectral COMMAND heat_equation_spectral)
endif()

if("${DDC_BUILD_KERNELS_SPLINES}")
    add_executable(characteristics_advection characteristics_advection.cpp)
    target_link_libraries(characteristics_advection PUBLIC DDC::core DDC::splines)
    add_test(NAME Examples.CharacteristicsAdvection COMMAND characteristics_advection)
endif()
