# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

add_executable(game_of_life game_of_life.cpp)
target_link_libraries(game_of_life PUBLIC DDC::core)

add_executable(heat_equation heat_equation.cpp)
target_link_libraries(heat_equation PUBLIC DDC::core)

add_executable(uniform_heat_equation uniform_heat_equation.cpp)
target_link_libraries(uniform_heat_equation PUBLIC DDC::core)

add_executable(non_uniform_heat_equation non_uniform_heat_equation.cpp)
target_link_libraries(non_uniform_heat_equation PUBLIC DDC::core)

if("${DDC_BUILD_PDI_WRAPPER}")
    add_executable(pdi_event pdi_event.cpp)
    target_link_libraries(pdi_event PUBLIC DDC::core DDC::pdi)
endif()

if("${DDC_BUILD_KERNELS_FFT}")
    add_executable(heat_equation_spectral heat_equation_spectral.cpp)
    target_link_libraries(heat_equation_spectral PUBLIC DDC::core DDC::fft)
endif()

if("${DDC_BUILD_KERNELS_SPLINES}")
    add_executable(characteristics_advection characteristics_advection.cpp)
    target_link_libraries(characteristics_advection PUBLIC DDC::core DDC::splines)
endif()
