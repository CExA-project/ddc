# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

add_executable(ddc_benchmark_deepcopy deepcopy.cpp)
target_link_libraries(ddc_benchmark_deepcopy PUBLIC benchmark::benchmark DDC::core)

if("${DDC_BUILD_KERNELS_SPLINES}")
    add_executable(ddc_benchmark_splines splines.cpp)
    target_link_libraries(ddc_benchmark_splines PUBLIC benchmark::benchmark DDC::core DDC::splines)
endif()
