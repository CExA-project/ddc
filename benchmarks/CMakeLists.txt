# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

## Google Benchmark
find_package(benchmark 1.8 REQUIRED)

add_executable(ddc_benchmark_deepcopy deepcopy.cpp)
target_link_libraries(ddc_benchmark_deepcopy PUBLIC benchmark::benchmark DDC::core)

if("${DDC_BUILD_KERNELS_SPLINES}")
    add_executable(ddc_benchmark_splines splines.cpp)
    target_link_libraries(ddc_benchmark_splines PUBLIC benchmark::benchmark DDC::core DDC::splines)
endif()
