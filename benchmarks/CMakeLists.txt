# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_executable(ddc_benchmark_deepcopy deepcopy.cpp)
target_link_libraries(ddc_benchmark_deepcopy PUBLIC benchmark::benchmark DDC::core)
add_test(
    NAME BenchmarksDryRun.Deepcopy
    COMMAND ddc_benchmark_deepcopy --benchmark_dry_run --benchmark_filter=.*_small
)

add_executable(ddc_benchmark_parallel_copy parallel_copy.cpp)
target_link_libraries(ddc_benchmark_parallel_copy PUBLIC benchmark::benchmark DDC::core)
add_test(
    NAME BenchmarksDryRun.ParallelCopy
    COMMAND ddc_benchmark_parallel_copy --benchmark_dry_run --benchmark_filter=.*_small
)

if("${DDC_BUILD_KERNELS_SPLINES}")
    add_executable(ddc_benchmark_splines splines.cpp)
    target_link_libraries(ddc_benchmark_splines PUBLIC benchmark::benchmark DDC::core DDC::splines)
    add_test(
        NAME BenchmarksDryRun.Splines
        COMMAND ddc_benchmark_splines --benchmark_dry_run --benchmark_filter=.*_small
    )
endif()
