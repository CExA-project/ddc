# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(DDC VERSION 0.8.0 LANGUAGES CXX)

# List of options

option(DDC_BUILD_BENCHMARKS "Build DDC benchmarks." OFF)
option(DDC_BUILD_DEPRECATED_CODE "Build DDC deprecated code." ON)
option(DDC_BUILD_DOCUMENTATION "Build DDC documentation/website" OFF)
option(
    DDC_BUILD_DOUBLE_PRECISION
    "Build DDC with double precision support, float is used otherwise"
    ON
)
option(DDC_BUILD_EXAMPLES "Build DDC examples" ON)
option(DDC_BUILD_KERNELS_FFT "Build DDC kernels for FFT" ON)
option(DDC_BUILD_KERNELS_SPLINES "Build DDC kernels for splines" ON)
option(DDC_BUILD_PDI_WRAPPER "Build DDC PDI wrapper" ON)
option(DDC_BUILD_TESTS "Build DDC tests if BUILD_TESTING is enabled" ON)

# Dependencies

set(DDC_DEPENDENCY_POLICIES "AUTO" "EMBEDDED" "INSTALLED" "SUBPROJECT")

## CMake modules

include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

set(DDC_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
set(DDC_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})

## Custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(DDCCheckRequiredKokkosOptions)
include(DDCVendorConfiguration)

## kokkos

set(DDC_Kokkos_DEPENDENCY_POLICY
    "AUTO"
    CACHE STRING
    "Policy to find the `Kokkos` package. Options: ${DDC_DEPENDENCY_POLICIES}"
)
set_property(CACHE DDC_Kokkos_DEPENDENCY_POLICY PROPERTY STRINGS "${DDC_DEPENDENCY_POLICIES}")
if("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "AUTO")
    if(NOT TARGET Kokkos::kokkos)
        find_package(Kokkos 4.4...<6 QUIET)
        if(NOT Kokkos_FOUND)
            ddc_configure_kokkos()
        else()
            ddc_check_required_kokkos_options()
        endif()
    endif()
elseif("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    ddc_configure_kokkos()
elseif("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(Kokkos 4.4...<6 REQUIRED)
    ddc_check_required_kokkos_options()
endif()

## GoogleTest

if("${BUILD_TESTING}" AND "${DDC_BUILD_TESTS}")
    set(DDC_GTest_DEPENDENCY_POLICY
        "AUTO"
        CACHE STRING
        "Policy to find the `GTest` package. Options: ${DDC_DEPENDENCY_POLICIES}"
    )
    set_property(CACHE DDC_GTest_DEPENDENCY_POLICY PROPERTY STRINGS ${DDC_DEPENDENCY_POLICIES})

    if("${DDC_GTest_DEPENDENCY_POLICY}" STREQUAL "AUTO")
        if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
            # 1.14...<2 but GTest module does not support range version
            find_package(GTest 1.14 QUIET)
            if(NOT GTest_FOUND)
                ddc_configure_googletest()
            endif()
        endif()
    elseif("${DDC_GTest_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_googletest()
    elseif("${DDC_GTest_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        # 1.14...<2 but GTest module does not support range version
        find_package(GTest 1.14 REQUIRED)
    endif()
endif()

## Google Benchmark

if("${DDC_BUILD_BENCHMARKS}")
    set(DDC_benchmark_DEPENDENCY_POLICY
        "AUTO"
        CACHE STRING
        "Policy to find the `benchmark` package. Options: ${DDC_DEPENDENCY_POLICIES}"
    )
    set_property(CACHE DDC_benchmark_DEPENDENCY_POLICY PROPERTY STRINGS ${DDC_DEPENDENCY_POLICIES})

    if("${DDC_benchmark_DEPENDENCY_POLICY}" STREQUAL "AUTO")
        if(NOT TARGET benchmark::benchmark)
            find_package(benchmark 1.8...<2 QUIET)
            if(NOT benchmark_FOUND)
                ddc_configure_benchmark()
            endif()
        endif()
    elseif("${DDC_benchmark_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_benchmark()
    elseif("${DDC_benchmark_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(benchmark 1.8...<2 REQUIRED)
    endif()
endif()

## Doxygen

if("${DDC_BUILD_DOCUMENTATION}")
    find_package(Doxygen 1.8.13...<2 REQUIRED OPTIONAL_COMPONENTS dot)
endif()

# Our project

## The library itself

add_library(ddc_core INTERFACE)
add_library(DDC::core ALIAS ddc_core)
set_target_properties(ddc_core PROPERTIES EXPORT_NAME core)
configure_file(cmake/config.hpp.in generated/ddc/config.hpp NO_SOURCE_PERMISSIONS @ONLY)
target_sources(
    ddc_core
    INTERFACE
        FILE_SET HEADERS
            BASE_DIRS include ${CMAKE_CURRENT_BINARY_DIR}/generated
            FILES
                ${CMAKE_CURRENT_BINARY_DIR}/generated/ddc/config.hpp
                include/ddc/detail/dual_discretization.hpp
                include/ddc/detail/kokkos.hpp
                include/ddc/detail/macros.hpp
                include/ddc/detail/tagged_vector.hpp
                include/ddc/detail/type_seq.hpp
                include/ddc/detail/type_traits.hpp
                include/ddc/aligned_allocator.hpp
                include/ddc/chunk.hpp
                include/ddc/chunk_common.hpp
                include/ddc/chunk_span.hpp
                include/ddc/chunk_traits.hpp
                include/ddc/coordinate.hpp
                include/ddc/create_mirror.hpp
                include/ddc/ddc.hpp
                include/ddc/ddc_to_kokkos_execution_policy.hpp
                include/ddc/discrete_domain.hpp
                include/ddc/discrete_element.hpp
                include/ddc/discrete_space.hpp
                include/ddc/discrete_vector.hpp
                include/ddc/for_each.hpp
                include/ddc/kokkos_allocator.hpp
                include/ddc/non_uniform_point_sampling.hpp
                include/ddc/parallel_deepcopy.hpp
                include/ddc/parallel_fill.hpp
                include/ddc/parallel_for_each.hpp
                include/ddc/parallel_transform_reduce.hpp
                include/ddc/periodic_sampling.hpp
                include/ddc/print.hpp
                include/ddc/real_type.hpp
                include/ddc/reducer.hpp
                include/ddc/scope_guard.hpp
                include/ddc/sparse_discrete_domain.hpp
                include/ddc/strided_discrete_domain.hpp
                include/ddc/transform_reduce.hpp
                include/ddc/trivial_space.hpp
                include/ddc/uniform_point_sampling.hpp
)
target_compile_features(ddc_core INTERFACE cxx_std_17)
target_link_libraries(ddc_core INTERFACE Kokkos::kokkos)

install(TARGETS ddc_core EXPORT DDCCoreTargets FILE_SET HEADERS)

# Link library to DDC

if("${DDC_BUILD_KERNELS_FFT}")
    # Kokkos-fft
    set(DDC_KokkosFFT_DEPENDENCY_POLICY
        "AUTO"
        CACHE STRING
        "Policy to find the `Kokkos-fft` package. Options: ${DDC_DEPENDENCY_POLICIES}"
    )
    set_property(
        CACHE DDC_KokkosFFT_DEPENDENCY_POLICY
        PROPERTY STRINGS "${DDC_DEPENDENCY_POLICIES}"
    )

    if("${DDC_KokkosFFT_DEPENDENCY_POLICY}" STREQUAL "AUTO")
        if(NOT TARGET KokkosFFT::fft)
            find_package(KokkosFFT 0.3.0...<1 QUIET)
            if(NOT KokkosFFT_FOUND)
                ddc_configure_kokkos_fft()
            endif()
        endif()
    elseif("${DDC_KokkosFFT_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_kokkos_fft()
    elseif("${DDC_KokkosFFT_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(KokkosFFT 0.3.0...<1 REQUIRED)
    endif()

    add_library(ddc_fft INTERFACE)
    add_library(DDC::fft ALIAS ddc_fft)
    set_target_properties(ddc_fft PROPERTIES EXPORT_NAME fft)
    target_link_libraries(ddc_fft INTERFACE DDC::core Kokkos::kokkos KokkosFFT::fft)
    target_sources(
        ddc_fft
        INTERFACE
            FILE_SET HEADERS
                BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
                FILES include/ddc/kernels/fft.hpp
    )

    install(TARGETS ddc_fft EXPORT DDCFftTargets FILE_SET HEADERS)
    install(EXPORT DDCFftTargets NAMESPACE DDC:: DESTINATION ${DDC_INSTALL_CMAKEDIR})
endif()

if("${DDC_BUILD_KERNELS_SPLINES}")
    # Ginkgo
    find_package(Ginkgo 1.8...<2 REQUIRED)

    # Lapacke
    find_package(LAPACKE REQUIRED)

    # Kokkos-kernels
    set(DDC_KokkosKernels_DEPENDENCY_POLICY
        "AUTO"
        CACHE STRING
        "Policy to find the `KokkosKernels` package. Options: ${DDC_DEPENDENCY_POLICIES}"
    )
    set_property(
        CACHE DDC_KokkosKernels_DEPENDENCY_POLICY
        PROPERTY STRINGS "${DDC_DEPENDENCY_POLICIES}"
    )
    if("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "AUTO")
        if(NOT TARGET Kokkos::kokkoskernels)
            find_package(KokkosKernels 4.5.1...<6 QUIET)
            if(NOT KokkosKernels_FOUND)
                ddc_configure_kokkos_kernels()
            endif()
        endif()
    elseif("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_kokkos_kernels()
    elseif("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(KokkosKernels 4.5.1...<6 REQUIRED)
    endif()

    add_library(ddc_splines INTERFACE)
    add_library(DDC::splines ALIAS ddc_splines)
    set_target_properties(ddc_splines PROPERTIES EXPORT_NAME splines)
    target_include_directories(ddc_splines SYSTEM INTERFACE ${LAPACKE_INCLUDE_DIRS})
    target_link_libraries(
        ddc_splines
        INTERFACE DDC::core Ginkgo::ginkgo Kokkos::kokkos Kokkos::kokkoskernels ${LAPACKE_LIBRARIES}
    )
    target_sources(
        ddc_splines
        INTERFACE
            FILE_SET HEADERS
                BASE_DIRS include
                FILES
                    include/ddc/kernels/splines.hpp
                    include/ddc/kernels/splines/bsplines_non_uniform.hpp
                    include/ddc/kernels/splines/bsplines_uniform.hpp
                    include/ddc/kernels/splines/constant_extrapolation_rule.hpp
                    include/ddc/kernels/splines/deriv.hpp
                    include/ddc/kernels/splines/greville_interpolation_points.hpp
                    include/ddc/kernels/splines/integrals.hpp
                    include/ddc/kernels/splines/knot_discrete_dimension_type.hpp
                    include/ddc/kernels/splines/knots_as_interpolation_points.hpp
                    include/ddc/kernels/splines/kokkos-kernels-ext/KokkosBatched_Gbtrs.hpp
                    include/ddc/kernels/splines/kokkos-kernels-ext/KokkosBatched_Gbtrs_Serial_Impl.hpp
                    include/ddc/kernels/splines/kokkos-kernels-ext/KokkosBatched_Getrs.hpp
                    include/ddc/kernels/splines/kokkos-kernels-ext/KokkosBatched_Getrs_Serial_Impl.hpp
                    include/ddc/kernels/splines/math_tools.hpp
                    include/ddc/kernels/splines/null_extrapolation_rule.hpp
                    include/ddc/kernels/splines/periodic_extrapolation_rule.hpp
                    include/ddc/kernels/splines/spline_boundary_conditions.hpp
                    include/ddc/kernels/splines/spline_builder.hpp
                    include/ddc/kernels/splines/spline_builder_2d.hpp
                    include/ddc/kernels/splines/spline_builder_3d.hpp
                    include/ddc/kernels/splines/spline_evaluator.hpp
                    include/ddc/kernels/splines/spline_evaluator_2d.hpp
                    include/ddc/kernels/splines/spline_evaluator_3d.hpp
                    include/ddc/kernels/splines/spline_traits.hpp
                    include/ddc/kernels/splines/splines_linear_problem.hpp
                    include/ddc/kernels/splines/splines_linear_problem_2x2_blocks.hpp
                    include/ddc/kernels/splines/splines_linear_problem_3x3_blocks.hpp
                    include/ddc/kernels/splines/splines_linear_problem_band.hpp
                    include/ddc/kernels/splines/splines_linear_problem_dense.hpp
                    include/ddc/kernels/splines/splines_linear_problem_maker.hpp
                    include/ddc/kernels/splines/splines_linear_problem_pds_band.hpp
                    include/ddc/kernels/splines/splines_linear_problem_pds_tridiag.hpp
                    include/ddc/kernels/splines/splines_linear_problem_sparse.hpp
                    include/ddc/kernels/splines/view.hpp
    )

    install(FILES cmake/FindLAPACKE.cmake DESTINATION ${DDC_INSTALL_CMAKEDIR})
    install(TARGETS ddc_splines EXPORT DDCSplinesTargets FILE_SET HEADERS)
    install(EXPORT DDCSplinesTargets NAMESPACE DDC:: DESTINATION ${DDC_INSTALL_CMAKEDIR})
endif()

## The PDI wrapper

if("${DDC_BUILD_PDI_WRAPPER}")
    if(NOT TARGET PDI::PDI_C)
        find_package(PDI 1.6...<2 REQUIRED COMPONENTS C)
    endif()

    add_library(ddc_pdi INTERFACE)
    add_library(DDC::pdi ALIAS ddc_pdi)
    set_target_properties(ddc_pdi PROPERTIES EXPORT_NAME pdi)
    target_compile_features(ddc_pdi INTERFACE cxx_std_17)
    target_link_libraries(ddc_pdi INTERFACE DDC::core PDI::PDI_C)
    target_sources(ddc_pdi INTERFACE FILE_SET HEADERS BASE_DIRS include FILES include/ddc/pdi.hpp)

    install(TARGETS ddc_pdi EXPORT DDCPdiTargets FILE_SET HEADERS)
    install(EXPORT DDCPdiTargets NAMESPACE DDC:: DESTINATION ${DDC_INSTALL_CMAKEDIR})
endif()

## if examples are enabled, build them

if("${DDC_BUILD_EXAMPLES}")
    add_subdirectory(examples/)
endif()

## if tests are enabled, build them

if("${BUILD_TESTING}" AND "${DDC_BUILD_TESTS}")
    add_subdirectory(tests/)
endif()

## if benchmarks are enabled, build them

if("${DDC_BUILD_BENCHMARKS}")
    add_subdirectory(benchmarks/)
endif()

## if documentation is enabled, build it

if("${DDC_BUILD_DOCUMENTATION}")
    add_subdirectory(docs/)
endif()

## installation

install(EXPORT DDCCoreTargets NAMESPACE DDC:: DESTINATION ${DDC_INSTALL_CMAKEDIR})

configure_package_config_file(
    cmake/DDCConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
    INSTALL_DESTINATION ${DDC_INSTALL_CMAKEDIR}
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# We use SameMinorVersion until major version 1
if(PROJECT_VERSION VERSION_GREATER_EQUAL 1)
    message(FATAL_ERROR "DDC must switch COMPATIBILITY mode to SameMajorVersion")
endif()
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DDCConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/DDCConfigVersion.cmake
    DESTINATION ${DDC_INSTALL_CMAKEDIR}
)
