# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(DDC VERSION 0.1.0 LANGUAGES CXX)

# List of options

option(DDC_BUILD_BENCHMARKS "Build DDC benchmarks." OFF)
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

## Custom cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
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
        find_package(Kokkos 4.4...4.5 QUIET)
        if(NOT Kokkos_FOUND)
            ddc_configure_kokkos()
        endif()
    endif()
elseif("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    ddc_configure_kokkos()
elseif("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(Kokkos 4.4...4.5 REQUIRED)
endif()

## PDI

if("${DDC_BUILD_PDI_WRAPPER}")
    if(NOT TARGET PDI::PDI_C)
        find_package(PDI REQUIRED COMPONENTS C)
    endif()
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
            find_package(GTest QUIET)
            if(NOT GTest_FOUND)
                ddc_configure_googletest()
            endif()
        endif()
    elseif("${DDC_GTest_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_googletest()
    elseif("${DDC_GTest_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(GTest REQUIRED)
    endif()

    include(GoogleTest)
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
            find_package(benchmark QUIET)
            if(NOT benchmark_FOUND)
                ddc_configure_benchmark()
            endif()
        endif()
    elseif("${DDC_benchmark_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_benchmark()
    elseif("${DDC_benchmark_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(benchmark REQUIRED)
    endif()
endif()

## Doxygen

if("${DDC_BUILD_DOCUMENTATION}")
    find_package(Doxygen 1.8.13 REQUIRED OPTIONAL_COMPONENTS dot)
endif()

# Our project

## The library itself

add_library(DDC INTERFACE)
target_compile_features(DDC INTERFACE cxx_std_17)
target_include_directories(
    DDC
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)
target_link_libraries(DDC INTERFACE Kokkos::kokkos)
if("${DDC_BUILD_DOUBLE_PRECISION}")
    target_compile_definitions(DDC INTERFACE DDC_BUILD_DOUBLE_PRECISION)
endif()
add_library(DDC::DDC ALIAS DDC)
install(TARGETS DDC EXPORT DDCTargets)

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
            find_package(KokkosFFT 0.2.1...<1 QUIET)
            if(NOT KokkosFFT_FOUND)
                ddc_configure_kokkos_fft()
            endif()
        endif()
    elseif("${DDC_KokkosFFT_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_kokkos_fft()
    elseif("${DDC_KokkosFFT_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(KokkosFFT 0.2.1...<1 REQUIRED)
    endif()

    target_link_libraries(DDC INTERFACE KokkosFFT::fft)
    target_compile_definitions(DDC INTERFACE "DDC_BUILD_KERNELS_FFT")
endif()

if("${DDC_BUILD_KERNELS_SPLINES}")
    # Ginkgo
    find_package(Ginkgo 1.8.0 EXACT REQUIRED)
    target_link_libraries(DDC INTERFACE Ginkgo::ginkgo)

    # Lapacke
    find_package(LAPACKE REQUIRED)
    target_link_libraries(DDC INTERFACE ${LAPACKE_LIBRARIES})
    target_include_directories(DDC SYSTEM INTERFACE ${LAPACKE_INCLUDE_DIRS})
    install(FILES cmake/FindLAPACKE.cmake DESTINATION lib/cmake/DDC)

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
            find_package(KokkosKernels QUIET)
            if(NOT KokkosKernels_FOUND)
                ddc_configure_kokkos_kernels()
            endif()
        endif()
    elseif("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_kokkos_kernels()
    elseif("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(KokkosKernels REQUIRED)
    endif()

    target_link_libraries(DDC INTERFACE Kokkos::kokkoskernels)
    target_compile_definitions(DDC INTERFACE "DDC_BUILD_KERNELS_SPLINES")
endif()

## The PDI wrapper

if("${DDC_BUILD_PDI_WRAPPER}")
    add_library(PDI_Wrapper INTERFACE)
    target_compile_features(PDI_Wrapper INTERFACE cxx_std_17)
    target_include_directories(
        PDI_Wrapper
        INTERFACE
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:include>"
    )
    target_link_libraries(PDI_Wrapper INTERFACE DDC::DDC PDI::PDI_C)
    target_compile_definitions(PDI_Wrapper INTERFACE "DDC_BUILD_PDI_WRAPPER")
    add_library(DDC::PDI_Wrapper ALIAS PDI_Wrapper)
    install(TARGETS PDI_Wrapper EXPORT DDCTargets)
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

install(EXPORT DDCTargets NAMESPACE DDC:: DESTINATION lib/cmake/DDC)

install(DIRECTORY include/ TYPE INCLUDE)

configure_package_config_file(
    cmake/DDCConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
    INSTALL_DESTINATION lib/cmake/DDC
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake DESTINATION lib/cmake/DDC)
