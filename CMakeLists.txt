# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(DDC VERSION 0.4.1 LANGUAGES CXX)

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
        find_package(Kokkos 4.4...<5 QUIET)
        if(NOT Kokkos_FOUND)
            ddc_configure_kokkos()
        else()
            ddc_check_required_kokkos_options()
        endif()
    endif()
elseif("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    ddc_configure_kokkos()
elseif("${DDC_Kokkos_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(Kokkos 4.4...<5 REQUIRED)
    ddc_check_required_kokkos_options()
endif()

## PDI

if("${DDC_BUILD_PDI_WRAPPER}")
    if(NOT TARGET PDI::PDI_C)
        find_package(PDI 1.6...<2 REQUIRED COMPONENTS C)
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

### DDC legacy target
add_library(DDC INTERFACE)
add_library(DDC::DDC ALIAS DDC)
install(TARGETS DDC EXPORT DDCTargets)

add_library(ddc_core INTERFACE)
add_library(DDC::core ALIAS ddc_core)
configure_file(cmake/config.hpp.in generated/ddc/config.hpp NO_SOURCE_PERMISSIONS @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/ddc/config.hpp DESTINATION include/ddc)
install(TARGETS ddc_core EXPORT DDCTargets)
target_compile_features(ddc_core INTERFACE cxx_std_17)
target_include_directories(
    ddc_core
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${DDC_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(ddc_core INTERFACE Kokkos::kokkos)

### DDC legacy target
target_link_libraries(DDC INTERFACE DDC::core)

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

    add_library(ddc_fft INTERFACE)
    add_library(DDC::fft ALIAS ddc_fft)
    install(TARGETS ddc_fft EXPORT DDCTargets)
    target_link_libraries(ddc_fft INTERFACE DDC::core Kokkos::kokkos KokkosFFT::fft)

    ### DDC legacy target
    target_link_libraries(DDC INTERFACE DDC::fft)
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
            find_package(KokkosKernels 4.5.1...<5 QUIET)
            if(NOT KokkosKernels_FOUND)
                ddc_configure_kokkos_kernels()
            endif()
        endif()
    elseif("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
        ddc_configure_kokkos_kernels()
    elseif("${DDC_KokkosKernels_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
        find_package(KokkosKernels 4.5.1...<5 REQUIRED)
    endif()

    add_library(ddc_splines INTERFACE)
    add_library(DDC::splines ALIAS ddc_splines)
    install(FILES cmake/FindLAPACKE.cmake DESTINATION ${DDC_INSTALL_CMAKEDIR})
    install(TARGETS ddc_splines EXPORT DDCTargets)
    target_include_directories(ddc_splines SYSTEM INTERFACE ${LAPACKE_INCLUDE_DIRS})
    target_link_libraries(
        ddc_splines
        INTERFACE DDC::core Ginkgo::ginkgo Kokkos::kokkos Kokkos::kokkoskernels ${LAPACKE_LIBRARIES}
    )

    ### DDC legacy target
    target_link_libraries(DDC INTERFACE DDC::splines)
endif()

## The PDI wrapper

if("${DDC_BUILD_PDI_WRAPPER}")
    add_library(ddc_pdi INTERFACE)
    add_library(DDC::pdi ALIAS ddc_pdi)
    install(TARGETS ddc_pdi EXPORT DDCTargets)
    target_compile_features(ddc_pdi INTERFACE cxx_std_17)
    target_include_directories(
        ddc_pdi
        INTERFACE
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${DDC_INSTALL_INCLUDEDIR}>"
    )
    target_link_libraries(ddc_pdi INTERFACE DDC::core PDI::PDI_C)

    ### DDC legacy target
    target_link_libraries(DDC INTERFACE DDC::pdi)
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

install(EXPORT DDCTargets NAMESPACE DDC::impl:: DESTINATION ${DDC_INSTALL_CMAKEDIR})

install(DIRECTORY include/ddc TYPE INCLUDE)

configure_package_config_file(
    cmake/DDCConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
    INSTALL_DESTINATION ${DDC_INSTALL_CMAKEDIR}
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
        cmake/DDCCheckRequiredKokkosOptions.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/DDCConfigVersion.cmake
    DESTINATION ${DDC_INSTALL_CMAKEDIR}
)
