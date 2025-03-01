# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(DDC VERSION 0.5.0 LANGUAGES CXX)

# List of options

option(DDC_BUILD_DEPRECATED_CODE "Build DDC deprecated code." ON)
option(DDC_BUILD_EXAMPLES "Build DDC examples" ON)
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

# Our project

## The library itself

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

## if examples are enabled, build them

if("${DDC_BUILD_EXAMPLES}")
    add_subdirectory(examples/)
endif()

## if tests are enabled, build them

if("${BUILD_TESTING}" AND "${DDC_BUILD_TESTS}")
    add_subdirectory(tests/)
endif()

## installation

install(EXPORT DDCTargets NAMESPACE DDC::impl:: DESTINATION ${DDC_INSTALL_CMAKEDIR})

install(DIRECTORY include/ddc TYPE INCLUDE)

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
