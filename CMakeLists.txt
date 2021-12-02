# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.15)
project(DDC VERSION 0.0.0 LANGUAGES CXX)

# List of options
option(BUILD_BENCHMARKS    "Build the benchmarks." OFF)
option(BUILD_DOCUMENTATION "Build the documentation" ON)


# Dependencies

find_package(PDI REQUIRED COMPONENTS C)
find_package(mdspan CONFIG REQUIRED)
find_package(GTest)
find_package(benchmark)

## Use CTest for running tests
include(CTest)

# Our project

## The library itself
add_library(DDC INTERFACE)
target_compile_features(DDC INTERFACE cxx_std_17)
target_include_directories(DDC
	INTERFACE
		"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_include_directories(DDC
	SYSTEM INTERFACE
		"$<INSTALL_INTERFACE:include>"
)
target_link_libraries(DDC
	INTERFACE std::mdspan)
add_library(DDC::DDC ALIAS DDC)

## The PDI wrapper
add_library(PDI INTERFACE)
target_compile_features(PDI INTERFACE cxx_std_17)
target_include_directories(PDI
	INTERFACE
		"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_include_directories(PDI
	SYSTEM INTERFACE
		"$<INSTALL_INTERFACE:include>"
)
target_link_libraries(PDI
	INTERFACE
		DDC::DDC
		PDI::PDI_C)
add_library(DDC::PDI ALIAS PDI)

## if tests are enabled, build the tests in `tests/`
if("${BUILD_TESTING}")
	add_subdirectory(tests/)
endif()

## if benchmarks are enabled, build the benchmarks in `benchmarks/`
if("${BUILD_BENCHMARKS}")
	add_subdirectory(benchmarks/)
endif()

## if documentation is enabled, build the documentation docs

if("${BUILD_DOCUMENTATION}")
	add_subdirectory(docs/)
endif()

install(
	TARGETS DDC PDI
	EXPORT DDCTargets)

install(
	EXPORT DDCTargets
	FILE DDCTargets.cmake
	NAMESPACE DDC::
	DESTINATION lib/cmake/DDC)

export(
	TARGETS DDC PDI
	FILE DDCTargets.cmake)

install(
	DIRECTORY include/
	TYPE INCLUDE)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DDCConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
	INSTALL_DESTINATION lib/cmake/DDC)

install(
	FILES ${CMAKE_CURRENT_BINARY_DIR}/DDCConfig.cmake
	DESTINATION lib/cmake/DDC)
