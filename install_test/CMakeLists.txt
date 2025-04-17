# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(test-installed-ddc LANGUAGES CXX)

# Test whether `find_package(DDC)` can be called multiple times
find_package(DDC 0.6 REQUIRED)
find_package(DDC 0.6 REQUIRED COMPONENTS fft)
find_package(DDC 0.6 REQUIRED COMPONENTS pdi splines)

message(STATUS "DDC options:")
message(STATUS "* DDC_BUILD_DOUBLE_PRECISION=${DDC_BUILD_DOUBLE_PRECISION}")
message(STATUS "* DDC_fft_FOUND=${DDC_fft_FOUND}")
message(STATUS "* DDC_pdi_FOUND=${DDC_pdi_FOUND}")
message(STATUS "* DDC_splines_FOUND=${DDC_splines_FOUND}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE DDC::core DDC::fft DDC::pdi DDC::splines)
