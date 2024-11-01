# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(test-installed-ddc LANGUAGES CXX)

find_package(DDC REQUIRED COMPONENTS fft pdi splines)

message("DDC options:")
message("DDC_BUILD_DOUBLE_PRECISION=${DDC_BUILD_DOUBLE_PRECISION}")
message("DDC_fft_FOUND=${DDC_fft_FOUND}")
message("DDC_splines_FOUND=${DDC_splines_FOUND}")
message("DDC_pdi_FOUND=${DDC_pdi_FOUND}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE DDC::core DDC::fft DDC::pdi DDC::splines)
