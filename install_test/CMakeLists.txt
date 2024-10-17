# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(test-installed-ddc LANGUAGES CXX)

find_package(DDC REQUIRED)

message("DDC options:")
message("DDC_BUILD_DOUBLE_PRECISION=${DDC_BUILD_DOUBLE_PRECISION}")
message("DDC_BUILD_KERNELS_FFT=${DDC_BUILD_KERNELS_FFT}")
message("DDC_BUILD_KERNELS_SPLINES=${DDC_BUILD_KERNELS_SPLINES}")
message("DDC_BUILD_PDI_WRAPPER=${DDC_BUILD_PDI_WRAPPER}")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE DDC::DDC DDC::PDI_Wrapper)
