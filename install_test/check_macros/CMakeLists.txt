# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)
project(test-ddc-macros LANGUAGES CXX)

find_package(DDC 0.8 REQUIRED COMPONENTS fft pdi splines)

add_library(check-macros with_my_lib.cpp without_my_lib.cpp)
target_link_libraries(check-macros PRIVATE DDC::core DDC::fft DDC::pdi DDC::splines)
