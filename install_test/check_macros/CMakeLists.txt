# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(test-ddc-macros LANGUAGES CXX)

find_package(DDC 0.12 REQUIRED COMPONENTS fft pdi splines)

add_library(check-macros)
target_sources(check-macros PRIVATE with_my_lib.cpp without_my_lib.cpp)
target_link_libraries(check-macros PRIVATE DDC::core DDC::fft DDC::pdi DDC::splines)
