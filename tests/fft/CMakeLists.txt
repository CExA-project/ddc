# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

include(GoogleTest)

add_executable(fft_tests ../main.cpp fft.cpp)
target_compile_features(fft_tests PUBLIC cxx_std_17)
target_link_libraries(fft_tests PUBLIC GTest::gtest DDC::DDC)
gtest_discover_tests(fft_tests DISCOVERY_MODE PRE_TEST)
