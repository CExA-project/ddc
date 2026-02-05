# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(fft_tests fft.cpp)
target_compile_features(fft_tests PRIVATE cxx_std_17)
target_link_libraries(fft_tests PRIVATE ddc_gtest_main DDC::core DDC::fft GTest::gtest)
gtest_discover_tests(fft_tests DISCOVERY_MODE PRE_TEST)
