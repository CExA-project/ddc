# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

include(GoogleTest)

add_executable(pdi_tests ../main.cpp pdi.cpp)
set_property(TARGET pdi_tests PROPERTY ENABLE_EXPORTS TRUE)
target_compile_features(pdi_tests PUBLIC cxx_std_17)
target_link_libraries(pdi_tests PUBLIC GTest::gtest DDC::core DDC::pdi)
gtest_discover_tests(pdi_tests DISCOVERY_MODE PRE_TEST)
