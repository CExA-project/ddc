# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

include(GoogleTest)

add_executable(reproducer2
    sum_version_a.cpp
    sum_version_b.cpp
    ../main.cpp
)

target_link_libraries(reproducer2
    PUBLIC
        GTest::gtest
        DDC::core
)

gtest_discover_tests(reproducer2 DISCOVERY_MODE PRE_TEST)
