# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

include(GoogleTest)

add_executable(reproducer
    test_a.cpp
    test_b.cpp
    ../main.cpp
)

target_link_libraries(reproducer
    PUBLIC
        GTest::gtest
        DDC::core
)

gtest_discover_tests(reproducer DISCOVERY_MODE PRE_TEST)
