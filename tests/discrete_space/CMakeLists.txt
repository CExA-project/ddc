# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

include(GoogleTest)

if(Kokkos_COMPILE_LANGUAGE)
    set_source_files_properties(
        discrete_space.cpp
        main.cpp
        PROPERTIES LANGUAGE ${Kokkos_COMPILE_LANGUAGE}
    )
endif()

add_library(discrete_space_tests_lib STATIC discrete_space.cpp)
target_link_libraries(discrete_space_tests_lib PUBLIC GTest::gtest DDC::core)

add_executable(discrete_space_tests main.cpp)
target_link_libraries(discrete_space_tests PUBLIC discrete_space_tests_lib GTest::gtest DDC::core)
gtest_discover_tests(discrete_space_tests DISCOVERY_MODE PRE_TEST)
