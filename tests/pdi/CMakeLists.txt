# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(pdi_tests pdi.cpp)
set_property(TARGET pdi_tests PROPERTY ENABLE_EXPORTS TRUE)
target_compile_features(pdi_tests PRIVATE cxx_std_17)
target_link_libraries(pdi_tests PRIVATE ddc_gtest_main DDC::core DDC::pdi GTest::gtest)
gtest_discover_tests(pdi_tests DISCOVERY_MODE PRE_TEST)
