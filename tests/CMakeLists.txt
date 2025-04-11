# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

include(GoogleTest)

add_subdirectory(discrete_space)

add_executable(
    ddc_tests
    main.cpp
    aligned_allocator.cpp
    chunk.cpp
    chunk_span.cpp
    create_mirror.cpp
    discrete_domain.cpp
    discrete_element.cpp
    discrete_space.cpp
    discrete_vector.cpp
    for_each.cpp
    multiple_discrete_dimensions.cpp
    non_uniform_point_sampling.cpp
    parallel_deepcopy.cpp
    parallel_fill.cpp
    parallel_for_each.cpp
    parallel_transform_reduce.cpp
    print.cpp
    relocatable_device_code.cpp
    relocatable_device_code_initialization.cpp
    sparse_discrete_domain.cpp
    strided_discrete_domain.cpp
    tagged_vector.cpp
    transform_reduce.cpp
    trivial_space.cpp
    type_seq.cpp
    uniform_point_sampling.cpp
)
target_compile_features(ddc_tests PUBLIC cxx_std_17)
target_link_libraries(ddc_tests PUBLIC GTest::gtest DDC::core)
gtest_discover_tests(ddc_tests DISCOVERY_MODE PRE_TEST)

if("${DDC_BUILD_KERNELS_FFT}")
    add_subdirectory(fft)
endif()

if("${DDC_BUILD_KERNELS_SPLINES}")
    add_subdirectory(splines)
endif()

if("${DDC_BUILD_PDI_WRAPPER}")
    add_subdirectory(pdi)
endif()
