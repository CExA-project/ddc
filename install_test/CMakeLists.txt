# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25...4.2)

project(test-installed-ddc LANGUAGES CXX)

# assert(<condition>)
#
# Fails configuration with a FATAL_ERROR if <condition> evaluates to FALSE.
# The full condition is printed in the error message.
#
# Example:
#   set(MY_VAR OFF)
#   assert(MY_VAR)  # Triggers: "Assertion failed: MY_VAR"
#
#   set(VERSION 3)
#   assert(VERSION GREATER 2)  # Passes
function(assert)
    if(NOT (${ARGN}))
        string(REPLACE ";" " " ASSERT_TEXT "${ARGN}")
        message(FATAL_ERROR "Assertion failed: ${ASSERT_TEXT}")
    endif()
endfunction()

# Test whether `find_package(DDC)` does not define global targets and variables
add_subdirectory(subdir)

## Test targets
assert(NOT TARGET DDC::core)
assert(NOT TARGET DDC::fft)
assert(NOT TARGET DDC::pdi)
assert(NOT TARGET DDC::splines)
## Test normal variables
assert(NOT DEFINED DDC_FOUND)
assert(NOT DEFINED DDC_BUILD_DOUBLE_PRECISION)
assert(NOT DEFINED DDC_fft_FOUND)
assert(NOT DEFINED DDC_pdi_FOUND)
assert(NOT DEFINED DDC_splines_FOUND)
## Test cache variables
assert(NOT DEFINED CACHE{DDC_FOUND})
assert(NOT DEFINED CACHE{DDC_BUILD_DOUBLE_PRECISION})
assert(NOT DEFINED CACHE{DDC_fft_FOUND})
assert(NOT DEFINED CACHE{DDC_pdi_FOUND})
assert(NOT DEFINED CACHE{DDC_splines_FOUND})

# Test whether `find_package(DDC)` can be called multiple times
find_package(DDC 0.10 REQUIRED)

## Test targets
assert(TARGET DDC::core)
assert(NOT TARGET DDC::fft)
assert(NOT TARGET DDC::pdi)
assert(NOT TARGET DDC::splines)
## Test normal variables
assert(DEFINED DDC_FOUND AND DDC_FOUND)
assert(DEFINED DDC_BUILD_DOUBLE_PRECISION AND DDC_BUILD_DOUBLE_PRECISION)
assert(NOT DEFINED DDC_fft_FOUND)
assert(NOT DEFINED DDC_pdi_FOUND)
assert(NOT DEFINED DDC_splines_FOUND)
## Test cache variables
assert(NOT DEFINED CACHE{DDC_FOUND})
assert(NOT DEFINED CACHE{DDC_BUILD_DOUBLE_PRECISION})
assert(NOT DEFINED CACHE{DDC_fft_FOUND})
assert(NOT DEFINED CACHE{DDC_pdi_FOUND})
assert(NOT DEFINED CACHE{DDC_splines_FOUND})

find_package(DDC 0.10 REQUIRED COMPONENTS fft)

## Test targets
assert(TARGET DDC::core)
assert(TARGET DDC::fft)
assert(NOT TARGET DDC::pdi)
assert(NOT TARGET DDC::splines)
## Test normal variables
assert(DEFINED DDC_FOUND AND DDC_FOUND)
assert(DEFINED DDC_BUILD_DOUBLE_PRECISION AND DDC_BUILD_DOUBLE_PRECISION)
assert(DEFINED DDC_fft_FOUND AND DDC_fft_FOUND)
assert(NOT DEFINED DDC_pdi_FOUND)
assert(NOT DEFINED DDC_splines_FOUND)
## Test cache variables
assert(NOT DEFINED CACHE{DDC_FOUND})
assert(NOT DEFINED CACHE{DDC_BUILD_DOUBLE_PRECISION})
assert(NOT DEFINED CACHE{DDC_fft_FOUND})
assert(NOT DEFINED CACHE{DDC_pdi_FOUND})
assert(NOT DEFINED CACHE{DDC_splines_FOUND})

find_package(DDC 0.10 REQUIRED COMPONENTS pdi splines)

## Test targets
assert(TARGET DDC::core)
assert(TARGET DDC::fft)
assert(TARGET DDC::pdi)
assert(TARGET DDC::splines)
## Test normal variables
assert(DEFINED DDC_FOUND AND DDC_FOUND)
assert(DEFINED DDC_BUILD_DOUBLE_PRECISION AND DDC_BUILD_DOUBLE_PRECISION)
assert(DEFINED DDC_fft_FOUND AND DDC_fft_FOUND)
assert(DEFINED DDC_pdi_FOUND AND DDC_pdi_FOUND)
assert(DEFINED DDC_splines_FOUND AND DDC_splines_FOUND)
## Test cache variables
assert(NOT DEFINED CACHE{DDC_FOUND})
assert(NOT DEFINED CACHE{DDC_BUILD_DOUBLE_PRECISION})
assert(NOT DEFINED CACHE{DDC_fft_FOUND})
assert(NOT DEFINED CACHE{DDC_pdi_FOUND})
assert(NOT DEFINED CACHE{DDC_splines_FOUND})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE DDC::core DDC::fft DDC::pdi DDC::splines)
