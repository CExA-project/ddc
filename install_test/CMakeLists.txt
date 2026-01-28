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

assert(NOT DEFINED DDC_FOUND)
assert(NOT TARGET DDC::core)
assert(NOT TARGET DDC::fft)

# Test whether `find_package(DDC)` can be called multiple times
# and the additivity of DDC components.

# Find DDC without components, only core will be defined
find_package(DDC 0.12 REQUIRED COMPONENTS core)

assert(DEFINED DDC_FOUND)
assert(TARGET DDC::core)
assert(NOT TARGET DDC::fft)

find_package(DDC 0.12 REQUIRED COMPONENTS core fft)

assert(DEFINED DDC_FOUND)
assert(TARGET DDC::core)
assert(TARGET DDC::fft)

add_executable(main)
target_sources(main PRIVATE main.cpp)
target_link_libraries(main PRIVATE DDC::core DDC::fft)
