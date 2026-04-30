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

# Targets that must NOT exist
foreach(target IN ITEMS DDC::core DDC::fft DDC::pdi DDC::splines)
    assert(NOT TARGET ${target})
endforeach()
# Variables that must NOT exist
foreach(
    var
    IN
    ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND
)
    assert(NOT DEFINED ${var})
endforeach()
# Cache variables that must NOT exist
foreach(
    var
    IN
    ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND
)
    assert(NOT DEFINED CACHE{${var}})
endforeach()

# Test whether `find_package(DDC)` can be called multiple times
# and the additivity of DDC components.

# Find DDC without components, only core will be defined
find_package(DDC 0.13 REQUIRED)

# Targets that must exist
foreach(target IN ITEMS DDC::core)
    assert(TARGET ${target})
endforeach()
# Targets that must NOT exist
foreach(target IN ITEMS DDC::fft DDC::pdi DDC::splines)
    assert(NOT TARGET ${target})
endforeach()
# Variables that must exist
foreach(var IN ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION)
    assert(DEFINED ${var} AND ${var})
endforeach()
# Variables that must NOT exist
foreach(var IN ITEMS DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND)
    assert(NOT DEFINED ${var})
endforeach()
# Cache variables that must NOT exist
foreach(
    var
    IN
    ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND
)
    assert(NOT DEFINED CACHE{${var}})
endforeach()

# Find again DDC with fft component, core and fft will be defined
find_package(DDC 0.13 REQUIRED COMPONENTS fft)

# Targets that must exist
foreach(target IN ITEMS DDC::core DDC::fft)
    assert(TARGET ${target})
endforeach()
# Targets that must NOT exist
foreach(target IN ITEMS DDC::pdi DDC::splines)
    assert(NOT TARGET ${target})
endforeach()
# Variables that must exist
foreach(var IN ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND)
    assert(DEFINED ${var} AND ${var})
endforeach()
# Variables that must NOT exist
foreach(var IN ITEMS DDC_pdi_FOUND DDC_splines_FOUND)
    assert(NOT DEFINED ${var})
endforeach()
# Cache variables that must NOT exist
foreach(
    var
    IN
    ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND
)
    assert(NOT DEFINED CACHE{${var}})
endforeach()

# Find again DDC with pdi and splines components, all components will be defined
find_package(DDC 0.13 REQUIRED COMPONENTS pdi splines)

# Targets that must exist
foreach(target IN ITEMS DDC::core DDC::fft DDC::pdi DDC::splines)
    assert(TARGET ${target})
endforeach()
# Variables that must exist
foreach(
    var
    IN
    ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND
)
    assert(DEFINED ${var} AND ${var})
endforeach()
# Cache variables that must NOT exist
foreach(
    var
    IN
    ITEMS DDC_FOUND DDC_BUILD_DOUBLE_PRECISION DDC_fft_FOUND DDC_pdi_FOUND DDC_splines_FOUND
)
    assert(NOT DEFINED CACHE{${var}})
endforeach()

add_executable(main main.cpp)
target_link_libraries(main PRIVATE DDC::core DDC::fft DDC::pdi DDC::splines)
