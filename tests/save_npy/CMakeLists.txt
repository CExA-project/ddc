# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# ── C++ generator executable ──────────────────────────────────────────────────
add_executable(generate_npy_fixtures generate_npy_fixtures.cpp)
target_link_libraries(generate_npy_fixtures PRIVATE DDC::core Kokkos::kokkos)

# ── Find Python + pytest ──────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy, pytest"
    RESULT_VARIABLE _py_deps_ok
    OUTPUT_QUIET
    ERROR_QUIET
)
if(NOT _py_deps_ok EQUAL 0)
    message(WARNING "numpy or pytest not found — skipping npy validation tests")
else()
    # ── CTest integration ─────────────────────────────────────────────────────
    add_test(NAME npy_generate_fixtures COMMAND generate_npy_fixtures)

    add_test(
        NAME npy_validate_python
        COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_npy_fixtures.py
    )

    # Guarantee ordering: generate before validate
    set_tests_properties(npy_validate_python PROPERTIES DEPENDS npy_generate_fixtures)
endif()
