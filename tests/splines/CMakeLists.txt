# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

include(GoogleTest)

set(DDC_SPLINES_TEST_DEGREE_MIN 3 CACHE STRING "Minimum degree to test splines.")
set(DDC_SPLINES_TEST_DEGREE_MAX 3 CACHE STRING "Maximum degree to test splines.")

add_executable(
    splines_tests
    bsplines.cpp
    knots_as_interpolation_points.cpp
    splines_linear_problem.cpp
    spline_builder.cpp
    spline_traits.cpp
    view.cpp
)
target_compile_features(splines_tests PRIVATE cxx_std_17)
target_link_libraries(splines_tests PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest)
gtest_discover_tests(splines_tests NO_PRETTY_TYPES DISCOVERY_MODE PRE_TEST)

foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
    foreach(bsplines_type "uniform" "non_uniform")
        string(TOUPPER "degree=${degree}" degree_macro)
        string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
        set(test_filename "periodic_spline_builder")
        set(test_name "${test_filename}_tests_degree_${degree}_${bsplines_type}")

        add_executable("${test_name}" "${test_filename}.cpp")
        target_compile_definitions(
            "${test_name}"
            PRIVATE -D${degree_macro} -D${bsplines_type_macro}
        )
        target_compile_features("${test_name}" PRIVATE cxx_std_17)
        target_link_libraries(
            "${test_name}"
            PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
        )
        gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
    endforeach()
endforeach()

foreach(bcl "greville" "hermite")
    foreach(bcr "greville" "hermite")
        foreach(evaluator "cosine" "polynomial")
            foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
                foreach(bsplines_type "uniform" "non_uniform")
                    string(TOUPPER "bcl_${bcl}" bcl_macro)
                    string(TOUPPER "bcr_${bcr}" bcr_macro)
                    string(TOUPPER "evaluator_${evaluator}" evaluator_macro)
                    string(TOUPPER "degree=${degree}" degree_macro)
                    string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                    set(test_filename "non_periodic_spline_builder")
                    set(test_name
                        "${test_filename}_tests_degree_${degree}_${bsplines_type}_${evaluator}_${bcl}_${bcr}"
                    )

                    add_executable("${test_name}" "${test_filename}.cpp")
                    target_compile_definitions(
                        "${test_name}"
                        PRIVATE
                            -D${bcl_macro}
                            -D${bcr_macro}
                            -D${evaluator_macro}
                            -D${degree_macro}
                            -D${bsplines_type_macro}
                    )
                    target_compile_features("${test_name}" PRIVATE cxx_std_17)
                    target_link_libraries(
                        "${test_name}"
                        PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                    )
                    gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
                endforeach()
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(bc "periodic" "greville")
    foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
        foreach(bsplines_type "uniform" "non_uniform")
            foreach(er "null" "constant")
                string(TOUPPER "bc_${bc}" bc_macro)
                string(TOUPPER "degree=${degree}" degree_macro)
                string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                string(TOUPPER "er_${er}" er_macro)
                set(test_filename "extrapolation_rule")
                set(test_name
                    "${test_filename}_tests_degree_${degree}_${bsplines_type}_${bc}_${er}"
                )

                add_executable("${test_name}" "${test_filename}.cpp")
                target_compile_definitions(
                    "${test_name}"
                    PRIVATE -D${bc_macro} -D${degree_macro} -D${bsplines_type_macro} -D${er_macro}
                )
                target_compile_features("${test_name}" PRIVATE cxx_std_17)
                target_link_libraries(
                    "${test_name}"
                    PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                )
                gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(solver "ginkgo" "lapack")
    foreach(bc "periodic" "greville" "hermite")
        foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
            foreach(bsplines_type "uniform" "non_uniform")
                string(TOUPPER "solver_${solver}" solver_macro)
                string(TOUPPER "bc_${bc}" bc_macro)
                string(TOUPPER "degree=${degree}" degree_macro)
                string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                set(test_filename "batched_spline_builder")
                set(test_name
                    "${test_filename}_tests_degree_${degree}_${bsplines_type}_${bc}_${solver}"
                )

                add_executable("${test_name}" "${test_filename}.cpp")
                target_compile_definitions(
                    "${test_name}"
                    PRIVATE
                        -D${solver_macro}
                        -D${bc_macro}
                        -D${degree_macro}
                        -D${bsplines_type_macro}
                )
                target_compile_features("${test_name}" PRIVATE cxx_std_17)
                target_link_libraries(
                    "${test_name}"
                    PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                )
                gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
    string(TOUPPER "degree=${degree}" degree_macro)
    set(test_filename "periodic_spline_builder_ordered_points")
    set(test_name "${test_filename}_tests_degree_${degree}")

    add_executable("${test_name}" "${test_filename}.cpp")
    target_compile_definitions("${test_name}" PRIVATE -D${degree_macro})
    target_compile_features("${test_name}" PRIVATE cxx_std_17)
    target_link_libraries("${test_name}" PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest)
    gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
endforeach()

foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
    foreach(bsplines_type "uniform" "non_uniform")
        string(TOUPPER "degree=${degree}" degree_macro)
        string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
        set(test_filename "periodicity_spline_builder")
        set(test_name "${test_filename}_tests_degree_${degree}_${bsplines_type}")

        add_executable("${test_name}" "${test_filename}.cpp")
        target_compile_definitions(
            "${test_name}"
            PRIVATE -D${degree_macro} -D${bsplines_type_macro}
        )
        target_compile_features("${test_name}" PRIVATE cxx_std_17)
        target_link_libraries(
            "${test_name}"
            PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
        )
        gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
    endforeach()
endforeach()

foreach(bc "periodic" "greville" "hermite")
    foreach(evaluator "polynomial")
        foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
            foreach(bsplines_type "uniform" "non_uniform")
                string(TOUPPER "bc_${bc}" bc_macro)
                string(TOUPPER "evaluator_${evaluator}" evaluator_macro)
                string(TOUPPER "degree=${degree}" degree_macro)
                string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                set(test_filename "batched_2d_spline_builder")
                set(test_name
                    "${test_filename}_tests_degree_${degree}_${bsplines_type}_${evaluator}_${bc}"
                )

                add_executable("${test_name}" "${test_filename}.cpp")
                target_compile_definitions(
                    "${test_name}"
                    PRIVATE
                        -D${bc_macro}
                        -D${evaluator_macro}
                        -D${degree_macro}
                        -D${bsplines_type_macro}
                )
                target_compile_features("${test_name}" PRIVATE cxx_std_17)
                target_link_libraries(
                    "${test_name}"
                    PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                )
                gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(bc "periodic" "greville" "hermite")
    foreach(evaluator "polynomial")
        foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
            foreach(bsplines_type "uniform" "non_uniform")
                string(TOUPPER "bc_${bc}" bc_macro)
                string(TOUPPER "evaluator_${evaluator}" evaluator_macro)
                string(TOUPPER "degree=${degree}" degree_macro)
                string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                set(test_filename "batched_3d_spline_builder")
                set(test_name
                    "${test_filename}_tests_degree_${degree}_${bsplines_type}_${evaluator}_${bc}"
                )

                add_executable("${test_name}" "${test_filename}.cpp")
                target_compile_definitions(
                    "${test_name}"
                    PRIVATE
                        -D${bc_macro}
                        -D${evaluator_macro}
                        -D${degree_macro}
                        -D${bsplines_type_macro}
                )
                target_compile_features("${test_name}" PRIVATE cxx_std_17)
                target_link_libraries(
                    "${test_name}"
                    PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                )
                gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 125)
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(bc "periodic" "greville" "hermite")
    foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
        foreach(bsplines_type "uniform" "non_uniform")
            string(TOUPPER "bc_${bc}" bc_macro)
            string(TOUPPER "degree=${degree}" degree_macro)
            string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
            set(test_filename "batched_nd_evaluator_1d_spline_builder")
            set(test_name
                "${test_filename}_tests_degree_${degree}_${bsplines_type}_${bc}"
            )

            add_executable("${test_name}" "${test_filename}.cpp")
            target_compile_definitions(
                "${test_name}"
                PRIVATE
                    -D${bc_macro}
                    -D${degree_macro}
                    -D${bsplines_type_macro}
            )
            target_compile_features("${test_name}" PRIVATE cxx_std_17)
            target_link_libraries(
                "${test_name}"
                PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
            )
            gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
        endforeach()
endforeach()
endforeach()

foreach(bc "periodic" "greville" "hermite")
    foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
        foreach(bsplines_type "uniform" "non_uniform")
            string(TOUPPER "bc_${bc}" bc_macro)
            string(TOUPPER "degree=${degree}" degree_macro)
            string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
            set(test_filename "batched_nd_evaluator_3d_spline_builder")
            set(test_name
                "${test_filename}_tests_degree_${degree}_${bsplines_type}_${bc}"
            )

            add_executable("${test_name}" "${test_filename}.cpp")
            target_compile_definitions(
                "${test_name}"
                PRIVATE
                    -D${bc_macro}
                    -D${degree_macro}
                    -D${bsplines_type_macro}
            )
            target_compile_features("${test_name}" PRIVATE cxx_std_17)
            target_link_libraries(
                "${test_name}"
                PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
            )
            gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
        endforeach()
    endforeach()
endforeach()

foreach(solver "ginkgo" "lapack")
    foreach(bc "periodic" "greville" "hermite")
        foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
            foreach(bsplines_type "uniform" "non_uniform")
                string(TOUPPER "solver_${solver}" solver_macro)
                string(TOUPPER "bc_${bc}" bc_macro)
                string(TOUPPER "degree=${degree}" degree_macro)
                string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                set(test_filename "multiple_batch_domain_spline_builder")
                set(test_name
                    "${test_filename}_tests_degree_${degree}_${bsplines_type}_${bc}_${solver}"
                )

                add_executable("${test_name}" "${test_filename}.cpp")
                target_compile_definitions(
                    "${test_name}"
                    PRIVATE
                        -D${solver_macro}
                        -D${bc_macro}
                        -D${degree_macro}
                        -D${bsplines_type_macro}
                )
                target_compile_features("${test_name}" PRIVATE cxx_std_17)
                target_link_libraries(
                    "${test_name}"
                    PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                )
                gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(bc "periodic" "greville" "hermite")
    foreach(evaluator "polynomial")
        foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
            foreach(bsplines_type "uniform" "non_uniform")
                string(TOUPPER "bc_${bc}" bc_macro)
                string(TOUPPER "evaluator_${evaluator}" evaluator_macro)
                string(TOUPPER "degree=${degree}" degree_macro)
                string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
                set(test_filename "multiple_batch_domain_2d_spline_builder")
                set(test_name
                    "${test_filename}_tests_degree_${degree}_${bsplines_type}_${evaluator}_${bc}"
                )

                add_executable("${test_name}" "${test_filename}.cpp")
                target_compile_definitions(
                    "${test_name}"
                    PRIVATE
                        -D${bc_macro}
                        -D${evaluator_macro}
                        -D${degree_macro}
                        -D${bsplines_type_macro}
                )
                target_compile_features("${test_name}" PRIVATE cxx_std_17)
                target_link_libraries(
                    "${test_name}"
                    PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
                )
                gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
            endforeach()
        endforeach()
    endforeach()
endforeach()

foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
    foreach(bsplines_type "uniform" "non_uniform")
        string(TOUPPER "degree=${degree}" degree_macro)
        string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
        set(test_filename "1d_spline_evaluator_derivatives")
        set(test_name "${test_filename}_tests_degree_${degree}_${bsplines_type}")

        add_executable("${test_name}" "${test_filename}.cpp")
        target_compile_definitions(
            "${test_name}"
            PRIVATE -D${degree_macro} -D${bsplines_type_macro}
        )
        target_compile_features("${test_name}" PRIVATE cxx_std_17)
        target_link_libraries(
            "${test_name}"
            PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
        )
        gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
    endforeach()
endforeach()

foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
    foreach(bsplines_type "uniform" "non_uniform")
        string(TOUPPER "degree=${degree}" degree_macro)
        string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
        set(test_filename "2d_spline_evaluator_derivatives")
        set(test_name "${test_filename}_tests_degree_${degree}_${bsplines_type}")

        add_executable("${test_name}" "${test_filename}.cpp")
        target_compile_definitions(
            "${test_name}"
            PRIVATE -D${degree_macro} -D${bsplines_type_macro}
        )
        target_compile_features("${test_name}" PRIVATE cxx_std_17)
        target_link_libraries(
            "${test_name}"
            PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
        )
        gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST)
    endforeach()
endforeach()

foreach(degree RANGE "${DDC_SPLINES_TEST_DEGREE_MIN}" "${DDC_SPLINES_TEST_DEGREE_MAX}")
    foreach(bsplines_type "uniform" "non_uniform")
        string(TOUPPER "degree=${degree}" degree_macro)
        string(TOUPPER "bsplines_type_${bsplines_type}" bsplines_type_macro)
        set(test_filename "3d_spline_evaluator_derivatives")
        set(test_name "${test_filename}_tests_degree_${degree}_${bsplines_type}")

        add_executable("${test_name}" "${test_filename}.cpp")
        target_compile_definitions(
            "${test_name}"
            PRIVATE -D${degree_macro} -D${bsplines_type_macro}
        )
        target_compile_features("${test_name}" PRIVATE cxx_std_17)
        target_link_libraries(
            "${test_name}"
            PRIVATE ddc_gtest_main DDC::core DDC::splines GTest::gtest
        )
        gtest_discover_tests("${test_name}" DISCOVERY_MODE PRE_TEST PROPERTIES TIMEOUT 600)
    endforeach()
endforeach()
