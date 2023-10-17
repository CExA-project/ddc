cmake_minimum_required(VERSION 3.15)

include(GoogleTest)

set(SPLINES_TEST_DEGREE_MIN 3 CACHE STRING "Minimum degree to test splines.")
set(SPLINES_TEST_DEGREE_MAX 3 CACHE STRING "Maximum degree to test splines.")

add_executable(splines_tests
    main.cpp
    chunk.cpp
    view.cpp
    gauss_legendre_integration.cpp
)

target_compile_features(splines_tests PUBLIC cxx_std_17)
target_link_libraries(splines_tests
    PUBLIC
        GTest::gtest
		DDC::DDC
)
gtest_discover_tests(splines_tests)

add_executable(bsplines_tests
	main.cpp
	bsplines.cpp
)
target_compile_features(bsplines_tests PUBLIC cxx_std_17)
target_link_libraries(bsplines_tests
    PUBLIC
        GTest::gtest
		DDC::DDC
)

add_executable(matrix_tests
  main.cpp
    matrix.cpp
)
target_compile_features(matrix_tests PUBLIC cxx_std_17)
target_link_libraries(matrix_tests
    PUBLIC
        GTest::gtest
        DDC::DDC
)
gtest_discover_tests(matrix_tests)

add_executable(bernstein_tests
  main.cpp
	bernstein.cpp
)
target_compile_features(bernstein_tests PUBLIC cxx_std_17)
target_link_libraries(bernstein_tests
    PUBLIC
        GTest::gtest
		DDC::DDC
)
gtest_discover_tests(bernstein_tests)

foreach(DEGREE_X RANGE "${SPLINES_TEST_DEGREE_MIN}" "${SPLINES_TEST_DEGREE_MAX}")
  foreach(BSPLINES_TYPE "BSPLINES_TYPE_UNIFORM" "BSPLINES_TYPE_NON_UNIFORM")
    set(test_name "splines_tests_DEGREE_X_${DEGREE_X}_${BSPLINES_TYPE}")
    add_executable("${test_name}" periodic_spline_builder.cpp)
    target_compile_features("${test_name}" PUBLIC cxx_std_17)
    target_link_libraries("${test_name}"
      PUBLIC
      GTest::gtest
      DDC::DDC
    )
    target_compile_definitions("${test_name}" PUBLIC -DDEGREE_X=${DEGREE_X} -D${BSPLINES_TYPE})
    add_test("${test_name}" "${test_name}")
  endforeach()
endforeach()

foreach(DEGREE_X RANGE "${SPLINES_TEST_DEGREE_MIN}" "${SPLINES_TEST_DEGREE_MAX}")
  foreach(BSPLINES_TYPE "BSPLINES_TYPE_UNIFORM" "BSPLINES_TYPE_NON_UNIFORM")
	set(test_name "splines_tests_BATCHED_DEGREE_X_${DEGREE_X}_${BSPLINES_TYPE}")
    add_executable("${test_name}" batched_spline_builder.cpp)
    target_compile_features("${test_name}" PUBLIC cxx_std_17)
    target_link_libraries("${test_name}"
      PUBLIC
      GTest::gtest
      DDC::DDC
    )
	target_compile_definitions("${test_name}" PUBLIC -DDEGREE_X=${DEGREE_X} -D${BSPLINES_TYPE})
	# add_test("${test_name}" "${test_name}")
	gtest_discover_tests("${test_name}")
  endforeach()
endforeach()

foreach(DEGREE_X RANGE "${SPLINES_TEST_DEGREE_MIN}" "${SPLINES_TEST_DEGREE_MAX}")
  set(test_name "splines_ordered_points_DEGREE_X_${DEGREE_X}")
  add_executable("${test_name}" periodic_spline_builder_ordered_points.cpp)
  target_compile_features("${test_name}" PUBLIC cxx_std_17)
  target_link_libraries("${test_name}"
    PUBLIC
    GTest::gtest
    DDC::DDC
  )
  target_compile_definitions("${test_name}" PUBLIC -DDEGREE_X=${DEGREE_X})
  add_test("${test_name}" "${test_name}")
endforeach()

foreach(BCL "BCL_GREVILLE" "BCL_HERMITE")
  foreach(BCR "BCR_GREVILLE" "BCR_HERMITE")
    foreach(EVALUATOR "EVALUATOR_COSINE" "EVALUATOR_POLYNOMIAL")
      foreach(DEGREE_X RANGE "${SPLINES_TEST_DEGREE_MIN}" "${SPLINES_TEST_DEGREE_MAX}")
        foreach(BSPLINES_TYPE "BSPLINES_TYPE_UNIFORM" "BSPLINES_TYPE_NON_UNIFORM")
          set(test_name "splines_tests_DEGREE_X_${DEGREE_X}_${BSPLINES_TYPE}_${EVALUATOR}_bcl_${BCL}_bcr_${BCR}")
          add_executable("${test_name}" non_periodic_spline_builder.cpp)
          target_compile_features("${test_name}" PUBLIC cxx_std_17)
          target_link_libraries("${test_name}"
            PUBLIC
            GTest::gtest
            DDC::DDC
          )
          target_compile_definitions("${test_name}" PUBLIC -DDEGREE_X=${DEGREE_X} -D${BSPLINES_TYPE} -D${EVALUATOR} -D${BCL} -D${BCR})
          add_test("${test_name}" "${test_name}")
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

foreach(DEGREE_X RANGE "${SPLINES_TEST_DEGREE_MIN}" "${SPLINES_TEST_DEGREE_MAX}")
  foreach(BSPLINES_TYPE "BSPLINES_TYPE_UNIFORM" "BSPLINES_TYPE_NON_UNIFORM")
	set(test_name "splines_tests_PERIODICITY_DEGREE_X_${DEGREE_X}_${BSPLINES_TYPE}")
    add_executable("${test_name}" periodicity_spline_builder.cpp)
    target_compile_features("${test_name}" PUBLIC cxx_std_17)
    target_link_libraries("${test_name}"
      PUBLIC
      GTest::gtest
      DDC::DDC
    )
	target_compile_definitions("${test_name}" PUBLIC -DDEGREE_X=${DEGREE_X} -D${BSPLINES_TYPE})
	# add_test("${test_name}" "${test_name}")
	gtest_discover_tests("${test_name}")
  endforeach()
endforeach()
