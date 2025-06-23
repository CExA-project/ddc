// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>
#include "spline_builder.hpp"
#include "spline_builder_2d.hpp"
#include "spline_evaluator.hpp"
#include "spline_evaluator_2d.hpp"

namespace ddc {
template <class T>
struct is_spline_builder : std::false_type {};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationDDim,
        ddc::BoundCond BcLower,
        ddc::BoundCond BcUpper,
        SplineSolver Solver>
struct is_spline_builder<SplineBuilder<ExecSpace, MemorySpace, BSplines, InterpolationDDim, BcLower, BcUpper, Solver>> : std::true_type {};

/**
 *  @brief A helper to check if T is a SplineBuilder
 *  @tparam T The type to be checked if is a SplineBuilder
 */
template <class T>
inline constexpr bool is_spline_builder_v = is_spline_builder<T>::value;

template <class T>
struct is_spline_builder2D : std::false_type {};

template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class DDimI1,
        class DDimI2,
        ddc::BoundCond BcLower1,
        ddc::BoundCond BcUpper1,
        ddc::BoundCond BcLower2,
        ddc::BoundCond BcUpper2,
        ddc::SplineSolver Solver>
struct is_spline_builder2D<SplineBuilder2D<ExecSpace, MemorySpace, BSpline1, BSpline2, DDimI1, DDimI2, BcLower1, BcUpper1, BcLower2, BcUpper2, Solver>> : std::true_type {};

/**
 *  @brief A helper to check if T is a SplineBuilder2D
 *  @tparam T The type to be checked if is a SplineBuilder2D
 */
template <class T>
inline constexpr bool is_spline_builder2D_v = is_spline_builder2D<T>::value;

template <class T>
struct is_spline_evaluator : std::false_type {};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class EvaluationDDim,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_spline_evaluator<SplineEvaluator<ExecSpace, MemorySpace, BSplines, EvaluationDDim, LowerExtrapolationRule, UpperExtrapolationRule>> : std::true_type {};

/**
 *  @brief A helper to check if T is a SplineEvaluator
 *  @tparam T The type to be checked if is a SplineEvaluator
 */
template <class T>
inline constexpr bool is_spline_evaluator_v = is_spline_evaluator<T>::value;

template <class T>
struct is_spline_evaluator2D : std::false_type {};

template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class EvaluationDDim1,
        class EvaluationDDim2,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2>
struct is_spline_evaluator2D<SplineEvaluator2D<ExecSpace, MemorySpace, BSpline1, BSpline2, EvaluationDDim1, EvaluationDDim2, LowerExtrapolationRule1, UpperExtrapolationRule1, LowerExtrapolationRule2, UpperExtrapolationRule2>> : std::true_type {};

/**
 *  @brief A helper to check if T is a SplineEvaluator2D
 *  @tparam T The type to be checked if is a SplineEvaluator2D
 */
template <class T>
inline constexpr bool is_spline_evaluator2D_v = is_spline_evaluator2D<T>::value;

template <class Builder, class Evaluator>
struct is_evaluator_admissible : std::false_type {};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationDDim,
        ddc::BoundCond BcLower,
        ddc::BoundCond BcUpper,
        SplineSolver Solver,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_evaluator_admissible<SplineBuilder<ExecSpace, MemorySpace, BSplines, InterpolationDDim, BcLower, BcUpper, Solver>, SplineEvaluator<ExecSpace, MemorySpace, BSplines, InterpolationDDim, LowerExtrapolationRule, UpperExtrapolationRule>> : std::true_type {};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        class DDimI1,
        class DDimI2,
        ddc::BoundCond BcLower1,
        ddc::BoundCond BcUpper1,
        ddc::BoundCond BcLower2,
        ddc::BoundCond BcUpper2,
        SplineSolver Solver,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2>
struct is_evaluator_admissible<SplineBuilder2D<ExecSpace, MemorySpace, BSplines1, BSplines2, DDimI1, DDimI2, BcLower1, BcUpper1, BcLower2, BcUpper2, Solver>,
                               SplineEvaluator2D<ExecSpace, MemorySpace, BSplines1, BSplines2, DDimI1, DDimI2, LowerExtrapolationRule1, UpperExtrapolationRule1, LowerExtrapolationRule2, UpperExtrapolationRule2>> : std::true_type {};

/**
 *  @brief A helper to check if SplineEvaluator is admissible for SplineBuilder
 *  @tparam Builder The builder type to be checked if it is admissible for Evaluator
 *  @tparam Evaluator The evaluator type to be checked if it is admissible for Builder
 */
template <class Builder, class Evaluator>
inline constexpr bool is_evaluator_admissible_v = is_evaluator_admissible<Builder, Evaluator>::value;

} // namespace ddc

