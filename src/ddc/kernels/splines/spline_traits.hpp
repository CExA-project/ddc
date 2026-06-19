// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "spline_builder.hpp"
#include "spline_builder_2d.hpp"
#include "spline_builder_closures.hpp"
#include "spline_evaluator.hpp"
#include "spline_evaluator_2d.hpp"

namespace ddc {

template <class T>
struct is_spline_builder : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationDDim,
        ddc::SplineBuilderClosure SBCLower,
        ddc::SplineBuilderClosure SBCUpper>
struct is_spline_builder<
        SplineBuilder<ExecSpace, MemorySpace, BSplines, InterpolationDDim, SBCLower, SBCUpper>>
    : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineBuilder
 *  @tparam T The type to be checked if is a SplineBuilder
 */
template <class T>
inline constexpr bool is_spline_builder_v = is_spline_builder<T>::value;

template <class T>
struct is_spline_builder2d : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class DDimI1,
        class DDimI2,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2>
struct is_spline_builder2d<SplineBuilder2D<
        ExecSpace,
        MemorySpace,
        BSpline1,
        BSpline2,
        DDimI1,
        DDimI2,
        SBCLower1,
        SBCUpper1,
        SBCLower2,
        SBCUpper2>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineBuilder2D
 *  @tparam T The type to be checked if is a SplineBuilder2D
 */
template <class T>
inline constexpr bool is_spline_builder2d_v = is_spline_builder2d<T>::value;

template <class T>
struct is_spline_evaluator : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class EvaluationDDim,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_spline_evaluator<SplineEvaluator<
        ExecSpace,
        MemorySpace,
        BSplines,
        EvaluationDDim,
        LowerExtrapolationRule,
        UpperExtrapolationRule>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineEvaluator
 *  @tparam T The type to be checked if is a SplineEvaluator
 */
template <class T>
inline constexpr bool is_spline_evaluator_v = is_spline_evaluator<T>::value;

template <class T>
struct is_spline_evaluator2d : std::false_type
{
};

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
struct is_spline_evaluator2d<SplineEvaluator2D<
        ExecSpace,
        MemorySpace,
        BSpline1,
        BSpline2,
        EvaluationDDim1,
        EvaluationDDim2,
        LowerExtrapolationRule1,
        UpperExtrapolationRule1,
        LowerExtrapolationRule2,
        UpperExtrapolationRule2>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineEvaluator2D
 *  @tparam T The type to be checked if is a SplineEvaluator2D
 */
template <class T>
inline constexpr bool is_spline_evaluator2d_v = is_spline_evaluator2d<T>::value;

template <class Builder, class Evaluator>
struct is_evaluator_admissible : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class InterpolationDDim,
        ddc::SplineBuilderClosure SBCLower,
        ddc::SplineBuilderClosure SBCUpper,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_evaluator_admissible<
        SplineBuilder<ExecSpace, MemorySpace, BSplines, InterpolationDDim, SBCLower, SBCUpper>,
        SplineEvaluator<
                ExecSpace,
                MemorySpace,
                BSplines,
                InterpolationDDim,
                LowerExtrapolationRule,
                UpperExtrapolationRule>> : std::true_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        class DDimI1,
        class DDimI2,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2>
struct is_evaluator_admissible<
        SplineBuilder2D<
                ExecSpace,
                MemorySpace,
                BSplines1,
                BSplines2,
                DDimI1,
                DDimI2,
                SBCLower1,
                SBCUpper1,
                SBCLower2,
                SBCUpper2>,
        SplineEvaluator2D<
                ExecSpace,
                MemorySpace,
                BSplines1,
                BSplines2,
                DDimI1,
                DDimI2,
                LowerExtrapolationRule1,
                UpperExtrapolationRule1,
                LowerExtrapolationRule2,
                UpperExtrapolationRule2>> : std::true_type
{
};

/**
 *  @brief A helper to check if SplineEvaluator is admissible for SplineBuilder
 *  @tparam Builder The builder type to be checked if it is admissible for Evaluator
 *  @tparam Evaluator The evaluator type to be checked if it is admissible for Builder
 */
template <class Builder, class Evaluator>
inline constexpr bool is_evaluator_admissible_v
        = is_evaluator_admissible<Builder, Evaluator>::value;

} // namespace ddc
