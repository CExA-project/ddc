// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "spline_builder.hpp"
#include "spline_builder_2d.hpp"
#include "spline_builder_3d.hpp"
#include "spline_builder_closures.hpp"
#include "spline_evaluator.hpp"
#include "spline_evaluator_2d.hpp"
#include "spline_evaluator_3d.hpp"
#include "spline_evaluator_nd.hpp"

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
        ddc::SplineBuilderClosure SBCUpper,
        SplineSolver Solver>
struct is_spline_builder<SplineBuilder<
        ExecSpace,
        MemorySpace,
        BSplines,
        InterpolationDDim,
        SBCLower,
        SBCUpper,
        Solver>> : std::true_type
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
        concepts::discrete_dimension DDimI1,
        concepts::discrete_dimension DDimI2,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2,
        ddc::SplineSolver Solver>
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
        SBCUpper2,
        Solver>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineBuilder2D
 *  @tparam T The type to be checked if is a SplineBuilder2D
 */
template <class T>
inline constexpr bool is_spline_builder2d_v = is_spline_builder2d<T>::value;

template <class T>
struct is_spline_builder3d : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class BSpline3,
        class DDimI1,
        class DDimI2,
        class DDimI3,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2,
        ddc::SplineBuilderClosure SBCLower3,
        ddc::SplineBuilderClosure SBCUpper3,
        ddc::SplineSolver Solver>
struct is_spline_builder3d<SplineBuilder3D<
        ExecSpace,
        MemorySpace,
        BSpline1,
        BSpline2,
        BSpline3,
        DDimI1,
        DDimI2,
        DDimI3,
        SBCLower1,
        SBCUpper1,
        SBCLower2,
        SBCUpper2,
        SBCLower3,
        SBCUpper3,
        Solver>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineBuilder3D
 *  @tparam T The type to be checked if is a SplineBuilder3D
 */
template <class T>
inline constexpr bool is_spline_builder3d_v = is_spline_builder3d<T>::value;

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

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class EvaluationDDim,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_spline_evaluator<SplineEvaluatorND<
        ExecSpace,
        MemorySpace,
        ddc::TypeSeq<BSplines>,
        ddc::TypeSeq<EvaluationDDim>,
        ddc::TypeSeq<LowerExtrapolationRule, UpperExtrapolationRule>>> : std::true_type
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
struct is_spline_evaluator2d<SplineEvaluatorND<
        ExecSpace,
        MemorySpace,
        ddc::TypeSeq<BSpline1, BSpline2>,
        ddc::TypeSeq<EvaluationDDim1, EvaluationDDim2>,
        ddc::TypeSeq<
                LowerExtrapolationRule1,
                UpperExtrapolationRule1,
                LowerExtrapolationRule2,
                UpperExtrapolationRule2>>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineEvaluator2D
 *  @tparam T The type to be checked if is a SplineEvaluator2D
 */
template <class T>
inline constexpr bool is_spline_evaluator2d_v = is_spline_evaluator2d<T>::value;

template <class T>
struct is_spline_evaluator3d : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class BSpline3,
        class EvaluationDDim1,
        class EvaluationDDim2,
        class EvaluationDDim3,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2,
        class LowerExtrapolationRule3,
        class UpperExtrapolationRule3>
struct is_spline_evaluator3d<SplineEvaluator3D<
        ExecSpace,
        MemorySpace,
        BSpline1,
        BSpline2,
        BSpline3,
        EvaluationDDim1,
        EvaluationDDim2,
        EvaluationDDim3,
        LowerExtrapolationRule1,
        UpperExtrapolationRule1,
        LowerExtrapolationRule2,
        UpperExtrapolationRule2,
        LowerExtrapolationRule3,
        UpperExtrapolationRule3>> : std::true_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSpline1,
        class BSpline2,
        class BSpline3,
        class EvaluationDDim1,
        class EvaluationDDim2,
        class EvaluationDDim3,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2,
        class LowerExtrapolationRule3,
        class UpperExtrapolationRule3>
struct is_spline_evaluator3d<SplineEvaluatorND<
        ExecSpace,
        MemorySpace,
        ddc::TypeSeq<BSpline1, BSpline2, BSpline3>,
        ddc::TypeSeq<EvaluationDDim1, EvaluationDDim2, EvaluationDDim3>,
        ddc::TypeSeq<
                LowerExtrapolationRule1,
                UpperExtrapolationRule1,
                LowerExtrapolationRule2,
                UpperExtrapolationRule2,
                LowerExtrapolationRule3,
                UpperExtrapolationRule3>>> : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineEvaluator3D
 *  @tparam T The type to be checked if is a SplineEvaluator3D
 */
template <class T>
inline constexpr bool is_spline_evaluator3d_v = is_spline_evaluator3d<T>::value;

template <class T>
struct is_spline_evaluatornd : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class EvaluationDDim,
        class ExtrapolationRule>
struct is_spline_evaluatornd<
        SplineEvaluatorND<ExecSpace, MemorySpace, BSplines, EvaluationDDim, ExtrapolationRule>>
    : std::true_type
{
};

/**
 *  @brief A helper to check if T is a SplineEvaluatorND
 *  @tparam T The type to be checked if is a SplineEvaluatorND
 */
template <class T>
inline constexpr bool is_spline_evaluatornd_v = is_spline_evaluatornd<T>::value;

template <class Builder, class Evaluator>
struct is_evaluator_admissible : std::false_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class DDim,
        ddc::SplineBuilderClosure SBCLower,
        ddc::SplineBuilderClosure SBCUpper,
        SplineSolver Solver,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_evaluator_admissible<
        SplineBuilder<ExecSpace, MemorySpace, BSplines, DDim, SBCLower, SBCUpper, Solver>,
        SplineEvaluator<
                ExecSpace,
                MemorySpace,
                BSplines,
                DDim,
                LowerExtrapolationRule,
                UpperExtrapolationRule>> : std::true_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        concepts::discrete_dimension DDimI1,
        concepts::discrete_dimension DDimI2,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2,
        SplineSolver Solver,
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
                SBCUpper2,
                Solver>,
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

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        class BSplines3,
        class DDimI1,
        class DDimI2,
        class DDimI3,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2,
        ddc::SplineBuilderClosure SBCLower3,
        ddc::SplineBuilderClosure SBCUpper3,
        SplineSolver Solver,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2,
        class LowerExtrapolationRule3,
        class UpperExtrapolationRule3>
struct is_evaluator_admissible<
        SplineBuilder3D<
                ExecSpace,
                MemorySpace,
                BSplines1,
                BSplines2,
                BSplines3,
                DDimI1,
                DDimI2,
                DDimI3,
                SBCLower1,
                SBCUpper1,
                SBCLower2,
                SBCUpper2,
                SBCLower3,
                SBCUpper3,
                Solver>,
        SplineEvaluator3D<
                ExecSpace,
                MemorySpace,
                BSplines1,
                BSplines2,
                BSplines3,
                DDimI1,
                DDimI2,
                DDimI3,
                LowerExtrapolationRule1,
                UpperExtrapolationRule1,
                LowerExtrapolationRule2,
                UpperExtrapolationRule2,
                LowerExtrapolationRule3,
                UpperExtrapolationRule3>> : std::true_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class DDim,
        ddc::SplineBuilderClosure SBCLower,
        ddc::SplineBuilderClosure SBCUpper,
        SplineSolver Solver,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule>
struct is_evaluator_admissible<
        SplineBuilder<ExecSpace, MemorySpace, BSplines, DDim, SBCLower, SBCUpper, Solver>,
        SplineEvaluatorND<
                ExecSpace,
                MemorySpace,
                TypeSeq<BSplines>,
                TypeSeq<DDim>,
                TypeSeq<LowerExtrapolationRule, UpperExtrapolationRule>>> : std::true_type
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
        SplineSolver Solver,
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
                SBCUpper2,
                Solver>,
        SplineEvaluatorND<
                ExecSpace,
                MemorySpace,
                TypeSeq<BSplines1, BSplines2>,
                TypeSeq<DDimI1, DDimI2>,
                TypeSeq<LowerExtrapolationRule1,
                        UpperExtrapolationRule1,
                        LowerExtrapolationRule2,
                        UpperExtrapolationRule2>>> : std::true_type
{
};

template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        class BSplines3,
        class DDimI1,
        class DDimI2,
        class DDimI3,
        ddc::SplineBuilderClosure SBCLower1,
        ddc::SplineBuilderClosure SBCUpper1,
        ddc::SplineBuilderClosure SBCLower2,
        ddc::SplineBuilderClosure SBCUpper2,
        ddc::SplineBuilderClosure SBCLower3,
        ddc::SplineBuilderClosure SBCUpper3,
        SplineSolver Solver,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2,
        class LowerExtrapolationRule3,
        class UpperExtrapolationRule3>
struct is_evaluator_admissible<
        SplineBuilder3D<
                ExecSpace,
                MemorySpace,
                BSplines1,
                BSplines2,
                BSplines3,
                DDimI1,
                DDimI2,
                DDimI3,
                SBCLower1,
                SBCUpper1,
                SBCLower2,
                SBCUpper2,
                SBCLower3,
                SBCUpper3,
                Solver>,
        SplineEvaluatorND<
                ExecSpace,
                MemorySpace,
                TypeSeq<BSplines1, BSplines2, BSplines3>,
                TypeSeq<DDimI1, DDimI2, DDimI3>,
                TypeSeq<LowerExtrapolationRule1,
                        UpperExtrapolationRule1,
                        LowerExtrapolationRule2,
                        UpperExtrapolationRule2,
                        LowerExtrapolationRule3,
                        UpperExtrapolationRule3>>> : std::true_type
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
