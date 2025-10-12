// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

#include "integrals.hpp"
#include "periodic_extrapolation_rule.hpp"

namespace ddc {

/**
 * @brief A class to evaluate, differentiate or integrate a 3D spline function.
 *
 * A class which contains an operator () which can be used to evaluate, differentiate or integrate a 3D spline function.
 *
 * @tparam ExecSpace The Kokkos execution space on which the spline evaluation is performed.
 * @tparam MemorySpace The Kokkos memory space on which the data (spline coefficients and evaluation) is stored.
 * @tparam BSplines1 The discrete dimension representing the B-splines along the first dimension of interest.
 * @tparam BSplines2 The discrete dimension representing the B-splines along the second dimension of interest.
 * @tparam BSplines3 The discrete dimension representing the B-splines along the third dimension of interest.
 * @tparam EvaluationDDim1 The first discrete dimension on which evaluation points are defined.
 * @tparam EvaluationDDim2 The second discrete dimension on which evaluation points are defined.
 * @tparam EvaluationDDim3 The third discrete dimension on which evaluation points are defined.
 * @tparam LowerExtrapolationRule1 The lower extrapolation rule type along first dimension of interest.
 * @tparam UpperExtrapolationRule1 The upper extrapolation rule type along first dimension of interest.
 * @tparam LowerExtrapolationRule2 The lower extrapolation rule type along second dimension of interest.
 * @tparam UpperExtrapolationRule2 The upper extrapolation rule type along second dimension of interest.
 * @tparam LowerExtrapolationRule3 The lower extrapolation rule type along third dimension of interest.
 * @tparam UpperExtrapolationRule3 The upper extrapolation rule type along third dimension of interest.
 */
template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        class BSplines3,
        class EvaluationDDim1,
        class EvaluationDDim2,
        class EvaluationDDim3,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2,
        class LowerExtrapolationRule3,
        class UpperExtrapolationRule3>
class SplineEvaluator3D
{
private:
    /**
     * @brief Tag to indicate that the value of the spline should be evaluated.
     */
    struct eval_type
    {
    };

    /**
     * @brief Tag to indicate that derivative of the spline should be evaluated.
     */
    struct eval_deriv_type
    {
    };

public:
    /// @brief The type of the first evaluation continuous dimension used by this class.
    using continuous_dimension_type1 = typename BSplines1::continuous_dimension_type;

    /// @brief The type of the second evaluation continuous dimension used by this class.
    using continuous_dimension_type2 = typename BSplines2::continuous_dimension_type;

    /// @brief The type of the third evaluation continuous dimension used by this class.
    using continuous_dimension_type3 = typename BSplines3::continuous_dimension_type;

    /// @brief The type of the Kokkos execution space used by this class.
    using exec_space = ExecSpace;

    /// @brief The type of the Kokkos memory space used by this class.
    using memory_space = MemorySpace;

    /// @brief The type of the first discrete dimension of interest used by this class.
    using evaluation_discrete_dimension_type1 = EvaluationDDim1;

    /// @brief The type of the second discrete dimension of interest used by this class.
    using evaluation_discrete_dimension_type2 = EvaluationDDim2;

    /// @brief The type of the third discrete dimension of interest used by this class.
    using evaluation_discrete_dimension_type3 = EvaluationDDim3;

    /// @brief The discrete dimension representing the B-splines along first dimension.
    using bsplines_type1 = BSplines1;

    /// @brief The discrete dimension representing the B-splines along second dimension.
    using bsplines_type2 = BSplines2;

    /// @brief The discrete dimension representing the B-splines along third dimension.
    using bsplines_type3 = BSplines3;

    /// @brief The type of the domain for the 1D evaluation mesh along first dimension used by this class.
    using evaluation_domain_type1 = ddc::DiscreteDomain<evaluation_discrete_dimension_type1>;

    /// @brief The type of the domain for the 1D evaluation mesh along second dimension used by this class.
    using evaluation_domain_type2 = ddc::DiscreteDomain<evaluation_discrete_dimension_type2>;

    /// @brief The type of the domain for the 1D evaluation mesh along third dimension used by this class.
    using evaluation_domain_type3 = ddc::DiscreteDomain<evaluation_discrete_dimension_type3>;

    /// @brief The type of the domain for the 3D evaluation mesh used by this class.
    using evaluation_domain_type = ddc::DiscreteDomain<
            evaluation_discrete_dimension_type1,
            evaluation_discrete_dimension_type2,
            evaluation_discrete_dimension_type3>;

    /**
     * @brief The type of the whole domain representing evaluation points.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_evaluation_domain_type = BatchedInterpolationDDom;

    /// @brief The type of the 1D spline domain corresponding to the first dimension of interest.
    using spline_domain_type1 = ddc::DiscreteDomain<bsplines_type1>;

    /// @brief The type of the 1D spline domain corresponding to the second dimension of interest.
    using spline_domain_type2 = ddc::DiscreteDomain<bsplines_type2>;

    /// @brief The type of the 1D spline domain corresponding to the third dimension of interest.
    using spline_domain_type3 = ddc::DiscreteDomain<bsplines_type3>;

    /// @brief The type of the 3D spline domain corresponding to the dimensions of interest.
    using spline_domain_type = ddc::DiscreteDomain<bsplines_type1, bsplines_type2, bsplines_type3>;

    /**
     * @brief The type of the batch domain (obtained by removing the dimensions of interest
     * from the whole domain).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batch_domain_type = typename ddc::remove_dims_of_t<
            BatchedInterpolationDDom,
            evaluation_discrete_dimension_type1,
            evaluation_discrete_dimension_type2,
            evaluation_discrete_dimension_type3>;

    /**
     * @brief The type of the whole spline domain (cartesian product of 3D spline domain
     * and batch domain) preserving the underlying memory layout (order of dimensions).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_spline_domain_type =
            typename ddc::detail::convert_type_seq_to_discrete_domain_t<ddc::type_seq_replace_t<
                    ddc::to_type_seq_t<BatchedInterpolationDDom>,
                    ddc::detail::TypeSeq<
                            evaluation_discrete_dimension_type1,
                            evaluation_discrete_dimension_type2,
                            evaluation_discrete_dimension_type3>,
                    ddc::detail::TypeSeq<bsplines_type1, bsplines_type2, bsplines_type3>>>;

    /// @brief The type of the extrapolation rule at the lower boundary along the first dimension.
    using lower_extrapolation_rule_1_type = LowerExtrapolationRule1;

    /// @brief The type of the extrapolation rule at the upper boundary along the first dimension.
    using upper_extrapolation_rule_1_type = UpperExtrapolationRule1;

    /// @brief The type of the extrapolation rule at the lower boundary along the second dimension.
    using lower_extrapolation_rule_2_type = LowerExtrapolationRule2;

    /// @brief The type of the extrapolation rule at the upper boundary along the second dimension.
    using upper_extrapolation_rule_2_type = UpperExtrapolationRule2;

    /// @brief The type of the extrapolation rule at the lower boundary along the third dimension.
    using lower_extrapolation_rule_3_type = LowerExtrapolationRule3;

    /// @brief The type of the extrapolation rule at the upper boundary along the third dimension.
    using upper_extrapolation_rule_3_type = UpperExtrapolationRule3;

private:
    LowerExtrapolationRule1 m_lower_extrap_rule_1;

    UpperExtrapolationRule1 m_upper_extrap_rule_1;

    LowerExtrapolationRule2 m_lower_extrap_rule_2;

    UpperExtrapolationRule2 m_upper_extrap_rule_2;

    LowerExtrapolationRule3 m_lower_extrap_rule_3;

    UpperExtrapolationRule3 m_upper_extrap_rule_3;

public:
    static_assert(
            std::is_same_v<LowerExtrapolationRule1,
                            typename ddc::PeriodicExtrapolationRule<continuous_dimension_type1>>
                            == bsplines_type1::is_periodic()
                    && std::is_same_v<
                               UpperExtrapolationRule1,
                               typename ddc::PeriodicExtrapolationRule<continuous_dimension_type1>>
                               == bsplines_type1::is_periodic()
                    && std::is_same_v<
                               LowerExtrapolationRule2,
                               typename ddc::PeriodicExtrapolationRule<continuous_dimension_type2>>
                               == bsplines_type2::is_periodic()
                    && std::is_same_v<
                               UpperExtrapolationRule2,
                               typename ddc::PeriodicExtrapolationRule<continuous_dimension_type2>>
                               == bsplines_type2::is_periodic()
                    && std::is_same_v<
                               LowerExtrapolationRule3,
                               typename ddc::PeriodicExtrapolationRule<continuous_dimension_type3>>
                               == bsplines_type3::is_periodic()
                    && std::is_same_v<
                               UpperExtrapolationRule3,
                               typename ddc::PeriodicExtrapolationRule<continuous_dimension_type3>>
                               == bsplines_type3::is_periodic(),
            "PeriodicExtrapolationRule has to be used if and only if dimension is periodic");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    LowerExtrapolationRule1,
                    ddc::Coordinate<continuous_dimension_type1>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "LowerExtrapolationRule1::operator() has to be callable "
            "with usual arguments.");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    UpperExtrapolationRule1,
                    ddc::Coordinate<continuous_dimension_type1>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "UpperExtrapolationRule1::operator() has to be callable "
            "with usual arguments.");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    LowerExtrapolationRule2,
                    ddc::Coordinate<continuous_dimension_type2>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "LowerExtrapolationRule2::operator() has to be callable "
            "with usual arguments.");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    UpperExtrapolationRule2,
                    ddc::Coordinate<continuous_dimension_type2>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "UpperExtrapolationRule2::operator() has to be callable "
            "with usual arguments.");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    LowerExtrapolationRule3,
                    ddc::Coordinate<continuous_dimension_type3>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "LowerExtrapolationRule3::operator() has to be callable "
            "with usual arguments.");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    UpperExtrapolationRule3,
                    ddc::Coordinate<continuous_dimension_type3>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "UpperExtrapolationRule3::operator() has to be callable "
            "with usual arguments.");

    /**
     * @brief Build a SplineEvaluator3D acting on batched_spline_domain.
     *
     * @param lower_extrap_rule1 The extrapolation rule at the lower boundary along the first dimension.
     * @param upper_extrap_rule1 The extrapolation rule at the upper boundary along the first dimension.
     * @param lower_extrap_rule2 The extrapolation rule at the lower boundary along the second dimension.
     * @param upper_extrap_rule2 The extrapolation rule at the upper boundary along the second dimension.
     * @param lower_extrap_rule3 The extrapolation rule at the lower boundary along the third dimension.
     * @param upper_extrap_rule3 The extrapolation rule at the upper boundary along the third dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    explicit SplineEvaluator3D(
            LowerExtrapolationRule1 const& lower_extrap_rule1,
            UpperExtrapolationRule1 const& upper_extrap_rule1,
            LowerExtrapolationRule2 const& lower_extrap_rule2,
            UpperExtrapolationRule2 const& upper_extrap_rule2,
            LowerExtrapolationRule3 const& lower_extrap_rule3,
            UpperExtrapolationRule3 const& upper_extrap_rule3)
        : m_lower_extrap_rule_1(lower_extrap_rule1)
        , m_upper_extrap_rule_1(upper_extrap_rule1)
        , m_lower_extrap_rule_2(lower_extrap_rule2)
        , m_upper_extrap_rule_2(upper_extrap_rule2)
        , m_lower_extrap_rule_3(lower_extrap_rule3)
        , m_upper_extrap_rule_3(upper_extrap_rule3)
    {
    }

    /**
     * @brief Copy-constructs.
     *
     * @param x A reference to another SplineEvaluator.
     */
    SplineEvaluator3D(SplineEvaluator3D const& x) = default;

    /**
     * @brief Move-constructs.
     *
     * @param x An rvalue to another SplineEvaluator.
     */
    SplineEvaluator3D(SplineEvaluator3D&& x) = default;

    /// @brief Destructs.
    ~SplineEvaluator3D() = default;

    /**
     * @brief Copy-assigns.
     *
     * @param x A reference to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluator3D& operator=(SplineEvaluator3D const& x) = default;

    /**
     * @brief Move-assigns.
     *
     * @param x An rvalue to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluator3D& operator=(SplineEvaluator3D&& x) = default;

    /**
     * @brief Get the lower extrapolation rule along the first dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The lower extrapolation rule along the first dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    lower_extrapolation_rule_1_type lower_extrapolation_rule_dim_1() const
    {
        return m_lower_extrap_rule_1;
    }

    /**
     * @brief Get the upper extrapolation rule along the first dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The upper extrapolation rule along the first dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    upper_extrapolation_rule_1_type upper_extrapolation_rule_dim_1() const
    {
        return m_upper_extrap_rule_1;
    }

    /**
     * @brief Get the lower extrapolation rule along the second dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The lower extrapolation rule along the second dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    lower_extrapolation_rule_2_type lower_extrapolation_rule_dim_2() const
    {
        return m_lower_extrap_rule_2;
    }

    /**
     * @brief Get the upper extrapolation rule along the second dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The upper extrapolation rule along the second dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    upper_extrapolation_rule_2_type upper_extrapolation_rule_dim_2() const
    {
        return m_upper_extrap_rule_2;
    }

    /**
     * @brief Get the lower extrapolation rule along the third dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The lower extrapolation rule along the third dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    lower_extrapolation_rule_3_type lower_extrapolation_rule_dim_3() const
    {
        return m_lower_extrap_rule_3;
    }

    /**
     * @brief Get the upper extrapolation rule along the third dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The upper extrapolation rule along the third dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    upper_extrapolation_rule_3_type upper_extrapolation_rule_dim_3() const
    {
        return m_upper_extrap_rule_3;
    }

    /**
     * @brief Evaluate 3D spline function (described by its spline coefficients) at a given coordinate.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * Remark: calling SplineBuilder3D then SplineEvaluator3D corresponds to a 3D spline interpolation.
     *
     * @param coord_eval The coordinate where the spline is evaluated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The value of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double operator()(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval(coord_eval, spline_coef);
    }

    /**
     * @brief Evaluate 3D spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD evaluation. This is a batched 3D evaluation. This means that for each slice of coordinates
     * identified by a batch_domain_type::discrete_element_type, the evaluation is performed with the 3D set of
     * spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilder3D then SplineEvaluator3D corresponds to a 3D spline interpolation.
     *
     * @param[out] spline_eval The values of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is evaluated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void operator()(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_evaluate_3d",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3)
                                        = eval(coords_eval_3D(i1, i2, i3), spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Evaluate 3D spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilder3D then SplineEvaluator3D corresponds to a 3D spline interpolation.
     *
     * @param[out] spline_eval The values of the 3D spline function at their coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void operator()(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_evaluate_3d",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3)
                                        = eval(coord_eval_3D(i1, i2, i3), spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) at a given coordinate along first dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_dim_1(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_deriv_type, eval_type, eval_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) at a given coordinate along second dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_dim_2(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_type, eval_deriv_type, eval_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) at a given coordinate along third dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_dim_3(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_type, eval_type, eval_deriv_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) at a given coordinate along the first and second dimensions.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_1_and_2(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_deriv_type, eval_deriv_type, eval_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) at a given coordinate along the second and third dimensions.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_2_and_3(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_type, eval_deriv_type, eval_deriv_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) at a given coordinate along the first and third dimensions.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_1_and_3(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_deriv_type, eval_type, eval_deriv_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) at a given coordinate along the dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_1_2_3(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<
                eval_deriv_type,
                eval_deriv_type,
                eval_deriv_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) at a given coordinate along a specified dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * @tparam InterestDim Dimension along which differentiation is performed.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class InterestDim, class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert(
                std::is_same_v<InterestDim, continuous_dimension_type1>
                || std::is_same_v<InterestDim, continuous_dimension_type2>
                || std::is_same_v<InterestDim, continuous_dimension_type3>);
        if constexpr (std::is_same_v<InterestDim, continuous_dimension_type1>) {
            return deriv_dim_1(coord_eval, spline_coef);
        } else if constexpr (std::is_same_v<InterestDim, continuous_dimension_type2>) {
            return deriv_dim_2(coord_eval, spline_coef);
        } else if constexpr (std::is_same_v<InterestDim, continuous_dimension_type3>) {
            return deriv_dim_3(coord_eval, spline_coef);
        }
    }

    /**
     * @brief Double-differentiate 3D spline function (described by its spline coefficients) at a given coordinate along specified dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * Note: double-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     *
     * @param coord_eval The coordinate where the spline is double-differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class InterestDim1, class InterestDim2, class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv2(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert(
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim1, continuous_dimension_type2>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim1, continuous_dimension_type1>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>));

        if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>)) {
            return deriv_1_and_2(coord_eval, spline_coef);
        } else if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type2>
                 && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)) {
            return deriv_2_and_3(coord_eval, spline_coef);
        } else if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)) {
            return deriv_1_and_3(coord_eval, spline_coef);
        }
    }

    /**
     * @brief Triple-differentiate 3D spline function (described by its spline coefficients) at a given coordinate along specified dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder3D.
     *
     * Note: triple-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     * @tparam InterestDim3 Third dimension along which differentiation is performed.
     *
     * @param coord_eval The coordinate where the spline is triple-differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 3D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class InterestDim3,
            class Layout,
            class... CoordsDims>
    KOKKOS_FUNCTION double deriv3(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert(
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>
                 && std::is_same_v<InterestDim3, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim3, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim3, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>));

        return deriv_1_2_3(coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) on a mesh along first dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD evaluation. This is a batched 3D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_dim_1(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_3d_dim_1",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_type,
                                        eval_type>(coords_eval_3D(i1, i2, i3), spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) on a mesh along first dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_dim_1(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_3d_dim_1",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_type,
                                        eval_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) on a mesh along second dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD differentiation. This is a batched 3D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_dim_2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_3d_dim_2",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_type,
                                        eval_deriv_type,
                                        eval_type>(coords_eval_3D(i1, i2, i3), spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) on a mesh along second dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_dim_2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_3d_dim_2",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_type,
                                        eval_deriv_type,
                                        eval_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) on a mesh along third dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD differentiation. This is a batched 3D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_dim_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_3d_dim_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3)
                                        = eval_no_bc<eval_type, eval_type, eval_deriv_type>(
                                                coords_eval_3D(i1, i2, i3),
                                                spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 3D spline function (described by its spline coefficients) on a mesh along third dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_dim_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_3d_dim_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_type,
                                        eval_type,
                                        eval_deriv_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the first and second dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD cross-differentiation. This is a batched 3D cross-differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the cross-differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_1_and_2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim1_2",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_deriv_type,
                                        eval_type>(coords_eval_3D(i1, i2, i3), spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the first and second dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_1_and_2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim1_2",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_deriv_type,
                                        eval_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the second and third dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD cross-differentiation. This is a batched 3D cross-differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the cross-differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_2_and_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim2_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3)
                                        = eval_no_bc<eval_type, eval_deriv_type, eval_deriv_type>(
                                                coords_eval_3D(i1, i2, i3),
                                                spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the second and third dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_2_and_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim2_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_type,
                                        eval_deriv_type,
                                        eval_deriv_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the first and third dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD cross-differentiation. This is a batched 3D cross-differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the cross-differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_1_and_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim1_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3)
                                        = eval_no_bc<eval_deriv_type, eval_type, eval_deriv_type>(
                                                coords_eval_3D(i1, i2, i3),
                                                spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the first and third dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_1_and_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim1_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_type,
                                        eval_deriv_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD cross-differentiation. This is a batched 3D cross-differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the cross-differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_1_2_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim1_2_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const coords_eval_3D = coords_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_deriv_type,
                                        eval_deriv_type>(
                                        coords_eval_3D(i1, i2, i3),
                                        spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Cross-differentiate 3D spline function (described by its spline coefficients) on a mesh along the dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv_1_2_3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        evaluation_domain_type3 const evaluation_domain3(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_3d_dim1_2_3",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_3D = spline_eval[j];
                    auto const spline_coef_3D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            for (auto const i3 : evaluation_domain3) {
                                ddc::Coordinate<
                                        continuous_dimension_type1,
                                        continuous_dimension_type2,
                                        continuous_dimension_type3>
                                        coord_eval_3D(
                                                ddc::coordinate(i1),
                                                ddc::coordinate(i2),
                                                ddc::coordinate(i3));
                                spline_eval_3D(i1, i2, i3) = eval_no_bc<
                                        eval_deriv_type,
                                        eval_deriv_type,
                                        eval_deriv_type>(coord_eval_3D, spline_coef_3D);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along a specified dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD evaluation. This is a batched 3D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam InterestDim Dimension along which differentiation is performed.
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class InterestDim,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        static_assert(
                std::is_same_v<InterestDim, continuous_dimension_type1>
                || std::is_same_v<InterestDim, continuous_dimension_type2>
                || std::is_same_v<InterestDim, continuous_dimension_type3>);
        if constexpr (std::is_same_v<InterestDim, continuous_dimension_type1>) {
            return deriv_dim_1(spline_eval, coords_eval, spline_coef);
        } else if constexpr (std::is_same_v<InterestDim, continuous_dimension_type2>) {
            return deriv_dim_2(spline_eval, coords_eval, spline_coef);
        } else if constexpr (std::is_same_v<InterestDim, continuous_dimension_type3>) {
            return deriv_dim_3(spline_eval, coords_eval, spline_coef);
        }
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along a specified dimension of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam InterestDim Dimension along which differentiation is performed.
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class InterestDim, class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        static_assert(
                std::is_same_v<InterestDim, continuous_dimension_type1>
                || std::is_same_v<InterestDim, continuous_dimension_type2>
                || std::is_same_v<InterestDim, continuous_dimension_type3>);
        if constexpr (std::is_same_v<InterestDim, continuous_dimension_type1>) {
            return deriv_dim_1(spline_eval, spline_coef);
        } else if constexpr (std::is_same_v<InterestDim, continuous_dimension_type2>) {
            return deriv_dim_2(spline_eval, spline_coef);
        } else if constexpr (std::is_same_v<InterestDim, continuous_dimension_type3>) {
            return deriv_dim_3(spline_eval, spline_coef);
        }
    }

    /**
     * @brief Double-differentiate 3D spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD evaluation. This is a batched 3D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Note: double-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        static_assert(
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim1, continuous_dimension_type2>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim1, continuous_dimension_type1>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>));

        if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>)) {
            return deriv_1_and_2(spline_eval, coords_eval, spline_coef);
        } else if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type2>
                 && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)) {
            return deriv_2_and_3(spline_eval, coords_eval, spline_coef);
        } else if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)) {
            return deriv_1_and_3(spline_eval, coords_eval, spline_coef);
        }
    }

    /**
     * @brief Double-differentiate 3D spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Note: double-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class Layout1,
            class Layout2,
            class BatchedInterpolationDDom>
    void deriv2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        static_assert(
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim1, continuous_dimension_type2>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim1, continuous_dimension_type1>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>));

        if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>)) {
            return deriv_1_and_2(spline_eval, spline_coef);
        } else if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type2>
                 && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)) {
            return deriv_2_and_3(spline_eval, spline_coef);
        } else if constexpr (
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>)) {
            return deriv_1_and_3(spline_eval, spline_coef);
        }
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along a specified dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD evaluation. This is a batched 3D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Note: triple-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     * @tparam InterestDim3 Third dimension along which differentiation is performed.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 3D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class InterestDim3,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    BatchedInterpolationDDom,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout3,
                    memory_space> const spline_coef) const
    {
        static_assert(
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>
                 && std::is_same_v<InterestDim3, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim3, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim3, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>));

        return deriv_1_2_3(spline_eval, coords_eval, spline_coef);
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a 3D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a multidimensional evaluation. This is a batched 3D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 3D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Note: triple-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     * @tparam InterestDim3 Third dimension along which differentiation is performed.
     *
     * @param[out] spline_eval The derivatives of the 3D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class InterestDim3,
            class Layout1,
            class Layout2,
            class BatchedInterpolationDDom>
    void deriv3(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        static_assert(
                (std::is_same_v<InterestDim1, continuous_dimension_type1>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>
                 && std::is_same_v<InterestDim3, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim3, continuous_dimension_type1>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>
                    && std::is_same_v<InterestDim2, continuous_dimension_type3>)
                || (std::is_same_v<InterestDim2, continuous_dimension_type1>
                    && std::is_same_v<InterestDim3, continuous_dimension_type2>
                    && std::is_same_v<InterestDim1, continuous_dimension_type3>));

        return deriv_1_2_3(spline_eval, spline_coef);
    }

    /** @brief Perform batched 3D integrations of a spline function (described by its spline coefficients) along the dimensions of interest and store results on a subdomain of batch_domain.
     *
     * The spline coefficients represent a 3D spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilder3D.
     *
     * This is not a nD integration. This is a batched 3D integration.
     * This means that for each element of integrals, the integration is performed with the 3D set of
     * spline coefficients identified by the same DiscreteElement.
     *
     * @param[out] integrals The integrals of the 3D spline function on the subdomain of batch_domain. For practical reasons those are
     * stored in a ChunkSpan defined on a batch_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant.
     * @param[in] spline_coef A ChunkSpan storing the 3D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedDDom, class BatchedSplineDDom>
    void integrate(
            ddc::ChunkSpan<double, BatchedDDom, Layout1, memory_space> const integrals,
            ddc::ChunkSpan<double const, BatchedSplineDDom, Layout2, memory_space> const
                    spline_coef) const
    {
        static_assert(
                ddc::type_seq_contains_v<
                        ddc::detail::TypeSeq<bsplines_type1, bsplines_type2, bsplines_type3>,
                        to_type_seq_t<BatchedSplineDDom>>,
                "The spline coefficients domain must contain the bsplines dimensions");
        using batch_domain_type = ddc::
                remove_dims_of_t<BatchedSplineDDom, bsplines_type1, bsplines_type2, bsplines_type3>;
        static_assert(
                std::is_same_v<batch_domain_type, BatchedDDom>,
                "The integrals domain must only contain the batch dimensions");

        batch_domain_type batch_domain(integrals.domain());
        ddc::Chunk values1_alloc(
                ddc::DiscreteDomain<bsplines_type1>(spline_coef.domain()),
                ddc::KokkosAllocator<double, memory_space>());
        ddc::ChunkSpan values1 = values1_alloc.span_view();
        ddc::integrals(exec_space(), values1);
        ddc::Chunk values2_alloc(
                ddc::DiscreteDomain<bsplines_type2>(spline_coef.domain()),
                ddc::KokkosAllocator<double, memory_space>());
        ddc::ChunkSpan values2 = values2_alloc.span_view();
        ddc::integrals(exec_space(), values2);
        ddc::Chunk values3_alloc(
                ddc::DiscreteDomain<bsplines_type3>(spline_coef.domain()),
                ddc::KokkosAllocator<double, memory_space>());
        ddc::ChunkSpan values3 = values3_alloc.span_view();
        ddc::integrals(exec_space(), values3);

        ddc::parallel_for_each(
                "ddc_splines_integrate_bsplines",
                exec_space(),
                batch_domain,
                KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    integrals(j) = 0;
                    for (typename spline_domain_type1::discrete_element_type const i1 :
                         values1.domain()) {
                        for (typename spline_domain_type2::discrete_element_type const i2 :
                             values2.domain()) {
                            for (typename spline_domain_type3::discrete_element_type const i3 :
                                 values3.domain()) {
                                integrals(j) += spline_coef(i1, i2, i3, j) * values1(i1)
                                                * values2(i2) * values3(i3);
                            }
                        }
                    }
                });
    }

private:
    /**
     * @brief Evaluate the function on B-splines at the coordinate given.
     *
     * This function firstly deals with the boundary conditions and calls the SplineEvaluator3D::eval_no_bc function
     * to evaluate.
     *
     * @param[in] coord_eval The 3D coordinate where we want to evaluate.
     * @param[in] spline_coef The B-splines coefficients of the function we want to evaluate.
     * @param[out] vals1 A ChunkSpan with the not-null values of each function of the spline in the first dimension.
     * @param[out] vals2 A ChunkSpan with the not-null values of each function of the spline in the second dimension.
     *
     * @return A double with the value of the function at the coordinate given.
     *
     * @see SplineBoundaryValue
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval(
            ddc::Coordinate<CoordsDims...> coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        using Dim1 = continuous_dimension_type1;
        using Dim2 = continuous_dimension_type2;
        using Dim3 = continuous_dimension_type3;
        if constexpr (bsplines_type1::is_periodic()) {
            if (ddc::get<Dim1>(coord_eval) < ddc::discrete_space<bsplines_type1>().rmin()
                || ddc::get<Dim1>(coord_eval) > ddc::discrete_space<bsplines_type1>().rmax()) {
                ddc::get<Dim1>(coord_eval)
                        -= Kokkos::floor(
                                   (ddc::get<Dim1>(coord_eval)
                                    - ddc::discrete_space<bsplines_type1>().rmin())
                                   / ddc::discrete_space<bsplines_type1>().length())
                           * ddc::discrete_space<bsplines_type1>().length();
            }
        }
        if constexpr (bsplines_type2::is_periodic()) {
            if (ddc::get<Dim2>(coord_eval) < ddc::discrete_space<bsplines_type2>().rmin()
                || ddc::get<Dim2>(coord_eval) > ddc::discrete_space<bsplines_type2>().rmax()) {
                ddc::get<Dim2>(coord_eval)
                        -= Kokkos::floor(
                                   (ddc::get<Dim2>(coord_eval)
                                    - ddc::discrete_space<bsplines_type2>().rmin())
                                   / ddc::discrete_space<bsplines_type2>().length())
                           * ddc::discrete_space<bsplines_type2>().length();
            }
        }
        if constexpr (bsplines_type3::is_periodic()) {
            if (ddc::get<Dim3>(coord_eval) < ddc::discrete_space<bsplines_type3>().rmin()
                || ddc::get<Dim3>(coord_eval) > ddc::discrete_space<bsplines_type3>().rmax()) {
                ddc::get<Dim3>(coord_eval)
                        -= Kokkos::floor(
                                   (ddc::get<Dim3>(coord_eval)
                                    - ddc::discrete_space<bsplines_type3>().rmin())
                                   / ddc::discrete_space<bsplines_type3>().length())
                           * ddc::discrete_space<bsplines_type3>().length();
            }
        }
        if constexpr (!bsplines_type1::is_periodic()) {
            if (ddc::get<Dim1>(coord_eval) < ddc::discrete_space<bsplines_type1>().rmin()) {
                return m_lower_extrap_rule_1(coord_eval, spline_coef);
            }
            if (ddc::get<Dim1>(coord_eval) > ddc::discrete_space<bsplines_type1>().rmax()) {
                return m_upper_extrap_rule_1(coord_eval, spline_coef);
            }
        }
        if constexpr (!bsplines_type2::is_periodic()) {
            if (ddc::get<Dim2>(coord_eval) < ddc::discrete_space<bsplines_type2>().rmin()) {
                return m_lower_extrap_rule_2(coord_eval, spline_coef);
            }
            if (ddc::get<Dim2>(coord_eval) > ddc::discrete_space<bsplines_type2>().rmax()) {
                return m_upper_extrap_rule_2(coord_eval, spline_coef);
            }
        }
        if constexpr (!bsplines_type3::is_periodic()) {
            if (ddc::get<Dim3>(coord_eval) < ddc::discrete_space<bsplines_type3>().rmin()) {
                return m_lower_extrap_rule_3(coord_eval, spline_coef);
            }
            if (ddc::get<Dim3>(coord_eval) > ddc::discrete_space<bsplines_type3>().rmax()) {
                return m_upper_extrap_rule_3(coord_eval, spline_coef);
            }
        }
        return eval_no_bc<eval_type, eval_type, eval_type>(
                ddc::Coordinate<
                        continuous_dimension_type1,
                        continuous_dimension_type2,
                        continuous_dimension_type3>(
                        ddc::get<Dim1>(coord_eval),
                        ddc::get<Dim2>(coord_eval),
                        ddc::get<Dim3>(coord_eval)),
                spline_coef);
    }

    /**
     * @brief Evaluate the function or its derivative at the coordinate given.
     *
     * @param[in] coord_eval The coordinate where we want to evaluate.
     * @param[in] splne_coef The B-splines coefficients of the function we want to evaluate.
     * @tparam EvalType1 A flag indicating if we evaluate the function or its derivative in the first dimension. The type of this object is either `eval_type` or `eval_deriv_type`.
     * @tparam EvalType2 A flag indicating if we evaluate the function or its derivative in the second dimension. The type of this object is either `eval_type` or `eval_deriv_type`.
     */
    template <class EvalType1, class EvalType2, class EvalType3, class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval_no_bc(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert(
                std::is_same_v<EvalType1, eval_type> || std::is_same_v<EvalType1, eval_deriv_type>);
        static_assert(
                std::is_same_v<EvalType2, eval_type> || std::is_same_v<EvalType2, eval_deriv_type>);
        static_assert(
                std::is_same_v<EvalType3, eval_type> || std::is_same_v<EvalType3, eval_deriv_type>);
        ddc::DiscreteElement<bsplines_type1> jmin1;
        ddc::DiscreteElement<bsplines_type2> jmin2;
        ddc::DiscreteElement<bsplines_type3> jmin3;

        std::array<double, bsplines_type1::degree() + 1> vals1_ptr;
        Kokkos::mdspan<double, Kokkos::extents<std::size_t, bsplines_type1::degree() + 1>> const
                vals1(vals1_ptr.data());
        std::array<double, bsplines_type2::degree() + 1> vals2_ptr;
        Kokkos::mdspan<double, Kokkos::extents<std::size_t, bsplines_type2::degree() + 1>> const
                vals2(vals2_ptr.data());
        std::array<double, bsplines_type3::degree() + 1> vals3_ptr;
        Kokkos::mdspan<double, Kokkos::extents<std::size_t, bsplines_type3::degree() + 1>> const
                vals3(vals3_ptr.data());
        ddc::Coordinate<continuous_dimension_type1> const coord_eval_interest1(coord_eval);
        ddc::Coordinate<continuous_dimension_type2> const coord_eval_interest2(coord_eval);
        ddc::Coordinate<continuous_dimension_type3> const coord_eval_interest3(coord_eval);

        if constexpr (std::is_same_v<EvalType1, eval_type>) {
            jmin1 = ddc::discrete_space<bsplines_type1>().eval_basis(vals1, coord_eval_interest1);
        } else if constexpr (std::is_same_v<EvalType1, eval_deriv_type>) {
            jmin1 = ddc::discrete_space<bsplines_type1>().eval_deriv(vals1, coord_eval_interest1);
        }
        if constexpr (std::is_same_v<EvalType2, eval_type>) {
            jmin2 = ddc::discrete_space<bsplines_type2>().eval_basis(vals2, coord_eval_interest2);
        } else if constexpr (std::is_same_v<EvalType2, eval_deriv_type>) {
            jmin2 = ddc::discrete_space<bsplines_type2>().eval_deriv(vals2, coord_eval_interest2);
        }
        if constexpr (std::is_same_v<EvalType3, eval_type>) {
            jmin3 = ddc::discrete_space<bsplines_type3>().eval_basis(vals3, coord_eval_interest3);
        } else if constexpr (std::is_same_v<EvalType3, eval_deriv_type>) {
            jmin3 = ddc::discrete_space<bsplines_type3>().eval_deriv(vals3, coord_eval_interest3);
        }

        double y = 0.0;
        for (std::size_t i = 0; i < bsplines_type1::degree() + 1; ++i) {
            for (std::size_t j = 0; j < bsplines_type2::degree() + 1; ++j) {
                for (std::size_t k = 0; k < bsplines_type3::degree() + 1; ++k) {
                    y += spline_coef(
                                 ddc::DiscreteElement<
                                         bsplines_type1,
                                         bsplines_type2,
                                         bsplines_type3>(jmin1 + i, jmin2 + j, jmin3 + k))
                         * vals1[i] * vals2[j] * vals3[k];
                }
            }
        }
        return y;
    }
};

} // namespace ddc
