// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

#include "deriv.hpp"
#include "integrals.hpp"
#include "periodic_extrapolation_rule.hpp"

namespace ddc {

/**
 * @brief A class to evaluate, differentiate or integrate a 2D spline function.
 *
 * A class which contains an operator () which can be used to evaluate, differentiate or integrate a 2D spline function.
 *
 * @tparam ExecSpace The Kokkos execution space on which the spline evaluation is performed.
 * @tparam MemorySpace The Kokkos memory space on which the data (spline coefficients and evaluation) is stored.
 * @tparam BSplines1 The discrete dimension representing the B-splines along the first dimension of interest.
 * @tparam BSplines2 The discrete dimension representing the B-splines along the second dimension of interest.
 * @tparam EvaluationDDim1 The first discrete dimension on which evaluation points are defined.
 * @tparam EvaluationDDim2 The second discrete dimension on which evaluation points are defined.
 * @tparam LowerExtrapolationRule1 The lower extrapolation rule type along first dimension of interest.
 * @tparam UpperExtrapolationRule1 The upper extrapolation rule type along first dimension of interest.
 * @tparam LowerExtrapolationRule2 The lower extrapolation rule type along second dimension of interest.
 * @tparam UpperExtrapolationRule2 The upper extrapolation rule type along second dimension of interest.
 */
template <
        class ExecSpace,
        class MemorySpace,
        class BSplines1,
        class BSplines2,
        class EvaluationDDim1,
        class EvaluationDDim2,
        class LowerExtrapolationRule1,
        class UpperExtrapolationRule1,
        class LowerExtrapolationRule2,
        class UpperExtrapolationRule2>
class SplineEvaluator2D
{
public:
    /// @brief The type of the first evaluation continuous dimension used by this class.
    using continuous_dimension_type1 = typename BSplines1::continuous_dimension_type;

    /// @brief The type of the second evaluation continuous dimension used by this class.
    using continuous_dimension_type2 = typename BSplines2::continuous_dimension_type;

    /// @brief The type of the Kokkos execution space used by this class.
    using exec_space = ExecSpace;

    /// @brief The type of the Kokkos memory space used by this class.
    using memory_space = MemorySpace;

    /// @brief The type of the first discrete dimension of interest used by this class.
    using evaluation_discrete_dimension_type1 = EvaluationDDim1;

    /// @brief The type of the second discrete dimension of interest used by this class.
    using evaluation_discrete_dimension_type2 = EvaluationDDim2;

    /// @brief The discrete dimension representing the B-splines along first dimension.
    using bsplines_type1 = BSplines1;

    /// @brief The discrete dimension representing the B-splines along second dimension.
    using bsplines_type2 = BSplines2;

    /// @brief The type of the domain for the 1D evaluation mesh along first dimension used by this class.
    using evaluation_domain_type1 = ddc::DiscreteDomain<evaluation_discrete_dimension_type1>;

    /// @brief The type of the domain for the 1D evaluation mesh along second dimension used by this class.
    using evaluation_domain_type2 = ddc::DiscreteDomain<evaluation_discrete_dimension_type2>;

    /// @brief The type of the domain for the 2D evaluation mesh used by this class.
    using evaluation_domain_type = ddc::DiscreteDomain<
            evaluation_discrete_dimension_type1,
            evaluation_discrete_dimension_type2>;

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

    /// @brief The type of the 2D spline domain corresponding to the dimensions of interest.
    using spline_domain_type = ddc::DiscreteDomain<bsplines_type1, bsplines_type2>;

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
            evaluation_discrete_dimension_type2>;

    /**
     * @brief The type of the whole spline domain (cartesian product of 2D spline domain
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
                            evaluation_discrete_dimension_type2>,
                    ddc::detail::TypeSeq<bsplines_type1, bsplines_type2>>>;

    /// @brief The type of the extrapolation rule at the lower boundary along the first dimension.
    using lower_extrapolation_rule_1_type = LowerExtrapolationRule1;

    /// @brief The type of the extrapolation rule at the upper boundary along the first dimension.
    using upper_extrapolation_rule_1_type = UpperExtrapolationRule1;

    /// @brief The type of the extrapolation rule at the lower boundary along the second dimension.
    using lower_extrapolation_rule_2_type = LowerExtrapolationRule2;

    /// @brief The type of the extrapolation rule at the upper boundary along the second dimension.
    using upper_extrapolation_rule_2_type = UpperExtrapolationRule2;

private:
    LowerExtrapolationRule1 m_lower_extrap_rule_1;

    UpperExtrapolationRule1 m_upper_extrap_rule_1;

    LowerExtrapolationRule2 m_lower_extrap_rule_2;

    UpperExtrapolationRule2 m_upper_extrap_rule_2;

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
                               == bsplines_type2::is_periodic(),
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

    /**
     * @brief Build a SplineEvaluator2D acting on batched_spline_domain.
     *
     * @param lower_extrap_rule1 The extrapolation rule at the lower boundary along the first dimension.
     * @param upper_extrap_rule1 The extrapolation rule at the upper boundary along the first dimension.
     * @param lower_extrap_rule2 The extrapolation rule at the lower boundary along the second dimension.
     * @param upper_extrap_rule2 The extrapolation rule at the upper boundary along the second dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    explicit SplineEvaluator2D(
            LowerExtrapolationRule1 const& lower_extrap_rule1,
            UpperExtrapolationRule1 const& upper_extrap_rule1,
            LowerExtrapolationRule2 const& lower_extrap_rule2,
            UpperExtrapolationRule2 const& upper_extrap_rule2)
        : m_lower_extrap_rule_1(lower_extrap_rule1)
        , m_upper_extrap_rule_1(upper_extrap_rule1)
        , m_lower_extrap_rule_2(lower_extrap_rule2)
        , m_upper_extrap_rule_2(upper_extrap_rule2)
    {
    }

    /**
     * @brief Copy-constructs.
     *
     * @param x A reference to another SplineEvaluator.
     */
    SplineEvaluator2D(SplineEvaluator2D const& x) = default;

    /**
     * @brief Move-constructs.
     *
     * @param x An rvalue to another SplineEvaluator.
     */
    SplineEvaluator2D(SplineEvaluator2D&& x) = default;

    /// @brief Destructs.
    ~SplineEvaluator2D() = default;

    /**
     * @brief Copy-assigns.
     *
     * @param x A reference to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluator2D& operator=(SplineEvaluator2D const& x) = default;

    /**
     * @brief Move-assigns.
     *
     * @param x An rvalue to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluator2D& operator=(SplineEvaluator2D&& x) = default;

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
     * @brief Evaluate 2D spline function (described by its spline coefficients) at a given coordinate.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * Remark: calling SplineBuilder2D then SplineEvaluator2D corresponds to a 2D spline interpolation.
     *
     * @param coord_eval The coordinate where the spline is evaluated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
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
     * @brief Evaluate 2D spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD evaluation. This is a batched 2D evaluation. This means that for each slice of coordinates
     * identified by a batch_domain_type::discrete_element_type, the evaluation is performed with the 2D set of
     * spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilder2D then SplineEvaluator2D corresponds to a 2D spline interpolation.
     *
     * @param[out] spline_eval The values of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is evaluated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
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
        ddc::parallel_for_each(
                "ddc_splines_evaluate_2d",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_2D = spline_eval[j];
                    auto const coords_eval_2D = coords_eval[j];
                    auto const spline_coef_2D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            spline_eval_2D(i1, i2) = eval(coords_eval_2D(i1, i2), spline_coef_2D);
                        }
                    }
                });
    }

    /**
     * @brief Evaluate 2D spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional evaluation. This is a batched 2D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilder2D then SplineEvaluator2D corresponds to a 2D spline interpolation.
     *
     * @param[out] spline_eval The values of the 2D spline function at their coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
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
        ddc::parallel_for_each(
                "ddc_splines_evaluate_2d",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_2D = spline_eval[j];
                    auto const spline_coef_2D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            ddc::Coordinate<continuous_dimension_type1, continuous_dimension_type2>
                                    coord_eval_2D(ddc::coordinate(i1), ddc::coordinate(i2));
                            spline_eval_2D(i1, i2) = eval(coord_eval_2D(i1, i2), spline_coef_2D);
                        }
                    }
                });
    }

#if defined(DDC_BUILD_DEPRECATED_CODE)
    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) at a given coordinate along first dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder2D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<Dim1>>(1), ...) instead)")]] KOKKOS_FUNCTION double
    deriv_dim_1(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return deriv(
                ddc::DiscreteElement<Deriv<continuous_dimension_type1>>(1),
                coord_eval,
                spline_coef);
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) at a given coordinate along second dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder2D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<Dim2>>(1), ...) instead)")]] KOKKOS_FUNCTION double
    deriv_dim_2(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return deriv(
                ddc::DiscreteElement<Deriv<continuous_dimension_type2>>(1),
                coord_eval,
                spline_coef);
    }

    /**
     * @brief Cross-differentiate 2D spline function (described by its spline coefficients) at a given coordinate.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder2D.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<Dim1>, Deriv<Dim2>>(1, 1), ...) "
            "instead)")]] KOKKOS_FUNCTION double
    deriv_1_and_2(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return deriv(
                ddc::DiscreteElement<
                        Deriv<continuous_dimension_type1>,
                        Deriv<continuous_dimension_type2>>(1, 1),
                coord_eval,
                spline_coef);
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) at a given coordinate along a specified dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder2D.
     *
     * @tparam InterestDim Dimension along which differentiation is performed.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class InterestDim, class Layout, class... CoordsDims>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<InterestDim>>(1), ...) "
            "instead)")]] KOKKOS_FUNCTION double
    deriv(ddc::Coordinate<CoordsDims...> const& coord_eval,
          ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const spline_coef)
            const
    {
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
        static_assert(
                std::is_same_v<InterestDim, continuous_dimension_type1>
                || std::is_same_v<InterestDim, continuous_dimension_type2>);
        if constexpr (std::is_same_v<
                              InterestDim,
                              typename evaluation_discrete_dimension_type1::
                                      continuous_dimension_type>) {
            return deriv_dim_1(coord_eval, spline_coef);
        } else if constexpr (std::is_same_v<
                                     InterestDim,
                                     typename evaluation_discrete_dimension_type2::
                                             continuous_dimension_type>) {
            return deriv_dim_2(coord_eval, spline_coef);
        }
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    }

    /**
     * @brief Double-differentiate 2D spline function (described by its spline coefficients) at a given coordinate along specified dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder2D.
     *
     * Note: double-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     *
     * @param coord_eval The coordinate where the spline is double-differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class InterestDim1, class InterestDim2, class Layout, class... CoordsDims>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<InterestDim1, InterestDim2>>(1, 1), ...) "
            "instead)")]] KOKKOS_FUNCTION double
    deriv2(ddc::Coordinate<CoordsDims...> const& coord_eval,
           ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const spline_coef)
            const
    {
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
        static_assert(
                (std::is_same_v<
                         InterestDim1,
                         typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<
                            InterestDim2,
                            typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>));
        return deriv_1_and_2(coord_eval, spline_coef);
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) on a mesh along first dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD evaluation. This is a batched 2D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    [[deprecated("Use deriv(DiscreteElement<Deriv<Dim1>>(1), ...) instead)")]] void deriv_dim_1(
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
        return deriv(
                ddc::DiscreteElement<Deriv<continuous_dimension_type1>>(1),
                spline_eval,
                coords_eval,
                spline_coef);
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) on a mesh along first dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional evaluation. This is a batched 2D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    [[deprecated("Use deriv(DiscreteElement<Deriv<Dim1>>(1), ...) instead)")]] void deriv_dim_1(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        return deriv(
                ddc::DiscreteElement<Deriv<continuous_dimension_type1>>(1),
                spline_eval,
                spline_coef);
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) on a mesh along second dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD differentiation. This is a batched 2D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    [[deprecated("Use deriv(DiscreteElement<Deriv<Dim2>>(1), ...) instead)")]] void deriv_dim_2(
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
        return deriv(
                ddc::DiscreteElement<Deriv<continuous_dimension_type2>>(1),
                spline_eval,
                coords_eval,
                spline_coef);
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) on a mesh along second dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional evaluation. This is a batched 2D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    [[deprecated("Use deriv(DiscreteElement<Deriv<Dim2>>(1), ...) instead)")]] void deriv_dim_2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        return deriv(
                ddc::DiscreteElement<Deriv<continuous_dimension_type2>>(1),
                spline_eval,
                spline_coef);
    }

    /**
     * @brief Cross-differentiate 2D spline function (described by its spline coefficients) on a mesh along dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD cross-differentiation. This is a batched 2D cross-differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the cross-differentiation is performed with the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    [[deprecated("Use deriv(DiscreteElement<Deriv<Dim1>, Deriv<Dim2>>(1, 1), ...) instead)")]] void
    deriv_1_and_2(
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
        return deriv(
                ddc::DiscreteElement<
                        Deriv<continuous_dimension_type1>,
                        Deriv<continuous_dimension_type2>>(1, 1),
                spline_eval,
                coords_eval,
                spline_coef);
    }

    /**
     * @brief Cross-differentiate 2D spline function (described by its spline coefficients) on a mesh along dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional evaluation. This is a batched 2D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The cross-derivatives of the 2D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedInterpolationDDom>
    [[deprecated("Use deriv(DiscreteElement<Deriv<Dim1>, Deriv<Dim2>>(1, 1), ...) instead)")]] void
    deriv_1_and_2(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        return deriv(
                ddc::DiscreteElement<
                        Deriv<continuous_dimension_type1>,
                        Deriv<continuous_dimension_type2>>(1, 1),
                spline_eval,
                spline_coef);
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along a specified dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD evaluation. This is a batched 2D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam InterestDim Dimension along which differentiation is performed.
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class InterestDim,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    [[deprecated("Use deriv(DiscreteElement<Deriv<InterestDim>>(1), ...) instead)")]] void deriv(
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
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
        static_assert(
                std::is_same_v<
                        InterestDim,
                        typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                || std::is_same_v<InterestDim, continuous_dimension_type2>);
        if constexpr (std::is_same_v<
                              InterestDim,
                              typename evaluation_discrete_dimension_type1::
                                      continuous_dimension_type>) {
            return deriv_dim_1(spline_eval, coords_eval, spline_coef);
        } else if constexpr (std::is_same_v<
                                     InterestDim,
                                     typename evaluation_discrete_dimension_type2::
                                             continuous_dimension_type>) {
            return deriv_dim_2(spline_eval, coords_eval, spline_coef);
        }
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along a specified dimension of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional evaluation. This is a batched 2D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam InterestDim Dimension along which differentiation is performed.
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <class InterestDim, class Layout1, class Layout2, class BatchedInterpolationDDom>
    [[deprecated("Use deriv(DiscreteElement<Deriv<InterestDim>>(1), ...) instead)")]] void deriv(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
        static_assert(
                std::is_same_v<
                        InterestDim,
                        typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                || std::is_same_v<InterestDim, continuous_dimension_type2>);
        if constexpr (std::is_same_v<
                              InterestDim,
                              typename evaluation_discrete_dimension_type1::
                                      continuous_dimension_type>) {
            return deriv_dim_1(spline_eval, spline_coef);
        } else if constexpr (std::is_same_v<
                                     InterestDim,
                                     typename evaluation_discrete_dimension_type2::
                                             continuous_dimension_type>) {
            return deriv_dim_2(spline_eval, spline_coef);
        }
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    }

    /**
     * @brief Double-differentiate 2D spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD evaluation. This is a batched 2D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Note: double-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     *
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<InterestDim1>, Deriv<InterestDim2>>(1, 1), ...) "
            "instead)")]] void
    deriv2(ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
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
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
        static_assert(
                (std::is_same_v<
                         InterestDim1,
                         typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<
                            InterestDim2,
                            typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>));
        return deriv_1_and_2(spline_eval, coords_eval, spline_coef);
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    }

    /**
     * @brief Double-differentiate 2D spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional evaluation. This is a batched 2D evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Note: double-differentiation other than cross-differentiation is not supported atm. See #440
     *
     * @tparam InterestDim1 First dimension along which differentiation is performed.
     * @tparam InterestDim2 Second dimension along which differentiation is performed.
     *
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class InterestDim1,
            class InterestDim2,
            class Layout1,
            class Layout2,
            class BatchedInterpolationDDom>
    [[deprecated(
            "Use deriv(DiscreteElement<Deriv<InterestDim1>, Deriv<InterestDim2>>(1, 1), ...) "
            "instead)")]] void
    deriv2(ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                   spline_eval,
           ddc::ChunkSpan<
                   double const,
                   batched_spline_domain_type<BatchedInterpolationDDom>,
                   Layout2,
                   memory_space> const spline_coef) const
    {
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_PUSH()
        static_assert(
                (std::is_same_v<
                         InterestDim1,
                         typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                 && std::is_same_v<InterestDim2, continuous_dimension_type2>)
                || (std::is_same_v<
                            InterestDim2,
                            typename evaluation_discrete_dimension_type1::continuous_dimension_type>
                    && std::is_same_v<InterestDim1, continuous_dimension_type2>));
        return deriv_1_and_2(spline_eval, spline_coef);
        KOKKOS_IMPL_DISABLE_DEPRECATED_WARNINGS_POP()
    }
#endif

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) at a given coordinate along the dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder2D.
     *
     * @param deriv_order A DiscreteElement containing the orders of derivation for each of the dimensions of interest.
     * If one of the dimensions is not present, its corresponding order of derivation is considered to be 0. The highest
     * valid order is 2.
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the 2D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class DElem, class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv(
            DElem const& deriv_order,
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc(deriv_order, coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) on a mesh along the dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD differentiation. This is a batched 2D differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[in] deriv_order A DiscreteElement containing the orders of derivation for each of the dimensions of interest.
     * If one of the dimensions is not present, its corresponding order of derivation is considered to be 0. The highest
     * valid order is 2.
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 2D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <
            class DElem,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv(
            DElem const& deriv_order,
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
        static_assert(is_discrete_element_v<DElem>);

        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_2d",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_2D = spline_eval[j];
                    auto const coords_eval_2D = coords_eval[j];
                    auto const spline_coef_2D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            spline_eval_2D(i1, i2) = eval_no_bc(
                                    deriv_order,
                                    coords_eval_2D(i1, i2),
                                    spline_coef_2D);
                        }
                    }
                });
    }

    /**
     * @brief Differentiate 2D spline function (described by its spline coefficients) on a mesh along the dimensions of interest.
     *
     * The spline coefficients represent a 2D spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a multidimensional differentiation. This is a batched 2D differentiation.
     * This means that for each slice of spline_eval the differentiation is performed with
     * the 2D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[in] deriv_order A DiscreteElement containing the orders of derivation for each of the dimensions of interest.
     * If one of the dimensions is not present, its corresponding order of derivation is considered to be 0. The highest
     * valid order is 2.
     * @param[out] spline_eval The derivatives of the 2D spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <class DElem, class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv(
            DElem const& deriv_order,
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        static_assert(is_discrete_element_v<DElem>);

        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        evaluation_domain_type1 const evaluation_domain1(spline_eval.domain());
        evaluation_domain_type2 const evaluation_domain2(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_2d",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_2D = spline_eval[j];
                    auto const spline_coef_2D = spline_coef[j];
                    for (auto const i1 : evaluation_domain1) {
                        for (auto const i2 : evaluation_domain2) {
                            ddc::Coordinate<continuous_dimension_type1, continuous_dimension_type2>
                                    coord_eval_2D(ddc::coordinate(i1), ddc::coordinate(i2));
                            spline_eval_2D(i1, i2)
                                    = eval_no_bc(deriv_order, coord_eval_2D, spline_coef_2D);
                        }
                    }
                });
    }

    /** @brief Perform batched 2D integrations of a spline function (described by its spline coefficients) along the dimensions of interest and store results on a subdomain of batch_domain.
     *
     * The spline coefficients represent a 2D spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilder2D.
     *
     * This is not a nD integration. This is a batched 2D integration.
     * This means that for each element of integrals, the integration is performed with the 2D set of
     * spline coefficients identified by the same DiscreteElement.
     *
     * @param[out] integrals The integrals of the 2D spline function on the subdomain of batch_domain. For practical reasons those are
     * stored in a ChunkSpan defined on a batch_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant.
     * @param[in] spline_coef A ChunkSpan storing the 2D spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedDDom, class BatchedSplineDDom>
    void integrate(
            ddc::ChunkSpan<double, BatchedDDom, Layout1, memory_space> const integrals,
            ddc::ChunkSpan<double const, BatchedSplineDDom, Layout2, memory_space> const
                    spline_coef) const
    {
        static_assert(
                ddc::type_seq_contains_v<
                        ddc::detail::TypeSeq<bsplines_type1, bsplines_type2>,
                        to_type_seq_t<BatchedSplineDDom>>,
                "The spline coefficients domain must contain the bsplines dimensions");
        using batch_domain_type
                = ddc::remove_dims_of_t<BatchedSplineDDom, bsplines_type1, bsplines_type2>;
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
                            integrals(j) += spline_coef(i1, i2, j) * values1(i1) * values2(i2);
                        }
                    }
                });
    }

private:
    /**
     * @brief Evaluate the function on B-splines at the coordinate given.
     *
     * This function firstly deals with the boundary conditions and calls the SplineEvaluator2D::eval_no_bc function
     * to evaluate.
     *
     * @param[in] coord_eval The 2D coordinate where we want to evaluate.
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
        return eval_no_bc(
                ddc::DiscreteElement<>(),
                ddc::Coordinate<continuous_dimension_type1, continuous_dimension_type2>(
                        ddc::get<Dim1>(coord_eval),
                        ddc::get<Dim2>(coord_eval)),
                spline_coef);
    }

    /**
     * @brief Evaluate the function or its derivative at the coordinate given.
     *
     * @param[in] deriv_order A DiscreteElement containing the orders of derivation for each of the dimensions of interest.
     * If one of the dimensions is not present, its corresponding order of derivation is considered to be 0.
     * @param[in] coord_eval The coordinate where we want to evaluate.
     * @param[in] splne_coef The B-splines coefficients of the function we want to evaluate.
     */
    template <class... DDims, class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval_no_bc(
            ddc::DiscreteElement<DDims...> const& deriv_order,
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        using deriv_dim1 = ddc::Deriv<continuous_dimension_type1>;
        using deriv_dim2 = ddc::Deriv<continuous_dimension_type2>;
        using deriv_dims = ddc::detail::TypeSeq<DDims...>;

        // Check that the tags are valid
        static_assert(
                (in_tags_v<DDims, ddc::detail::TypeSeq<deriv_dim1, deriv_dim2>> && ...),
                "The only valid dimensions for deriv_order are Deriv<Dim1> and Deriv<Dim2>");

        auto const order1 = select_or(deriv_order, DiscreteElement<deriv_dim1>(0)).uid();
        auto const order2 = select_or(deriv_order, DiscreteElement<deriv_dim2>(0)).uid();
        KOKKOS_ASSERT(order1 >= 0 && order1 < 3);
        KOKKOS_ASSERT(order2 >= 0 && order2 < 3);

        ddc::DiscreteElement<bsplines_type1> jmin1;
        ddc::DiscreteElement<bsplines_type2> jmin2;

        std::array<double, bsplines_type1::degree() + 1> vals1_ptr;
        Kokkos::mdspan<double, Kokkos::extents<std::size_t, bsplines_type1::degree() + 1>> const
                vals1(vals1_ptr.data());
        std::array<double, bsplines_type2::degree() + 1> vals2_ptr;
        Kokkos::mdspan<double, Kokkos::extents<std::size_t, bsplines_type2::degree() + 1>> const
                vals2(vals2_ptr.data());
        ddc::Coordinate<continuous_dimension_type1> const coord_eval_interest1(coord_eval);
        ddc::Coordinate<continuous_dimension_type2> const coord_eval_interest2(coord_eval);

        if constexpr (!in_tags_v<deriv_dim1, deriv_dims>) {
            jmin1 = ddc::discrete_space<bsplines_type1>().eval_basis(vals1, coord_eval_interest1);
        } else {
            if (order1 == 0) {
                jmin1 = ddc::discrete_space<bsplines_type1>()
                                .eval_basis(vals1, coord_eval_interest1);
            } else if (order1 == 1) {
                jmin1 = ddc::discrete_space<bsplines_type1>()
                                .eval_deriv(vals1, coord_eval_interest1);
            } else {
                std::array<double, 3 * (bsplines_type1::degree() + 1)> derivs1_ptr;
                Kokkos::mdspan<
                        double,
                        Kokkos::extents<std::size_t, bsplines_type1::degree() + 1, 3>>
                        derivs1(derivs1_ptr.data());

                jmin1 = ddc::discrete_space<bsplines_type1>()
                                .eval_basis_and_n_derivs(derivs1, coord_eval_interest1, order1);

                for (std::size_t i = 0; i < bsplines_type1::degree() + 1; ++i) {
                    vals1[i] = derivs1(i, order1);
                }
            }
        }

        if constexpr (!in_tags_v<deriv_dim2, deriv_dims>) {
            jmin2 = ddc::discrete_space<bsplines_type2>().eval_basis(vals2, coord_eval_interest2);
        } else {
            if (order2 == 0) {
                jmin2 = ddc::discrete_space<bsplines_type2>()
                                .eval_basis(vals2, coord_eval_interest2);
            } else if (order2 == 1) {
                jmin2 = ddc::discrete_space<bsplines_type2>()
                                .eval_deriv(vals2, coord_eval_interest2);
            } else {
                std::array<double, 3 * (bsplines_type2::degree() + 1)> derivs2_ptr;
                Kokkos::mdspan<
                        double,
                        Kokkos::extents<std::size_t, bsplines_type2::degree() + 1, 3>>
                        derivs2(derivs2_ptr.data());

                jmin2 = ddc::discrete_space<bsplines_type2>()
                                .eval_basis_and_n_derivs(derivs2, coord_eval_interest2, order2);

                for (std::size_t i = 0; i < bsplines_type2::degree() + 1; ++i) {
                    vals2[i] = derivs2(i, order2);
                }
            }
        }

        double y = 0.0;
        for (std::size_t i = 0; i < bsplines_type1::degree() + 1; ++i) {
            for (std::size_t j = 0; j < bsplines_type2::degree() + 1; ++j) {
                y += spline_coef(
                             ddc::DiscreteElement<
                                     bsplines_type1,
                                     bsplines_type2>(jmin1 + i, jmin2 + j))
                     * vals1[i] * vals2[j];
            }
        }
        return y;
    }
};

} // namespace ddc
