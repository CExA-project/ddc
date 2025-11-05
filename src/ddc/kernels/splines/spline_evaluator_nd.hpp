// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

#include "ddc/coordinate.hpp"
#include "ddc/detail/type_seq.hpp"
#include "ddc/discrete_domain.hpp"

#include "integrals.hpp"
#include "periodic_extrapolation_rule.hpp"

namespace ddc {

namespace detail {
template <typename... Args>
class SplineEvaluatorND;

template <
        typename ExecSpace,
        typename MemorySpace,
        typename... BSplines,
        typename... EvaluationDDim,
        typename... LowerExtrapolationRule,
        typename... UpperExtrapolationRule,
        std::size_t... Idx>
class SplineEvaluatorND<
        ExecSpace,
        MemorySpace,
        ddc::detail::TypeSeq<BSplines...>,
        ddc::detail::TypeSeq<EvaluationDDim...>,
        ddc::detail::TypeSeq<LowerExtrapolationRule...>,
        ddc::detail::TypeSeq<UpperExtrapolationRule...>,
        std::index_sequence<Idx...>>
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

    using bsplines_ts = ddc::detail::TypeSeq<BSplines...>;
    using evaluation_ddim_ts = ddc::detail::TypeSeq<EvaluationDDim...>;
    using lower_extrap_rule_ts = ddc::detail::TypeSeq<LowerExtrapolationRule...>;
    using upper_extrap_rule_ts = ddc::detail::TypeSeq<UpperExtrapolationRule...>;

    static constexpr std::size_t dimension = sizeof...(Idx);

    template <std::size_t I, std::size_t... Others>
    static constexpr bool index_seq_contains()
    {
        return ((I == Others) || ...);
    }

public:
    /// @brief The type of the Ith evaluation continuous dimension used by this class.
    /// @tparam I the requested dimension
    template <std::size_t I>
    using continuous_dimension_type =
            typename ddc::type_seq_element_t<I, bsplines_ts>::continuous_dimension_type;

    /// @brief The type of the Kokkos execution space used by this class.
    using exec_space = ExecSpace;

    /// @brief The type of the Kokkos memory space used by this class.
    using memory_space = MemorySpace;

    /// @brief The type of the Ith discrete dimension of interest used by this class.
    template <std::size_t I>
    using evaluation_discrete_dimension_type = ddc::type_seq_element_t<I, evaluation_ddim_ts>;

    /// @brief The discrete dimension representing the B-splines along Ith dimension.
    template <std::size_t I>
    using bsplines_type = ddc::type_seq_element_t<I, bsplines_ts>;

    /**
     * @brief The type of the domain for the 1D, 2D, ... or ND evaluation mesh along specified dimensions used by this class.
     *
     * @tparam Dims the required dimensions, 0 indexed
     */
    template <std::size_t... Dims>
    using evaluation_domain_type = ddc::DiscreteDomain<evaluation_discrete_dimension_type<Dims>...>;

    /**
     * @brief The type of the whole domain representing evaluation points.
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batched_evaluation_domain_type = BatchedInterpolationDDom;

    /**
     * @brief The type of the 1D, 2D, ... or ND spline domain corresponding to the specified dimensions of interest.
     *
     * @tparam Dims the required dimensions, 0 indexed
     */
    template <std::size_t... Dims>
    using spline_domain_type = ddc::DiscreteDomain<bsplines_type<Dims>...>;

    /**
     * @brief The type of the batch domain (obtained by removing the dimensions of interest
     * from the whole domain).
     *
     * @tparam The batched discrete domain on which the interpolation points are defined.
     */
    template <
            class BatchedInterpolationDDom,
            class = std::enable_if_t<ddc::is_discrete_domain_v<BatchedInterpolationDDom>>>
    using batch_domain_type =
            typename ddc::remove_dims_of_t<BatchedInterpolationDDom, EvaluationDDim...>;

    /**
     * @brief The type of the whole spline domain (cartesian product of N:wspline domain
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
                    evaluation_ddim_ts,
                    bsplines_ts>>;

    /// @brief The type of the extrapolation rule at the lower boundary along the Ith dimension.
    template <std::size_t I>
    using lower_extrapolation_rule_type = ddc::type_seq_element_t<I, lower_extrap_rule_ts>;

    /// @brief The type of the extrapolation rule at the upper boundary along the Ith dimension.
    template <std::size_t I>
    using upper_extrapolation_rule_type = ddc::type_seq_element_t<I, upper_extrap_rule_ts>;

private:
    std::tuple<LowerExtrapolationRule...> m_lower_extrap_rules;
    std::tuple<UpperExtrapolationRule...> m_upper_extrap_rules;

public:
    static_assert(
            sizeof...(BSplines) == sizeof...(Idx),
            "Number of BSpline dims should be equal to the dimension");
    static_assert(
            sizeof...(EvaluationDDim) == sizeof...(Idx),
            "Number of evaluation dims should be equal to the dimensions");
    static_assert(
            sizeof...(LowerExtrapolationRule) == sizeof...(Idx),
            "Number of lower extrapolation rules should be equal to the dimension");
    static_assert(
            sizeof...(UpperExtrapolationRule) == sizeof...(Idx),
            "Number of upper extrapolation rules should be equal to the dimension");

    static_assert(
            ((std::is_same_v<
                      LowerExtrapolationRule,
                      typename ddc::PeriodicExtrapolationRule<continuous_dimension_type<Idx>>>
                      == bsplines_type<Idx>::is_periodic()
              && std::is_same_v<
                         UpperExtrapolationRule,
                         typename ddc::PeriodicExtrapolationRule<continuous_dimension_type<Idx>>>
                         == bsplines_type<Idx>::is_periodic())
             && ...),
            "PeriodicExtrapolationRule has to be used if and only if dimension is periodic");
    static_assert(
            (std::is_invocable_r_v<
                     double,
                     LowerExtrapolationRule,
                     ddc::Coordinate<continuous_dimension_type<Idx>>,
                     ddc::ChunkSpan<
                             double const,
                             spline_domain_type<Idx>,
                             Kokkos::layout_right,
                             memory_space>>
             && ...),
            "LowerExtrapolationRule::operator() has to be callable "
            "with usual arguments.");
    static_assert(
            (std::is_invocable_r_v<
                     double,
                     UpperExtrapolationRule,
                     ddc::Coordinate<continuous_dimension_type<Idx>>,
                     ddc::ChunkSpan<
                             double const,
                             spline_domain_type<Idx>,
                             Kokkos::layout_right,
                             memory_space>>
             && ...),
            "UpperExtrapolationRule::operator() has to be callable "
            "with usual arguments.");

    /**
     * @brief Build a SplineEvaluatorND acting on batched_spline_domain.
     *
     * @param lower_extrap_rules... The extrapolation rules at the lower boundary along the each dimension.
     * @param upper_extrap_rules... The extrapolation rules at the upper boundary along the each dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    explicit SplineEvaluatorND(
            LowerExtrapolationRule const&... lower_extrap_rules,
            UpperExtrapolationRule const&... upper_extrap_rules)
        : m_lower_extrap_rules(lower_extrap_rules...)
        , m_upper_extrap_rules(upper_extrap_rules...)
    {
    }

    /**
     * @brief Copy-constructs.
     *
     * @param x A reference to another SplineEvaluator.
     */
    SplineEvaluatorND(SplineEvaluatorND const& x) = default;

    /**
     * @brief Move-constructs.
     *
     * @param x An rvalue to another SplineEvaluator.
     */
    SplineEvaluatorND(SplineEvaluatorND&& x) = default;

    /// @brief Destructs.
    ~SplineEvaluatorND() = default;

    /**
     * @brief Copy-assigns.
     *
     * @param x A reference to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluatorND& operator=(SplineEvaluatorND const& x) = default;

    /**
     * @brief Move-assigns.
     *
     * @param x An rvalue to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluatorND& operator=(SplineEvaluatorND&& x) = default;

    /**
     * @brief Get the lower extrapolation rule along the Ith dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The lower extrapolation rule along the Ith dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    template <std::size_t I>
    auto lower_extrapolation_rule() const
    {
        return std::get<I>(m_lower_extrap_rules);
    }

    /**
     * @brief Get the upper extrapolation rule along the Ith dimension.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The upper extrapolation rule along the Ith dimension.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    template <std::size_t I>
    auto upper_extrapolation_rule() const
    {
        return std::get<I>(m_upper_extrap_rules);
    }

    /**
     * @brief Evaluate ND spline function (described by its spline coefficients) at a given coordinate.
     *
     * The spline coefficients represent a ND spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * Remark: calling SplineBuilderND then SplineEvaluatorND corresponds to a ND spline interpolation.
     *
     * @param coord_eval The coordinate where the spline is evaluated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the ND spline coefficients.
     *
     * @return The value of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double operator()(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type<Idx...>, Layout, memory_space> const
                    spline_coef) const
    {
        return eval(coord_eval, spline_coef);
    }

    /**
     * @brief Evaluate ND spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a ND spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a nD evaluation. This is a batched ND evaluation. This means that for each slice of coordinates
     * identified by a batch_domain_type::discrete_element_type, the evaluation is performed with the ND set of
     * spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilderND then SplineEvaluatorND corresponds to a ND spline interpolation.
     *
     * @param[out] spline_eval The values of the ND spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is evaluated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of ND spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
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
        ddc::parallel_for_each(
                "ddc_splines_evaluate_Nd",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_ND = spline_eval[j];
                    auto const coords_eval_ND = coords_eval[j];
                    auto const spline_coef_ND = spline_coef[j];
                    ddc::for_each(
                            evaluation_domain_type<Idx...>(spline_eval.domain()),
                            [=](typename evaluation_domain_type<Idx...>::discrete_element_type const
                                        i) {
                                spline_eval_ND(i) = eval(coords_eval_ND(i), spline_coef_ND);
                            });
                });
    }

    /**
     * @brief Evaluate ND spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a ND spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a multidimensional evaluation. This is a batched ND evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the ND set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilderND then SplineEvaluatorND corresponds to a ND spline interpolation.
     *
     * @param[out] spline_eval The values of the ND spline function at their coordinates.
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
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
        ddc::parallel_for_each(
                "ddc_splines_evaluate_Nd",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_ND = spline_eval[j];
                    auto const spline_coef_ND = spline_coef[j];

                    ddc::for_each(
                            evaluation_domain_type<Idx...>(spline_eval.domain()),
                            [=](typename evaluation_domain_type<Idx...>::discrete_element_type const
                                        i) {
                                ddc::Coordinate<continuous_dimension_type<Idx>...> coord_eval_ND(
                                        ddc::coordinate(i));
                                spline_eval_ND(i) = eval(coord_eval_3D(i), spline_coef_ND);
                            });
                });
    }

    /**
     * @brief Differentiate ND spline function (described by its spline coefficients) at a given coordinate along Ith dimension of interest.
     *
     * The spline coefficients represent a ND spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilderND.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the ND spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <std::size_t... DerivDims, class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv_dim_I(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type<Idx...>, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert(sizeof...(DerivDims) > 0 && sizeof...(DerivDims) <= dimension);
        return eval_no_bc<std::conditional_t<
                index_seq_contains<Idx, DerivDims...>(),
                eval_deriv_type,
                eval_type>...>(coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate ND spline function (described by its spline coefficients) on a mesh along first dimension of interest.
     *
     * The spline coefficients represent a ND spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a nD evaluation. This is a batched ND differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the ND set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam DerivDims The dimensions which should be differentiated.
     *
     * @param[out] spline_eval The derivatives of the ND spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of ND spline coefficients retained to perform the evaluation).
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of ND spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
     */
    template <
            std::size_t... DerivDims,
            class Layout1,
            class Layout2,
            class Layout3,
            class BatchedInterpolationDDom,
            class... CoordsDims>
    void deriv_dim_I(
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
        // FIXME: Add static_assert for DerivDims
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(coords_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_Nd_dims",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_ND = spline_eval[j];
                    auto const coords_eval_ND = coords_eval[j];
                    auto const spline_coef_ND = spline_coef[j];
                    ddc::for_each(
                            evaluation_domain_type<Idx...>(spline_eval.domain()),
                            [=](typename ddc::DiscreteDomain<
                                    evaluation_domain_type<Idx>...>::discrete_element_type i) {
                                spline_eval_ND(i) = eval_no_bc<std::conditional_t<
                                        index_seq_contains<Idx, DerivDims...>(),
                                        eval_deriv_type,
                                        eval_type>...>(coords_eval_ND(i), spline_coef_ND);
                            });
                });
    }

    /**
     * @brief Differentiate ND spline function (described by its spline coefficients) on a mesh along first dimension of interest.
     *
     * The spline coefficients represent a ND spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a multidimensional evaluation. This is a batched ND evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the ND set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam DerivDims The dimensions which should be differentiated.
     *
     * @param[out] spline_eval The derivatives of the ND spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
     */
    template <
            std::size_t... DerivDims,
            class Layout1,
            class Layout2,
            class BatchedInterpolationDDom>
    void deriv_dim_I(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());
        ddc::parallel_for_each(
                "ddc_splines_differentiate_Nd_dims",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_ND = spline_eval[j];
                    auto const spline_coef_ND = spline_coef[j];
                    ddc::for_each(
                            evaluation_domain_type<Idx...>(spline_eval.domain()),
                            [=](typename ddc::DiscreteDomain<
                                    evaluation_domain_type<Idx>...>::discrete_element_type i) {
                                ddc::Coordinate<continuous_dimension_type<Idx>...> coord_eval_ND(
                                        ddc::coordinate(i));

                                spline_eval_ND(i) = eval_no_bc<std::conditional_t<
                                        index_seq_contains<Idx, DerivDims...>(),
                                        eval_deriv_type,
                                        eval_type>...>(coord_eval_ND, spline_coef_ND);
                            });
                });
    }

    /**
     * @brief Differentiate ND spline function (described by its spline coefficients) at a given coordinate along specified dimensions of interest.
     *
     * The spline coefficients represent a ND spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilderND.
     *
     * @tparam InterestDims Dimensions along which differentiation is performed.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the components along the dimensions of interest are used.
     * @param spline_coef A ChunkSpan storing the ND spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class... InterestDims, class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type<Idx...>, Layout, memory_space> const
                    spline_coef) const
    {
        using interest_dims_ts = ddc::detail::TypeSeq<InterestDims...>;

        static_assert(ddc::type_seq_contains_v<
                      interest_dims_ts,
                      ddc::detail::TypeSeq<continuous_dimension_type<Idx>...>>);

        return eval_no_bc<std::conditional_t<
                ddc::in_tags_v<continuous_dimension_type<Idx>, interest_dims_ts>,
                eval_deriv_type,
                eval_type>...>(coord_eval, spline_coef);
    }


    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a ND spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a nD evaluation. This is a batched ND differentiation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the differentiation is performed with the ND set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam InterestDims Dimensions along which differentiation is performed.
     * @param[out] spline_eval The derivatives of the ND spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of ND spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
     */
    template <
            class... InterestDims,
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
        using interest_dims_ts = ddc::detail::TypeSeq<InterestDims...>;

        static_assert(ddc::type_seq_contains_v<
                      interest_dims_ts,
                      ddc::detail::TypeSeq<continuous_dimension_type<Idx>...>>);

        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());

        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_Nd",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_ND = spline_eval[j];
                    auto const coords_eval_ND = coords_eval[j];
                    auto const spline_coef_ND = spline_coef[j];
                    ddc::for_each(
                            evaluation_domain_type<Idx...>(spline_eval.domain()),
                            [=](typename evaluation_domain_type<Idx...>::discrete_element_type const
                                        i) {
                                spline_eval_ND(i) = eval_no_bc<std::conditional_t<
                                        ddc::in_tags_v<
                                                continuous_dimension_type<Idx>,
                                                interest_dims_ts>,
                                        eval_deriv_type,
                                        eval_type>...>(coords_eval_ND(i), spline_coef_ND);
                            });
                });
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh along specified dimensions of interest.
     *
     * The spline coefficients represent a ND spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a multidimensional evaluation. This is a batched ND evaluation.
     * This means that for each slice of spline_eval the evaluation is performed with
     * the ND set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @tparam InterestDims Dimensions along which differentiation is performed.
     * @param[out] spline_eval The derivatives of the ND spline function at the desired coordinates.
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
     */
    template <class... InterestDims, class Layout1, class Layout2, class BatchedInterpolationDDom>
    void deriv(
            ddc::ChunkSpan<double, BatchedInterpolationDDom, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    double const,
                    batched_spline_domain_type<BatchedInterpolationDDom>,
                    Layout2,
                    memory_space> const spline_coef) const
    {
        using interest_dims_ts = ddc::detail::TypeSeq<InterestDims...>;

        static_assert(ddc::type_seq_contains_v<
                      interest_dims_ts,
                      ddc::detail::TypeSeq<continuous_dimension_type<Idx>...>>);

        batch_domain_type<BatchedInterpolationDDom> const batch_domain(spline_eval.domain());

        ddc::parallel_for_each(
                "ddc_splines_cross_differentiate_Nd",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(
                        typename batch_domain_type<
                                BatchedInterpolationDDom>::discrete_element_type const j) {
                    auto const spline_eval_ND = spline_eval[j];
                    auto const spline_coef_ND = spline_coef[j];
                    ddc::for_each(
                            evaluation_domain_type<Idx...>(spline_eval.domain()),
                            [=](typename evaluation_domain_type<Idx...>::discrete_element_type const
                                        i) {
                                ddc::Coordinate<continuous_dimension_type<Idx>...> coord_eval_ND(
                                        ddc::coordinate(i));
                                spline_eval_ND(i) = eval_no_bc<std::conditional_t<
                                        ddc::in_tags_v<
                                                continuous_dimension_type<Idx>,
                                                interest_dims_ts>,
                                        eval_deriv_type,
                                        eval_type>...>(coord_eval_ND, spline_coef_ND);
                            });
                });
    }

    /** @brief Perform batched ND integrations of a spline function (described by its spline coefficients) along the dimensions of interest and store results on a subdomain of batch_domain.
     *
     * The spline coefficients represent a ND spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilderND.
     *
     * This is not a nD integration. This is a batched ND integration.
     * This means that for each element of integrals, the integration is performed with the ND set of
     * spline coefficients identified by the same DiscreteElement.
     *
     * @param[out] integrals The integrals of the ND spline function on the subdomain of batch_domain. For practical reasons those are
     * stored in a ChunkSpan defined on a batch_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant.
     * @param[in] spline_coef A ChunkSpan storing the ND spline coefficients.
     */
    template <class Layout1, class Layout2, class BatchedDDom, class BatchedSplineDDom>
    void integrate(
            ddc::ChunkSpan<double, BatchedDDom, Layout1, memory_space> const integrals,
            ddc::ChunkSpan<double const, BatchedSplineDDom, Layout2, memory_space> const
                    spline_coef) const
    {
        static_assert(
                ddc::type_seq_contains_v<bsplines_ts, to_type_seq_t<BatchedSplineDDom>>,
                "The spline coefficients domain must contain the bsplines dimensions");
        using batch_domain_type = ddc::remove_dims_of_t<BatchedSplineDDom, bsplines_type<Idx>...>;
        static_assert(
                std::is_same_v<batch_domain_type, BatchedDDom>,
                "The integrals domain must only contain the batch dimensions");

        batch_domain_type batch_domain(integrals.domain());
        auto values_alloc = std::make_tuple(
                ddc::
                        Chunk(ddc::DiscreteDomain<bsplines_type<Idx>>(spline_coef.domain()),
                              ddc::KokkosAllocator<double, memory_space>())...);
        auto values = std::make_tuple(std::get<Idx>(values_alloc).span_view()...);
        (ddc::integrals(exec_space(), std::get<Idx>(values)), ...);

        ddc::parallel_for_each(
                "ddc_splines_integrate_bsplines",
                exec_space(),
                batch_domain,
                KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    integrals(j) = 0;
                    ddc::for_each(
                            spline_domain_type<Idx...>(),
                            [=](typename spline_domain_type<Idx...>::discrete_element_type const
                                        i) {
                                integrals(j) += spline_coef(i, j)
                                                * (std::get<Idx>(values)(
                                                           ddc::get<spline_domain_type<Idx>>(i))
                                                   * ...);
                            });
                    // for (typename spline_domain_type1::discrete_element_type const i1 :
                    //      values1.domain()) {
                    //     for (typename spline_domain_type2::discrete_element_type const i2 :
                    //          values2.domain()) {
                    //         for (typename spline_domain_type3::discrete_element_type const i3 :
                    //              values3.domain()) {
                    //             integrals(j) += spline_coef(i1, i2, i3, j) * values1(i1)
                    //                             * values2(i2) * values3(i3);
                    //         }
                    //     }
                    // }
                });
    }

private:
    template <std::size_t I, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION static void update_coord_eval(ddc::Coordinate<CoordsDims...>& coord_eval)
    {
        using Dim = continuous_dimension_type<I>;
        using bsplines_t = bsplines_type<I>;

        if constexpr (bsplines_t::is_periodic()) {
            if (ddc::get<Dim>(coord_eval) < ddc::discrete_space<bsplines_t>().rmin()
                || ddc::get<Dim>(coord_eval) > ddc::discrete_space<bsplines_t>().rmax()) {
                ddc::get<Dim>(coord_eval) -= Kokkos::floor(
                                                     (ddc::get<Dim>(coord_eval)
                                                      - ddc::discrete_space<bsplines_t>().rmin())
                                                     / ddc::discrete_space<bsplines_t>().length())
                                             * ddc::discrete_space<bsplines_t>().length();
            }
        }
    }

    template <std::size_t I, class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION bool check_needs_extrapolation(
            ddc::Coordinate<CoordsDims...> coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type<Idx...>, Layout, memory_space> const
                    spline_coef,
            double& res) const
    {
        if constexpr (!bsplines_type<I>::is_periodic()) {
            if (ddc::get<continuous_dimension_type<I>>(coord_eval)
                < ddc::discrete_space<bsplines_type<I>>().rmin()) {
                res = std::get<I>(m_lower_extrap_rules)(coord_eval, spline_coef);
                return true;
            }
            if (ddc::get<continuous_dimension_type<I>>(coord_eval)
                > ddc::discrete_space<bsplines_type<I>>().rmax()) {
                res = std::get<I>(m_upper_extrap_rules)(coord_eval, spline_coef);
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Evaluate the function on B-splines at the coordinate given.
     *
     * This function firstly deals with the boundary conditions and calls the SplineEvaluatorND::eval_no_bc function
     * to evaluate.
     *
     * @param[in] coord_eval The ND coordinate where we want to evaluate.
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
            ddc::ChunkSpan<double const, spline_domain_type<Idx...>, Layout, memory_space> const
                    spline_coef) const
    {
        (update_coord_eval<Idx>(coord_eval), ...);

        double res = 0.;
        bool needs_extrap = (... || check_needs_extrapolation<Idx>(coord_eval, spline_coef, res));

        if (needs_extrap) {
            return res;
        }

        return eval_no_bc<eval_type, eval_type, eval_type>(
                ddc::Coordinate<continuous_dimension_type<Idx>...>(
                        ddc::get<continuous_dimension_type<Idx>>(coord_eval)...),
                spline_coef);
    }

    template <typename EvalType, typename BSplinesType, typename CoordDim>
    KOKKOS_INLINE_FUNCTION static ddc::DiscreteElement<BSplinesType> get_jmin(
            Kokkos::mdspan<double, Kokkos::extents<std::size_t, BSplinesType::degree() + 1>> vals,
            ddc::Coordinate<CoordDim> const& coord_eval)
    {
        if constexpr (std::is_same_v<EvalType, eval_type>) {
            return ddc::discrete_space<BSplinesType>().eval_basis(vals, coord_eval);
        } else if constexpr (std::is_same_v<EvalType, eval_deriv_type>) {
            return ddc::discrete_space<BSplinesType>().eval_deriv(vals, coord_eval);
        } else {
            return ddc::DiscreteElement {};
        }
    }

    template <std::size_t N, class Functor, typename... Is>
    KOKKOS_INLINE_FUNCTION static void for_each(
            std::array<std::size_t, N> const& bounds,
            Functor const& f,
            Is... is)
    {
        static constexpr std::size_t I = sizeof...(Is);
        if constexpr (I == N) {
            f(std::make_tuple(is...));
        } else {
            for (std::size_t i = 0; i < bounds[I]; ++i) {
                for_each(bounds, f, is..., i);
            }
        }
    }

    /**
     * @brief Evaluate the function or its derivative at the coordinate given.
     *
     * @param[in] coord_eval The coordinate where we want to evaluate.
     * @param[in] splne_coef The B-splines coefficients of the function we want to evaluate.
     * @tparam EvalType1 A flag indicating if we evaluate the function or its derivative in the first dimension. The type of this object is either `eval_type` or `eval_deriv_type`.
     * @tparam EvalType2 A flag indicating if we evaluate the function or its derivative in the second dimension. The type of this object is either `eval_type` or `eval_deriv_type`.
     */
    template <class... EvalTypes, class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval_no_bc(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type<Idx...>, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert((
                (std::is_same_v<EvalTypes, eval_type> || std::is_same_v<EvalTypes, eval_deriv_type>)
                && ...));

        std::tuple vals_ptr
                = std::make_tuple(std::array<double, bsplines_type<Idx>::degree() + 1> {}...);
        std::tuple const vals = std::make_tuple(
                Kokkos::mdspan<
                        double,
                        Kokkos::extents<std::size_t, bsplines_type<Idx>::degree() + 1>>(
                        std::get<Idx>(vals_ptr).data())...);

        std::tuple const jmin = std::make_tuple(
                get_jmin<EvalTypes, bsplines_type<Idx>>(
                        std::get<Idx>(vals),
                        ddc::Coordinate<continuous_dimension_type<Idx>>(coord_eval))...);

        double y = 0.0;
        for_each(
                std::array<std::size_t, dimension> {(bsplines_type<Idx>::degree() + 1)...},
                [&](auto idx) {
                    y += spline_coef(
                                 ddc::DiscreteElement<bsplines_type<Idx>...>(
                                         (std::get<Idx>(jmin) + std::get<Idx>(idx))...))
                         * (std::get<Idx>(vals)[std::get<Idx>(idx)] * ...);
                });

        return y;
    }
};

template <typename... Args>
struct SplineEvaluatorNDHelper;

template <
        typename ExecSpace,
        typename MemorySpace,
        typename... BSplines,
        typename... EvaluationDDim,
        typename... LowerExtrapolationRule,
        typename... UpperExtrapolationRule>
struct SplineEvaluatorNDHelper<
        ExecSpace,
        MemorySpace,
        ddc::detail::TypeSeq<BSplines...>,
        ddc::detail::TypeSeq<EvaluationDDim...>,
        ddc::detail::TypeSeq<LowerExtrapolationRule...>,
        ddc::detail::TypeSeq<UpperExtrapolationRule...>>
{
    using type = SplineEvaluatorND<
            ExecSpace,
            MemorySpace,
            ddc::detail::TypeSeq<BSplines...>,
            ddc::detail::TypeSeq<EvaluationDDim...>,
            ddc::detail::TypeSeq<LowerExtrapolationRule...>,
            ddc::detail::TypeSeq<UpperExtrapolationRule...>,
            std::make_index_sequence<sizeof...(BSplines)>>;
};

} // namespace detail

template <typename... Args>
using SplineEvaluatorND = typename detail::SplineEvaluatorNDHelper<Args...>::type;

} // namespace ddc
