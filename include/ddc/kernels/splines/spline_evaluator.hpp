// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "integrals.hpp"
#include "periodic_extrapolation_rule.hpp"

namespace ddc {

/**
 * @brief A class to evaluate, differentiate or integrate a spline function.
 *
 * A class which contains an operator () which can be used to evaluate, differentiate or integrate a spline function.
 *
 * @tparam ExecSpace The Kokkos execution space on which the spline evaluation is performed.
 * @tparam MemorySpace The Kokkos memory space on which the data (spline coefficients and evaluation) is stored.
 * @tparam BSplines The discrete dimension representing the B-splines.
 * @tparam EvaluationDDim The discrete dimension on which evaluation points are defined.
 * @tparam LowerExtrapolationRule The lower extrapolation rule type.
 * @tparam UpperExtrapolationRule The upper extrapolation rule type.
 * @tparam IDimX A variadic template of all the discrete dimensions forming the full space (EvaluationDDim + batched dimensions).
 */
template <
        class ExecSpace,
        class MemorySpace,
        class BSplines,
        class EvaluationDDim,
        class LowerExtrapolationRule,
        class UpperExtrapolationRule,
        class... IDimX>
class SplineEvaluator
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
    /// @brief The type of the Kokkos execution space used by this class.
    using exec_space = ExecSpace;

    /// @brief The type of the Kokkos memory space used by this class.
    using memory_space = MemorySpace;

    /// @brief The type of the evaluation continuous dimension (continuous dimension of interest) used by this class.
    using continuous_dimension_type = typename BSplines::continuous_dimension_type;

    /// @brief The type of the evaluation discrete dimension (discrete dimension of interest) used by this class.
    using evaluation_discrete_dimension_type = EvaluationDDim;

    /// @brief The discrete dimension representing the B-splines.
    using bsplines_type = BSplines;

    /// @brief The type of the domain for the 1D evaluation mesh used by this class.
    using evaluation_domain_type = ddc::DiscreteDomain<evaluation_discrete_dimension_type>;

    /// @brief The type of the whole domain representing evaluation points.
    using batched_evaluation_domain_type = ddc::DiscreteDomain<IDimX...>;

    /// @brief The type of the 1D spline domain corresponding to the dimension of interest.
    using spline_domain_type = ddc::DiscreteDomain<bsplines_type>;

    /**
     * @brief The type of the batch domain (obtained by removing the dimension of interest
     * from the whole domain).
     */
    using batch_domain_type = ddc::
            remove_dims_of_t<batched_evaluation_domain_type, evaluation_discrete_dimension_type>;

    /**
     * @brief The type of the whole spline domain (cartesian product of 1D spline domain
     * and batch domain) preserving the order of dimensions.
     */
    using batched_spline_domain_type = ddc::replace_dim_of_t<
            batched_evaluation_domain_type,
            evaluation_discrete_dimension_type,
            bsplines_type>;

    /// @brief The type of the extrapolation rule at the lower boundary.
    using lower_extrapolation_rule_type = LowerExtrapolationRule;

    /// @brief The type of the extrapolation rule at the upper boundary.
    using upper_extrapolation_rule_type = UpperExtrapolationRule;


private:
    LowerExtrapolationRule m_lower_extrap_rule;

    UpperExtrapolationRule m_upper_extrap_rule;

public:
    static_assert(
            std::is_same_v<LowerExtrapolationRule,
                            typename ddc::PeriodicExtrapolationRule<continuous_dimension_type>>
                            == bsplines_type::is_periodic()
                    && std::is_same_v<
                               UpperExtrapolationRule,
                               typename ddc::PeriodicExtrapolationRule<continuous_dimension_type>>
                               == bsplines_type::is_periodic(),
            "PeriodicExtrapolationRule has to be used if and only if dimension is periodic");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    LowerExtrapolationRule,
                    ddc::Coordinate<continuous_dimension_type>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "LowerExtrapolationRule::operator() has to be callable with usual arguments.");
    static_assert(
            std::is_invocable_r_v<
                    double,
                    UpperExtrapolationRule,
                    ddc::Coordinate<continuous_dimension_type>,
                    ddc::ChunkSpan<
                            double const,
                            spline_domain_type,
                            Kokkos::layout_right,
                            memory_space>>,
            "UpperExtrapolationRule::operator() has to be callable with usual arguments.");

    /**
     * @brief Build a SplineEvaluator acting on batched_spline_domain.
     *
     * @param lower_extrap_rule The extrapolation rule at the lower boundary.
     * @param upper_extrap_rule The extrapolation rule at the upper boundary.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    explicit SplineEvaluator(
            LowerExtrapolationRule const& lower_extrap_rule,
            UpperExtrapolationRule const& upper_extrap_rule)
        : m_lower_extrap_rule(lower_extrap_rule)
        , m_upper_extrap_rule(upper_extrap_rule)
    {
    }

    /**
     * @brief Copy-constructs.
     *
     * @param x A reference to another SplineEvaluator.
     */
    SplineEvaluator(SplineEvaluator const& x) = default;

    /**
     * @brief Move-constructs.
     *
     * @param x An rvalue to another SplineEvaluator.
     */
    SplineEvaluator(SplineEvaluator&& x) = default;

    /// @brief Destructs
    ~SplineEvaluator() = default;

    /**
     * @brief Copy-assigns.
     *
     * @param x A reference to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluator& operator=(SplineEvaluator const& x) = default;

    /**
     * @brief Move-assigns.
     *
     * @param x An rvalue to another SplineEvaluator.
     * @return A reference to this object.
     */
    SplineEvaluator& operator=(SplineEvaluator&& x) = default;

    /**
     * @brief Get the lower extrapolation rule.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The lower extrapolation rule.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    lower_extrapolation_rule_type lower_extrapolation_rule() const
    {
        return m_lower_extrap_rule;
    }

    /**
     * @brief Get the upper extrapolation rule.
     *
     * Extrapolation rules are functors used to define the behavior of the SplineEvaluator out of the domain where the break points of the B-splines are defined.
     *
     * @return The upper extrapolation rule.
     *
     * @see NullExtrapolationRule ConstantExtrapolationRule PeriodicExtrapolationRule
     */
    upper_extrapolation_rule_type upper_extrapolation_rule() const
    {
        return m_upper_extrap_rule;
    }

    /**
     * @brief Evaluate 1D spline function (described by its spline coefficients) at a given coordinate.
     *
     * The spline coefficients represent a 1D spline function defined on a B-splines (basis splines). They can be obtained via various methods, such as using a SplineBuilder.
     *
     * Remark: calling SplineBuilder then SplineEvaluator corresponds to a spline interpolation.
     *
     * @param coord_eval The coordinate where the spline is evaluated. Note that only the component along the dimension of interest is used.
     * @param spline_coef A ChunkSpan storing the 1D spline coefficients.
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
     * @brief Evaluate spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder.
     *
     * This is not a multidimensional evaluation. This is a batched 1D evaluation. This means that for each slice of coordinates
     * identified by a batch_domain_type::discrete_element_type, the evaluation is performed with the 1D set of
     * spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilder then SplineEvaluator corresponds to a spline interpolation.
     *
     * @param[out] spline_eval The values of the spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is evaluated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 1D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the spline coefficients.
     */
    template <class Layout1, class Layout2, class Layout3, class... CoordsDims>
    void operator()(
            ddc::ChunkSpan<double, batched_evaluation_domain_type, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    batched_evaluation_domain_type,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<double const, batched_spline_domain_type, Layout3, memory_space> const
                    spline_coef) const
    {
        evaluation_domain_type const evaluation_domain(spline_eval.domain());
        batch_domain_type const batch_domain(spline_eval.domain());

        ddc::parallel_for_each(
                "ddc_splines_evaluate",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    const auto spline_eval_1D = spline_eval[j];
                    const auto coords_eval_1D = coords_eval[j];
                    const auto spline_coef_1D = spline_coef[j];
                    for (auto const i : evaluation_domain) {
                        spline_eval_1D(i) = eval(coords_eval_1D(i), spline_coef_1D);
                    }
                });
    }

    /**
     * @brief Evaluate spline function (described by its spline coefficients) on a
     * mesh.
     *
     * The spline coefficients represent a spline function defined on a cartesian
     * product of batch_domain and B-splines (basis splines). They can be obtained
     * via various methods, such as using a SplineBuilder.
     *
     * This is not a multidimensional evaluation. This is a batched 1D evaluation.
     * This means that for each slice of coordinates identified by a
     * batch_domain_type::discrete_element_type, the evaluation is performed with
     * the 1D set of spline coefficients identified by the same
     * batch_domain_type::discrete_element_type.
     *
     * Remark: calling SplineBuilder then SplineEvaluator corresponds to a spline
     * interpolation.
     *
     * @param[out] spline_eval The values of the spline function at the desired
     * coordinates. For practical reasons those are stored in a ChunkSpan defined
     * on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is evaluated. Those
     * are stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note
     * that the coordinates of the points represented by this domain are unused
     * and irrelevant (but the points themselves (DiscreteElement) are used to
     * select the set of 1D spline coefficients retained to perform the
     * evaluation).
     * @param[in] spline_coef A ChunkSpan storing the spline coefficients.
     */
    template <class Layout1, class Layout2>
    void operator()(
            ddc::ChunkSpan<double, batched_evaluation_domain_type, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<double const, batched_spline_domain_type, Layout2, memory_space> const
                    spline_coef) const
    {
        evaluation_domain_type const evaluation_domain(spline_eval.domain());
        batch_domain_type const batch_domain(spline_eval.domain());

        ddc::parallel_for_each(
                "ddc_splines_evaluate",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    const auto spline_eval_1D = spline_eval[j];
                    const auto spline_coef_1D = spline_coef[j];
                    for (auto const i : evaluation_domain) {
                        spline_eval_1D(i) = eval(
                                ddc::coordinate(ddc::select<evaluation_discrete_dimension_type>(j)),
                                spline_coef_1D);
                    }
                });
    }

    /**
     * @brief Differentiate 1D spline function (described by its spline coefficients) at a given coordinate.
     *
     * The spline coefficients represent a 1D spline function defined on a B-splines (basis splines). They can be
     * obtained via various methods, such as using a SplineBuilder.
     *
     * @param coord_eval The coordinate where the spline is differentiated. Note that only the component along the dimension of interest is used.
     * @param spline_coef A ChunkSpan storing the 1D spline coefficients.
     *
     * @return The derivative of the spline function at the desired coordinate.
     */
    template <class Layout, class... CoordsDims>
    KOKKOS_FUNCTION double deriv(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        return eval_no_bc<eval_deriv_type>(coord_eval, spline_coef);
    }

    /**
     * @brief Differentiate spline function (described by its spline coefficients) on a mesh.
     *
     * The spline coefficients represent a spline function defined on a cartesian product of batch_domain and B-splines
     * (basis splines). They can be obtained via various methods, such as using a SplineBuilder.
     *
     * The derivation is not performed in a multidimensional way (in any sense). This is a batched 1D derivation.
     * This means that for each slice of coordinates identified by a batch_domain_type::discrete_element_type,
     * the derivation is performed with the 1D set of spline coefficients identified by the same batch_domain_type::discrete_element_type.
     *
     * @param[out] spline_eval The derivatives of the spline function at the desired coordinates. For practical reasons those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type.
     * @param[in] coords_eval The coordinates where the spline is differentiated. Those are
     * stored in a ChunkSpan defined on a batched_evaluation_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant (but the points themselves (DiscreteElement) are used to select
     * the set of 1D spline coefficients retained to perform the evaluation).
     * @param[in] spline_coef A ChunkSpan storing the spline coefficients.
     */
    template <class Layout1, class Layout2, class Layout3, class... CoordsDims>
    void deriv(
            ddc::ChunkSpan<double, batched_evaluation_domain_type, Layout1, memory_space> const
                    spline_eval,
            ddc::ChunkSpan<
                    ddc::Coordinate<CoordsDims...> const,
                    batched_evaluation_domain_type,
                    Layout2,
                    memory_space> const coords_eval,
            ddc::ChunkSpan<double const, batched_spline_domain_type, Layout3, memory_space> const
                    spline_coef) const
    {
        evaluation_domain_type const evaluation_domain(spline_eval.domain());
        batch_domain_type const batch_domain(spline_eval.domain());

        ddc::parallel_for_each(
                "ddc_splines_differentiate",
                exec_space(),
                batch_domain,
                KOKKOS_CLASS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    const auto spline_eval_1D = spline_eval[j];
                    const auto coords_eval_1D = coords_eval[j];
                    const auto spline_coef_1D = spline_coef[j];
                    for (auto const i : evaluation_domain) {
                        spline_eval_1D(i)
                                = eval_no_bc<eval_deriv_type>(coords_eval_1D(i), spline_coef_1D);
                    }
                });
    }

    /** @brief Perform batched 1D integrations of a spline function (described by its spline coefficients) along the dimension of interest and store results on a subdomain of batch_domain.
     *
     * The spline coefficients represent a spline function defined on a B-splines (basis splines). They can be obtained via the SplineBuilder.
     *
     * The integration is not performed in a multidimensional way (in any sense). This is a batched 1D integration.
     * This means that for each element of integrals, the integration is performed with the 1D set of
     * spline coefficients identified by the same DiscreteElement.
     *
     * @param[out] integrals The integrals of the spline function on the subdomain of batch_domain. For practical reasons those are
     * stored in a ChunkSpan defined on a batch_domain_type. Note that the coordinates of the
     * points represented by this domain are unused and irrelevant.
     * @param[in] spline_coef A ChunkSpan storing the spline coefficients.
     */
    template <class Layout1, class Layout2>
    void integrate(
            ddc::ChunkSpan<double, batch_domain_type, Layout1, memory_space> const integrals,
            ddc::ChunkSpan<double const, batched_spline_domain_type, Layout2, memory_space> const
                    spline_coef) const
    {
        batch_domain_type const batch_domain(integrals.domain());
        ddc::Chunk values_alloc(
                ddc::DiscreteDomain<bsplines_type>(spline_coef.domain()),
                ddc::KokkosAllocator<double, memory_space>());
        ddc::ChunkSpan const values = values_alloc.span_view();
        ddc::integrals(exec_space(), values);

        ddc::parallel_for_each(
                "ddc_splines_integrate",
                exec_space(),
                batch_domain,
                KOKKOS_LAMBDA(typename batch_domain_type::discrete_element_type const j) {
                    integrals(j) = 0;
                    for (typename spline_domain_type::discrete_element_type const i :
                         values.domain()) {
                        integrals(j) += spline_coef(i, j) * values(i);
                    }
                });
    }

private:
    template <class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        ddc::Coordinate<continuous_dimension_type> coord_eval_interest
                = ddc::select<continuous_dimension_type>(coord_eval);
        if constexpr (bsplines_type::is_periodic()) {
            if (coord_eval_interest < ddc::discrete_space<bsplines_type>().rmin()
                || coord_eval_interest > ddc::discrete_space<bsplines_type>().rmax()) {
                coord_eval_interest -= Kokkos::floor(
                                               (coord_eval_interest
                                                - ddc::discrete_space<bsplines_type>().rmin())
                                               / ddc::discrete_space<bsplines_type>().length())
                                       * ddc::discrete_space<bsplines_type>().length();
            }
        } else {
            if (coord_eval_interest < ddc::discrete_space<bsplines_type>().rmin()) {
                return m_lower_extrap_rule(coord_eval_interest, spline_coef);
            }
            if (coord_eval_interest > ddc::discrete_space<bsplines_type>().rmax()) {
                return m_upper_extrap_rule(coord_eval_interest, spline_coef);
            }
        }
        return eval_no_bc<eval_type>(coord_eval_interest, spline_coef);
    }

    template <class EvalType, class Layout, class... CoordsDims>
    KOKKOS_INLINE_FUNCTION double eval_no_bc(
            ddc::Coordinate<CoordsDims...> const& coord_eval,
            ddc::ChunkSpan<double const, spline_domain_type, Layout, memory_space> const
                    spline_coef) const
    {
        static_assert(
                std::is_same_v<EvalType, eval_type> || std::is_same_v<EvalType, eval_deriv_type>);
        ddc::DiscreteElement<bsplines_type> jmin;
        std::array<double, bsplines_type::degree() + 1> vals_ptr;
        Kokkos::mdspan<double, Kokkos::extents<std::size_t, bsplines_type::degree() + 1>> const
                vals(vals_ptr.data());
        ddc::Coordinate<continuous_dimension_type> const coord_eval_interest
                = ddc::select<continuous_dimension_type>(coord_eval);
        if constexpr (std::is_same_v<EvalType, eval_type>) {
            jmin = ddc::discrete_space<bsplines_type>().eval_basis(vals, coord_eval_interest);
        } else if constexpr (std::is_same_v<EvalType, eval_deriv_type>) {
            jmin = ddc::discrete_space<bsplines_type>().eval_deriv(vals, coord_eval_interest);
        }
        double y = 0.0;
        for (std::size_t i = 0; i < bsplines_type::degree() + 1; ++i) {
            y += spline_coef(ddc::DiscreteElement<bsplines_type>(jmin + i)) * vals[i];
        }
        return y;
    }
};

} // namespace ddc
