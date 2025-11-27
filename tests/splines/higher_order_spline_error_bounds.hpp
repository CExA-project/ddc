// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <ddc/kernels/splines.hpp>

template <class Evaluator>
class HigherOrderSplineErrorBounds
{
private:
    static double C(int k, int degree)
    {
        using Kokkos::numbers::pi;
        return ddc::detail::ipow((pi * pi) / 2, k) / (2 * ddc::detail::ipow(pi, degree));
    }

    /*******************************************************************************
     * Error bound in max norm for spline interpolation from:
     *
     * F. Dubeau and J. Savoie
     * https://doi.org/10.1006/jath.1995.1064 (in the conclusion)
     *******************************************************************************/
    static double dubeau_error_bound(int order, int degree, double cell_width, double norm)
    {
        return C(order, degree) * ddc::detail::ipow(cell_width, degree + 1 - order) * norm;
    }

    template <class DDimI, class... DDims>
    double max_norm(
            ddc::DiscreteElement<DDims...> const& orders,
            std::array<int, sizeof...(DDims)> const& degrees) const
    {
        return m_evaluator.max_norm(
                (orders.template uid<DDims>()
                 + (std::is_same_v<DDims, DDimI>
                            ? degrees[ddc::type_seq_rank_v<DDimI, ddc::detail::TypeSeq<DDims...>>]
                            : 0))...);
    }

    Evaluator m_evaluator;

public:
    explicit HigherOrderSplineErrorBounds(Evaluator const& evaluator) : m_evaluator(evaluator) {}

    template <class... DDims>
    double error_bound(
            ddc::DiscreteElement<DDims...> const& orders,
            std::array<double, sizeof...(DDims)> const& cell_width,
            std::array<int, sizeof...(DDims)> const& degrees) const
    {
        using ddims = ddc::detail::TypeSeq<DDims...>;
        return (dubeau_error_bound(
                        orders.template uid<DDims>(),
                        degrees[ddc::type_seq_rank_v<DDims, ddims>],
                        cell_width[ddc::type_seq_rank_v<DDims, ddims>],
                        max_norm<DDims>(orders, degrees))
                + ...);
    }
};
