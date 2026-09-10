// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <utility>

#include <ddc/kernels/splines.hpp>

template <class Evaluator>
class SplineErrorBounds
{
private:
    static constexpr std::array<ddc::Real, 10> tikhomirov_error_bound_array
            = std::array<ddc::Real, 10>(
                    {static_cast<ddc::Real>(1.0 / 2.0),
                     static_cast<ddc::Real>(1.0 / 8.0),
                     static_cast<ddc::Real>(1.0 / 24.0),
                     static_cast<ddc::Real>(5.0 / 384.0),
                     static_cast<ddc::Real>(1.0 / 240.0),
                     static_cast<ddc::Real>(61.0 / 46080.0),
                     static_cast<ddc::Real>(17.0 / 40320.0),
                     static_cast<ddc::Real>(277.0 / 2064384.0),
                     static_cast<ddc::Real>(31.0 / 725760.0),
                     static_cast<ddc::Real>(50521.0 / 3715891200.0)});

    Evaluator m_evaluator;

    /*******************************************************************************
     * Error bound in max norm for spline interpolation of periodic functions from:
     *
     * V M Tikhomirov 1969 Math. USSR Sb. 9 275
     * https://doi.org/10.1070/SM1969v009n02ABEH002052 (page 286, bottom)
     *
     * Yu. S. Volkov and Yu. N. Subbotin
     * https://doi.org/10.1134/S0081543815020236 (equation 14)
     *
     * Also applicable to first derivative by passing deg-1 instead of deg
     * Volkov & Subbotin 2015, eq. 15
     *******************************************************************************/
    static ddc::Real tikhomirov_error_bound(ddc::Real cell_width, int degree, ddc::Real max_norm)
    {
        degree = std::min(degree, 9);
        return tikhomirov_error_bound_array[degree] * ddc::detail::ipow(cell_width, degree + 1)
               * max_norm;
    }

    /// @brief Computes the max norm on the ith component
    template <std::size_t N, std::size_t... Ints>
    ddc::Real max_norm(
            std::array<ddc::DiscreteElementType, N> const& orders,
            std::array<std::size_t, N> const& degrees,
            std::size_t i,
            std::index_sequence<Ints...>) const
    {
        return m_evaluator.max_norm((Ints == i ? degrees[i] + 1 : orders[Ints])...);
    }

public:
    explicit SplineErrorBounds(Evaluator const& evaluator) : m_evaluator(evaluator) {}

    ddc::Real error_bound(ddc::Real cell_width, int degree) const
    {
        return tikhomirov_error_bound(cell_width, degree, m_evaluator.max_norm(degree + 1));
    }

    ddc::Real error_bound(ddc::Real cell_width1, ddc::Real cell_width2, int degree1, int degree2)
            const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1);
        return tikhomirov_error_bound(cell_width1, degree1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2, norm2);
    }

    ddc::Real error_bound(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1, 0);
        ddc::Real const norm3 = m_evaluator.max_norm(0, 0, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2, norm2)
               + tikhomirov_error_bound(cell_width3, degree3, norm3);
    }

    ddc::Real error_bound_on_deriv(ddc::Real cell_width, int degree) const
    {
        return tikhomirov_error_bound(cell_width, degree - 1, m_evaluator.max_norm(degree + 1));
    }

    ddc::Real error_bound_on_deriv_1(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            int degree1,
            int degree2) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1);
        return tikhomirov_error_bound(cell_width1, degree1 - 1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2, norm2);
    }

    ddc::Real error_bound_on_deriv_1(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1, 0);
        ddc::Real const norm3 = m_evaluator.max_norm(0, 0, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1 - 1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2, norm2)
               + tikhomirov_error_bound(cell_width3, degree3, norm3);
    }

    ddc::Real error_bound_on_deriv_2(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            int degree1,
            int degree2) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1);
        return tikhomirov_error_bound(cell_width1, degree1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2 - 1, norm2);
    }

    ddc::Real error_bound_on_deriv_2(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1, 0);
        ddc::Real const norm3 = m_evaluator.max_norm(0, 0, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2 - 1, norm2)
               + tikhomirov_error_bound(cell_width3, degree3, norm3);
    }

    ddc::Real error_bound_on_deriv_3(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1, 0);
        ddc::Real const norm3 = m_evaluator.max_norm(0, 0, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2, norm2)
               + tikhomirov_error_bound(cell_width3, degree3 - 1, norm3);
    }

    /*******************************************************************************
     * NOTE: The following estimates have no theoretical justification but capture
     *       the correct asympthotic rate of convergence.
     *       The error constant may be overestimated.
     *******************************************************************************/
    ddc::Real error_bound_on_deriv_12(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            int degree1,
            int degree2) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 1);
        ddc::Real const norm2 = m_evaluator.max_norm(1, degree2 + 1);
        return tikhomirov_error_bound(cell_width1, degree1 - 1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2 - 1, norm2);
    }

    ddc::Real error_bound_on_deriv_12(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 1, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(1, degree2 + 1, 0);
        ddc::Real const norm3 = m_evaluator.max_norm(0, 0, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1 - 1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2 - 1, norm2)
               + tikhomirov_error_bound(cell_width3, degree3, norm3);
    }

    ddc::Real error_bound_on_deriv_23(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0, 0);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1, 1);
        ddc::Real const norm3 = m_evaluator.max_norm(0, 1, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2 - 1, norm2)
               + tikhomirov_error_bound(cell_width3, degree3 - 1, norm3);
    }

    ddc::Real error_bound_on_deriv_13(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 0, 1);
        ddc::Real const norm2 = m_evaluator.max_norm(0, degree2 + 1, 0);
        ddc::Real const norm3 = m_evaluator.max_norm(1, 0, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1 - 1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2, norm2)
               + tikhomirov_error_bound(cell_width3, degree3 - 1, norm3);
    }

    ddc::Real error_bound_on_deriv_123(
            ddc::Real cell_width1,
            ddc::Real cell_width2,
            ddc::Real cell_width3,
            int degree1,
            int degree2,
            int degree3) const
    {
        ddc::Real const norm1 = m_evaluator.max_norm(degree1 + 1, 1, 1);
        ddc::Real const norm2 = m_evaluator.max_norm(1, degree2 + 1, 1);
        ddc::Real const norm3 = m_evaluator.max_norm(1, 1, degree3 + 1);
        return tikhomirov_error_bound(cell_width1, degree1 - 1, norm1)
               + tikhomirov_error_bound(cell_width2, degree2 - 1, norm2)
               + tikhomirov_error_bound(cell_width3, degree3 - 1, norm3);
    }

    ddc::Real error_bound_on_int(ddc::Real cell_width, int degree) const
    {
        return tikhomirov_error_bound(cell_width, degree + 1, m_evaluator.max_norm(degree + 1));
    }

    /*******************************************************************************
     * NOTE: We assume that the error bound formula for derivatives of order 0 and
     * 1 works for higher orders as well.
     *******************************************************************************/

    template <std::size_t N>
    ddc::Real error_bound(
            std::array<ddc::DiscreteElementType, N> const& orders,
            std::array<ddc::Real, N> const& cell_width,
            std::array<std::size_t, N> const& degrees) const
    {
        ddc::Real error = 0.;
        for (std::size_t i = 0; i < N; i++) {
            error += tikhomirov_error_bound(
                    cell_width[i],
                    degrees[i] - orders[i],
                    max_norm(orders, degrees, i, std::make_index_sequence<N> {}));
        }
        return error;
    }
};
