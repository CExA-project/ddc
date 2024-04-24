// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ddc {

/**
 * @brief A functor for describing a spline boundary value by a null extrapolation for 1D evaluator.
 */
struct NullExtrapolationRule
{
    /**
     * @brief Evaluates 0. out of the domain.
     *
     * @param[in] pos
     *          The coordinate where we want to evaluate the function on B-splines.
     * @param[in] spline_coef
     *          The coefficients of the function on B-splines.
     *
     * @return A double with the value of the function on B-splines evaluated at the coordinate (here, 0.).
     */
    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION double operator()(CoordType, ChunkSpan) const
    {
        return 0.0;
    }
};
} // namespace ddc
