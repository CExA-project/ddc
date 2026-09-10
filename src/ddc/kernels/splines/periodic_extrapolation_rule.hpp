// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/real_type.hpp>

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

namespace ddc {

/**
 * @brief A functor for describing a spline boundary value by a periodic extrapolation for 1D evaluator.
 *
 * For a periodic domain, any position outside the domain should be equivalent to a point inside the
 * domain. As a result boundary conditions should never be called.
 */
template <class DimI>
struct PeriodicExtrapolationRule
{
    static_assert(DimI::PERIODIC, "PeriodicExtrapolationRule requires periodic dimension");

    /**
     * @brief Get the value of the function on B-splines at a coordinate outside the domain.
     *
     * As all coordinates outside the domain are equivalent to coordinates inside the domain,
     * this function raises an assertion error if it is ever called.
     *
     * @param[in] pos The coordinate where we want to evaluate the function on B-splines.
     * @param[in] spline_coef The coefficients of the function on B-splines.
     *
     * @return A Real with the value of the function on B-splines evaluated at the coordinate.
     */
    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION Real operator()(CoordType, ChunkSpan) const
    {
        KOKKOS_ASSERT("PeriodicExtrapolationRule::operator() should never be called")

        return 0.;
    }
};

} // namespace ddc
