// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

namespace ddc {

/**
 * @brief A functor to represent periodic extrapolation in a 1D spline evaluator.
 *
 * This rule is handled by the spline evaluator itself and therefore this
 * functor should never be invoked.
 */
template <class DimI>
struct PeriodicExtrapolationRule
{
    /**
     * @brief This function should never be called.
     *
     * @return Undefined.
     */
    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION double operator()(CoordType, ChunkSpan) const
    {
        KOKKOS_ASSERT("PeriodicExtrapolationRule::operator() should never be called")

        return 0.;
    }
};

} // namespace ddc
