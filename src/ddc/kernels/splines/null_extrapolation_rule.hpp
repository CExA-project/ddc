// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/real_type.hpp>

#include <Kokkos_Macros.hpp>

namespace ddc {

/**
 * @brief A functor describing a null extrapolation boundary value for 1D spline evaluator.
 */
struct NullExtrapolationRule
{
    /**
     * @brief Evaluates the spline at a coordinate outside of the domain.
     *
     * @return A Real with the value of the function outside the domain (here, 0.).
     */
    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION Real operator()(CoordType, ChunkSpan) const
    {
        return 0.0;
    }
};

} // namespace ddc
