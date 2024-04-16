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
    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION double operator()(CoordType, ChunkSpan) const
    {
        return 0.0;
    }
};
} // namespace ddc
