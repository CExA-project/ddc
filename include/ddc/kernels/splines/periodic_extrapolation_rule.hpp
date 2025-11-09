// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>

namespace ddc {

template <class DimI>
struct PeriodicExtrapolationRule
{
    static_assert(DimI::PERIODIC, "PeriodicExtrapolationRule requires periodic dimension");

    template <class CoordType, class ChunkSpan>
    KOKKOS_FUNCTION double operator()(CoordType, ChunkSpan) const
    {
        KOKKOS_ASSERT(("PeriodicExtrapolationRule::operator() should never be called"))

        return 0.;
    }
};

} // namespace ddc
