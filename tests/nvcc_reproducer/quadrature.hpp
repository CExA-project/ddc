// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

struct DDimX
{
};

inline int sum(ddc::ChunkSpan<
               const int,
               ddc::DiscreteDomain<DDimX>,
               Kokkos::layout_right,
               Kokkos::DefaultExecutionSpace::memory_space> const& chk_span)
{
    return ddc::parallel_transform_reduce(
            chk_span.domain(),
            0,
            ddc::reducer::sum<int>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX> const ix) { return chk_span(ix); });
}
