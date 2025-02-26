#pragma once

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

struct GridVx
{
};

inline int integrate(ddc::ChunkSpan<
                     const int,
                     ddc::DiscreteDomain<GridVx>,
                     Kokkos::layout_right,
                     Kokkos::DefaultExecutionSpace::memory_space> const& integrated_function)
{
    return ddc::parallel_transform_reduce(
            integrated_function.domain(),
            0,
            ddc::reducer::sum<int>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridVx> const ix) {
                return integrated_function(ix);
            });
}
