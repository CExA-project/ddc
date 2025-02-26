#pragma once

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

struct GridVx
{
};

inline double integrate(
        Kokkos::DefaultExecutionSpace const& exec_space,
        ddc::ChunkSpan<
                const double,
                ddc::DiscreteDomain<GridVx>,
                Kokkos::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space> const& integrated_function)
{
    return ddc::parallel_transform_reduce(
            exec_space,
            integrated_function.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridVx> const ix) {
                return integrated_function(ix);
            });
}
