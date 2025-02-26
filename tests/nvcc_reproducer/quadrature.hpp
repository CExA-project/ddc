#pragma once

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"

struct Vx
{
};

struct GridVx : ddc::UniformPointSampling<Vx>
{
};

inline double integrate(
        Kokkos::DefaultExecutionSpace const& exec_space,
        ddc::ChunkSpan<
                const double,
                ddc::DiscreteDomain<GridVx>,
                Kokkos::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space> const& coeffs,
        ddc::ChunkSpan<
                const double,
                ddc::DiscreteDomain<GridVx>,
                Kokkos::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space> const& integrated_function)
{
    return ddc::parallel_transform_reduce(
            exec_space,
            coeffs.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<GridVx> const ix) {
                return coeffs(ix) * integrated_function(ix);
            });
}
