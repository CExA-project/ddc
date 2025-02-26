// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "quadrature.hpp"

TEST(Quadrature, VersionA)
{
    ddc::DiscreteDomain<GridVx> const gridvx
            = ddc::init_trivial_space(ddc::DiscreteVector<GridVx>(100));

    ddc::Chunk quadrature_coeffs(gridvx, ddc::DeviceAllocator<double>());
    ddc::parallel_fill(quadrature_coeffs, 1);

    // density
    double const density_res
            = integrate(Kokkos::DefaultExecutionSpace(), quadrature_coeffs.span_cview());
    EXPECT_DOUBLE_EQ(density_res, 100);
}
