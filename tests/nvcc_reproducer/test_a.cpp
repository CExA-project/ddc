// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "quadrature.hpp"

TEST(Quadrature, VersionA)
{
    ddc::DiscreteVector<GridVx> const n(100);
    ddc::DiscreteDomain<GridVx> const gridvx = ddc::init_trivial_space(n);

    ddc::Chunk quadrature_coeffs(gridvx, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(quadrature_coeffs, 1);

    EXPECT_EQ(integrate(quadrature_coeffs.span_cview()), n.value());
}
