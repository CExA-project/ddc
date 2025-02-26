// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "quadrature.hpp"

TEST(SumVersionA, Small)
{
    ddc::DiscreteVector<DDimX> const n(10);
    ddc::DiscreteDomain<DDimX> const gridvx = ddc::init_trivial_space(n);

    ddc::Chunk quadrature_coeffs(gridvx, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(quadrature_coeffs, 1);

    EXPECT_EQ(sum(quadrature_coeffs.span_cview()), n.value());
}

TEST(SumVersionA, Bigger)
{
    ddc::DiscreteVector<DDimX> const n(1000);
    ddc::DiscreteDomain<DDimX> const gridvx = ddc::init_trivial_space(n);

    ddc::Chunk quadrature_coeffs(gridvx, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(quadrature_coeffs, 1);

    EXPECT_EQ(sum(quadrature_coeffs.span_cview()), n.value());
}
