// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace {

class DDimX;

} // namespace

TEST(TrivialDimension, Size)
{
    ddc::DiscreteVector<DDimX> const n(10);
    ddc::DiscreteDomain<DDimX> const ddom = ddc::init_trivial_space(n);
    EXPECT_EQ(ddom.extents(), n);
}
