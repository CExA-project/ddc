// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

inline namespace anonymous_namespace_workaround_trivial_dimension_cpp {

struct DDimX
{
};

} // namespace anonymous_namespace_workaround_trivial_dimension_cpp

TEST(TrivialBoundedSpace, Size)
{
    ddc::DiscreteVector<DDimX> constexpr n(10);
    ddc::DiscreteDomain<DDimX> constexpr ddom = ddc::init_trivial_bounded_space(n);
    EXPECT_EQ(ddom.extents(), n);
    EXPECT_EQ(ddom.front().uid(), 0);
}

TEST(TrivialHalfBoundedSpace, Size)
{
    ddc::DiscreteElement<DDimX> constexpr delem = ddc::init_trivial_half_bounded_space<DDimX>();
    EXPECT_EQ(delem.uid(), 0);
}
