// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(TRIVIAL_DIMENSION_CPP) {

class DDimX;

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(TRIVIAL_DIMENSION_CPP)

TEST(TrivialDimension, Size)
{
    ddc::DiscreteVector<DDimX> const n(10);
    ddc::DiscreteDomain<DDimX> const ddom = ddc::init_trivial_space(n);
    EXPECT_EQ(ddom.extents(), n);
}
