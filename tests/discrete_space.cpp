// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(DISCRETE_SPACE_CPP) {

struct DimX;
struct DDimX : ddc::UniformPointSampling<DimX>
{
};

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(DISCRETE_SPACE_CPP)

TEST(DiscreteSpace, IsDiscreteSpaceInitialized)
{
    EXPECT_FALSE(ddc::is_discrete_space_initialized<DDimX>());
    ddc::create_uniform_point_sampling<DDimX>(
            ddc::Coordinate<DimX>(0),
            ddc::Coordinate<DimX>(1),
            ddc::DiscreteVector<DDimX>(2));
    EXPECT_TRUE(ddc::is_discrete_space_initialized<DDimX>());
}
