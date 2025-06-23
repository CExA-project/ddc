// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

inline namespace anonymous_namespace_workaround_discrete_space_cpp {

struct DimX;
struct DDimX : ddc::UniformPointSampling<DimX>
{
};

} // namespace anonymous_namespace_workaround_discrete_space_cpp

TEST(DiscreteSpace, Initialization)
{
    EXPECT_FALSE(ddc::is_discrete_space_initialized<DDimX>());
#if !defined(NDEBUG) // The assertion is only checked if NDEBUG isn't defined
    EXPECT_DEATH(
            ddc::host_discrete_space<DDimX>(),
            R"rgx([Aa]ssert.*is_discrete_space_initialized<DDim>\(\))rgx");
    EXPECT_DEATH(
            ddc::discrete_space<DDimX>(),
            R"rgx([Aa]ssert.*is_discrete_space_initialized<DDim>\(\))rgx");
#endif
    ddc::init_discrete_space<DDimX>(DDimX::template init<DDimX>(
            ddc::Coordinate<DimX>(0),
            ddc::Coordinate<DimX>(1),
            ddc::DiscreteVector<DDimX>(2)));
    EXPECT_THROW(
            ddc::init_discrete_space<DDimX>(DDimX::template init<DDimX>(
                    ddc::Coordinate<DimX>(0),
                    ddc::Coordinate<DimX>(1),
                    ddc::DiscreteVector<DDimX>(2))),
            std::runtime_error);
    ASSERT_TRUE(ddc::is_discrete_space_initialized<DDimX>());
    EXPECT_EQ(
            static_cast<void const*>(&ddc::discrete_space<DDimX>()),
            static_cast<void const*>(&ddc::host_discrete_space<DDimX>()));
}
