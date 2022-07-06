// SPDX-License-Identifier: MIT
#include <memory>
#include <sstream>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

struct CDimX;

using DDimX = UniformPointSampling<CDimX>;

class UniformPointSamplingTest : public ::testing::Test
{
protected:
    Coordinate<CDimX> origin = Coordinate<CDimX>(-1.);
    Coordinate<CDimX> step = Coordinate<CDimX>(0.5);
    DiscreteVectorElement npoints = 5;
    DiscreteElementType lbound = 0;
    DiscreteElementType point_ix = 2;
    Coordinate<CDimX> point_rx = Coordinate<CDimX>(0.);
};

TEST_F(UniformPointSamplingTest, Rank)
{
    EXPECT_EQ(DDimX::discretization_type::rank(), 1);
}

TEST_F(UniformPointSamplingTest, Constructor)
{
    DDimX::Impl<Kokkos::HostSpace> ddim_x(origin, step);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST_F(UniformPointSamplingTest, Constructor2)
{
    DDimX::Impl<Kokkos::HostSpace> ddim_x(origin, Coordinate<CDimX>(1.), npoints);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(UniformPointSampling, Formatting)
{
    DDimX::Impl<Kokkos::HostSpace> ddim_x(Coordinate<CDimX>(-1.), Coordinate<CDimX>(0.5));
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "UniformPointSampling( origin=(-1), step=(0.5) )");
}
