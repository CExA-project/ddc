// SPDX-License-Identifier: MIT
#include <memory>
#include <sstream>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

struct DimX;

using DDimX = ddc::UniformPointSampling<DimX>;

static ddc::Coordinate<DimX> constexpr origin(-1.);
static double constexpr step = 0.5;
static ddc::DiscreteElement<DDimX> constexpr point_ix(2);
static ddc::Coordinate<DimX> constexpr point_rx(0.);

} // namespace

TEST(UniformPointSamplingTest, Constructor)
{
    DDimX::Impl<Kokkos::HostSpace> ddim_x(origin, step);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(UniformPointSampling, Formatting)
{
    DDimX::Impl<Kokkos::HostSpace> ddim_x(origin, step);
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "UniformPointSampling( origin=(-1), step=0.5 )");
}

TEST_F(UniformPointSamplingTest, Coordinate)
{
    struct DimY;
    using DDimY = ddc::UniformPointSampling<DimY>;
    ddc::DiscreteElement<DDimY> point_iy = ddc::DiscreteElement<DDimY>(4);
    ddc::Coordinate<DimY> point_ry = ddc::Coordinate<DimY>(-6.);

    ddc::DiscreteElement<DDimX, DDimY> point_ixy(point_ix, point_iy);
    ddc::Coordinate<DimX, DimY> point_rxy(point_rx, point_ry);

    ddc::init_discrete_space<DDimX>(origin, step);
    ddc::init_discrete_space<DDimY>(ddc::Coordinate<DimY>(-10.), 1.);
    EXPECT_EQ(coordinate(point_ix), point_rx);
    EXPECT_EQ(coordinate(point_iy), point_ry);
    EXPECT_EQ(coordinate(point_ixy), point_rxy);
}
