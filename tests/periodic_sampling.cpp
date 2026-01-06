// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <string>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_periodic_sampling_cpp {

struct DimX;
struct DimY;

struct DDimX : ddc::PeriodicSampling<DimX>
{
};

struct DDimY : ddc::PeriodicSampling<DimY>
{
};

ddc::Coordinate<DimX> constexpr origin(-1.);
ddc::Real constexpr step = 0.5;
ddc::DiscreteElement<DDimX> constexpr point_ix(2);
ddc::Coordinate<DimX> constexpr point_rx(0.);

} // namespace anonymous_namespace_workaround_periodic_sampling_cpp

TEST(PeriodicSamplingTest, Constructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(origin, step, 10);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.n_period(), 10);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(PeriodicSampling, Formatting)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(origin, step, 10);
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "PeriodicSampling( origin=(-1), step=0.5 )");
}

TEST(PeriodicSamplingTest, Coordinate)
{
    ddc::DiscreteElement<DDimY> const point_iy(4);
    ddc::Coordinate<DimY> const point_ry(-6);

    ddc::DiscreteElement<DDimX, DDimY> const point_ixy(point_ix, point_iy);
    ddc::Coordinate<DimX, DimY> const point_rxy(point_rx, point_ry);

    ddc::init_discrete_space<DDimX>(origin, step, 10);
    ddc::init_discrete_space<DDimY>(ddc::Coordinate<DimY>(-10.), 1., 10);
    EXPECT_EQ(ddc::coordinate(point_ix), point_rx);
    EXPECT_EQ(ddc::coordinate(point_iy), point_ry);
    EXPECT_EQ(ddc::coordinate(point_ixy), point_rxy);
}

TEST(PeriodicSamplingTest, Attributes)
{
    ddc::DiscreteDomain<DDimX> const ddom_x(point_ix, ddc::DiscreteVector<DDimX>(2));
    ddc::init_discrete_space<DDimX>(origin, step, 10);
    EXPECT_EQ(ddc::origin<DDimX>(), origin);
    EXPECT_EQ(ddc::step<DDimX>(), step);
    EXPECT_EQ(ddc::rmin(ddom_x), point_rx);
    EXPECT_EQ(ddc::rmax(ddom_x), point_rx + step);
    EXPECT_EQ(ddc::rlength(ddom_x), step);
    EXPECT_EQ(ddc::distance_at_left(point_ix), step);
    EXPECT_EQ(ddc::distance_at_right(point_ix), step);
}
