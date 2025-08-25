// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <list>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#define VALUES_X                                                                                   \
    {ddc::Coordinate<DimX>(0.1),                                                                   \
     ddc::Coordinate<DimX>(0.2),                                                                   \
     ddc::Coordinate<DimX>(0.3),                                                                   \
     ddc::Coordinate<DimX>(0.4)}

#define VALUES_Y                                                                                   \
    {ddc::Coordinate<DimY>(0.1),                                                                   \
     ddc::Coordinate<DimY>(0.2),                                                                   \
     ddc::Coordinate<DimY>(0.3),                                                                   \
     ddc::Coordinate<DimY>(0.4)}

inline namespace anonymous_namespace_workaround_non_uniform_point_sampling_cpp {

struct DimX;
struct DimY;

struct DDimX : ddc::NonUniformPointSampling<DimX>
{
};

struct DDimY : ddc::NonUniformPointSampling<DimY>
{
};

std::array<ddc::Coordinate<DimX>, 4> const array_points_x VALUES_X;
std::list<ddc::Coordinate<DimX>> const list_points_x VALUES_X;
std::vector<ddc::Coordinate<DimX>> const vector_points_x VALUES_X;

std::vector<ddc::Coordinate<DimY>> const vector_points_y VALUES_Y;

ddc::DiscreteElement<DDimX> constexpr point_ix(2);
ddc::Coordinate<DimX> constexpr point_rx(0.3);

ddc::DiscreteElement<DDimY> constexpr point_iy(1);
ddc::Coordinate<DimY> constexpr point_ry(0.2);

ddc::DiscreteElement<DDimX, DDimY> constexpr point_ixy(2, 1);
ddc::Coordinate<DimX, DimY> constexpr point_rxy(0.3, 0.2);

} // namespace anonymous_namespace_workaround_non_uniform_point_sampling_cpp

TEST(NonUniformPointSamplingTest, InitializerListConstructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(VALUES_X);
    EXPECT_EQ(ddim_x.size(), 4);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointSamplingTest, ArrayConstructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(array_points_x);
    EXPECT_EQ(ddim_x.size(), array_points_x.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointSamplingTest, VectorConstructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(vector_points_x);
    EXPECT_EQ(ddim_x.size(), vector_points_x.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointSamplingTest, ListConstructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(list_points_x);
    EXPECT_EQ(ddim_x.size(), list_points_x.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointSamplingTest, NotSortedVectorConstructor)
{
    std::vector unordered_vector_points_x = vector_points_x;
    std::swap(unordered_vector_points_x.front(), unordered_vector_points_x.back());
    EXPECT_THROW(
            (DDimX::Impl<DDimX, Kokkos::HostSpace>(unordered_vector_points_x)),
            std::runtime_error);
}

TEST(NonUniformPointSamplingTest, IteratorConstructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const
            ddim_x(vector_points_x.begin(), vector_points_x.end());
    EXPECT_EQ(ddim_x.size(), vector_points_x.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointSampling, Formatting)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(
            {ddc::Coordinate<DimX>(0.1), ddc::Coordinate<DimX>(0.4)});
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "NonUniformPointSampling(2)");
}

TEST(NonUniformPointSampling, Coordinate)
{
    ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(vector_points_x));
    ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(vector_points_y));
    EXPECT_EQ(ddc::coordinate(point_ix), point_rx);
    EXPECT_EQ(ddc::coordinate(point_iy), point_ry);
    EXPECT_EQ(ddc::coordinate(point_ixy), point_rxy);
}
