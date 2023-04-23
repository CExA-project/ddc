// SPDX-License-Identifier: MIT
#include <array>
#include <sstream>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/experimental/non_uniform_mesh.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#define VALUES                                                                                     \
    {                                                                                              \
        ddc::Coordinate<DimX>(0.1), ddc::Coordinate<DimX>(0.2), ddc::Coordinate<DimX>(0.3),        \
                ddc::Coordinate<DimX>(0.4)                                                         \
    }

namespace {

struct DimX;

using DSDimX = ddc::experimental::NonUniformMesh<DimX>;

static std::array<double, 4> const array_points VALUES;
static std::vector<double> const vector_points VALUES;
ddc::DiscreteElementType constexpr point_ix = ddc::DiscreteElementType(2);
ddc::Coordinate<DimX> constexpr point_rx = ddc::Coordinate<DimX>(0.3);

} // namespace

TEST(NonUniformPointMeshTest, ListConstructor)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(VALUES);
    EXPECT_EQ(ddim_x.size(), 4);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointMeshTest, ArrayConstructor)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(array_points);
    EXPECT_EQ(ddim_x.size(), array_points.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointMeshTest, VectorConstructor)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(vector_points);
    EXPECT_EQ(ddim_x.size(), vector_points.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointMeshTest, IteratorConstructor)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(vector_points.begin(), vector_points.end());
    EXPECT_EQ(ddim_x.size(), vector_points.size());
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(NonUniformPointMesh, Formatting)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(
            {ddc::Coordinate<DimX>(0.1), ddc::Coordinate<DimX>(0.4)});
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "NonUniformMesh(2)");
}
