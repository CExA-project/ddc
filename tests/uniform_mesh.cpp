// SPDX-License-Identifier: MIT
#include <memory>
#include <sstream>

#include <ddc/ddc.hpp>
#include <ddc/experimental/uniform_mesh.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

struct DimX
{
};

struct DSDimX : ddc::experimental::UniformMesh<DimX>
{
};

ddc::Coordinate<DimX> constexpr origin(-1.);
ddc::Coordinate<DimX> constexpr step(0.5);
ddc::DiscreteVectorElement constexpr npoints(5);
ddc::DiscreteElementType constexpr lbound(0);
ddc::DiscreteElementType constexpr point_ix(2);
ddc::Coordinate<DimX> constexpr point_rx(0.);

} // namespace

TEST(UniformMeshTest, Constructor)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(origin, step);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(UniformMeshTest, Formatting)
{
    DSDimX::Impl<Kokkos::HostSpace> ddim_x(ddc::Coordinate<DimX>(-1.), ddc::Coordinate<DimX>(0.5));
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "UniformMesh( origin=(-1), step=0.5 )");
}
