// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <vector>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_parallel_transform_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(10);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(12);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);

struct ScaleBy2Fn
{
    [[nodiscard]] KOKKOS_FUNCTION int operator()(int const value) const noexcept
    {
        return 2 * value;
    }
};

ScaleBy2Fn const scale_by_2;

} // namespace anonymous_namespace_workaround_parallel_transform_cpp

TEST(ParallelTransform, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> const view(storage.data(), dom);
    ddc::parallel_fill(view, 1);
    ddc::parallel_transform(view, scale_by_2);
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 2), dom.size());
}

TEST(ParallelTransform, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomXY> const view(storage.data(), dom);
    ddc::parallel_fill(view, 1);
    ddc::parallel_transform(view, scale_by_2);
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 2), dom.size());
}

TEST(ParallelTransform, OneDimensionWithExecutionSpace)
{
    DDomX const dom(lbound_x, nelems_x);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> const view(storage.data(), dom);
    Kokkos::DefaultHostExecutionSpace const exec_space;
    ddc::parallel_fill(exec_space, view, 1);
    ddc::parallel_transform(exec_space, view, scale_by_2);
    exec_space.fence();
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 2), dom.size());
}

TEST(ParallelTransform, TwoDimensionsWithExecutionSpace)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomXY> const view(storage.data(), dom);
    Kokkos::DefaultHostExecutionSpace const exec_space;
    ddc::parallel_fill(exec_space, view, 1);
    ddc::parallel_transform(exec_space, view, scale_by_2);
    exec_space.fence();
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 2), dom.size());
}
