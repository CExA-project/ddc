// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_parallel_transform_scan_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

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

} // namespace anonymous_namespace_workaround_parallel_transform_scan_cpp

TEST(ParallelTransformScanDevice, XCumsumX)
{
    Kokkos::DefaultExecutionSpace const exec_space;

    DDomX const dom_x(lbound_x, nelems_x);

    ddc::Chunk chunk_x(dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_x, -1);

    ddc::Chunk chunk_x_in(dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_x_in, 1);

    {
        ddc::experimental::parallel_transform_exclusive_scan(
                "cumsum",
                exec_space,
                ddc::experimental::Dims<DDimX>(),
                chunk_x,
                ddc::reducer::sum<int>(),
                chunk_x_in.span_cview());

        auto const chunk_x_y_host = ddc::create_mirror_and_copy(chunk_x.span_cview());
        int expected = 0;
        for (DElemX const ix : dom_x) {
            EXPECT_EQ(chunk_x_y_host(ix), expected) << "at x=" << ix;
            ++expected;
        }
    }
    {
        ddc::experimental::parallel_transform_inclusive_scan(
                "cumsum",
                exec_space,
                ddc::experimental::Dims<DDimX>(),
                chunk_x,
                ddc::reducer::sum<int>(),
                chunk_x_in.span_cview());

        auto const chunk_x_y_host = ddc::create_mirror_and_copy(chunk_x.span_cview());
        int expected = 0;
        for (DElemX const ix : dom_x) {
            ++expected;
            EXPECT_EQ(chunk_x_y_host(ix), expected) << "at x=" << ix;
        }
    }
}

TEST(ParallelTransformScanDevice, XYCumprodX)
{
    Kokkos::DefaultExecutionSpace const exec_space;

    DDomXY const dom_x_y(lbound_x_y, nelems_x_y);

    ddc::Chunk chunk_x_y(dom_x_y, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_x_y, -1);

    ddc::Chunk chunk_x_y_in(dom_x_y, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_x_y_in, 2);

    {
        ddc::experimental::parallel_transform_exclusive_scan(
                "cumprod",
                exec_space,
                ddc::experimental::Dims<DDimX>(),
                chunk_x_y,
                ddc::reducer::prod<int>(),
                chunk_x_y_in.span_cview());

        auto const chunk_x_y_host = ddc::create_mirror_and_copy(chunk_x_y.span_cview());
        for (DElemY const iy : DDomY(dom_x_y)) {
            int expected = 1;
            for (DElemX const ix : DDomX(dom_x_y)) {
                EXPECT_EQ(chunk_x_y_host(ix, iy), expected) << "at x=" << ix << ", y=" << iy;
                expected *= 2;
            }
        }
    }
    {
        ddc::experimental::parallel_transform_inclusive_scan(
                "cumprod",
                exec_space,
                ddc::experimental::Dims<DDimX>(),
                chunk_x_y,
                ddc::reducer::prod<int>(),
                chunk_x_y_in.span_cview());

        auto const chunk_x_y_host = ddc::create_mirror_and_copy(chunk_x_y.span_cview());
        for (DElemY const iy : DDomY(dom_x_y)) {
            int expected = 1;
            for (DElemX const ix : DDomX(dom_x_y)) {
                expected *= 2;
                EXPECT_EQ(chunk_x_y_host(ix, iy), expected) << "at x=" << ix << ", y=" << iy;
            }
        }
    }
}
