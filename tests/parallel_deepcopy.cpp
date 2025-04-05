// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_parallel_deepcopy_cpp {

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
DVectX constexpr nelems_x(2);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(2);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);

} // namespace anonymous_namespace_workaround_parallel_deepcopy_cpp

TEST(ParallelDeepcopy, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    ddc::Chunk chk(dom, ddc::HostAllocator<int>());
    chk(dom.front() + DVectXY(0, 0)) = 1;
    chk(dom.front() + DVectXY(1, 0)) = 2;
    chk(dom.front() + DVectXY(0, 1)) = 3;
    chk(dom.front() + DVectXY(1, 1)) = 4;
    ddc::Chunk chk_copy(dom, ddc::HostAllocator<int>());
    ddc::parallel_deepcopy(chk_copy, chk);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(0, 0)), 1);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(1, 0)), 2);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(0, 1)), 3);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(1, 1)), 4);
}

TEST(ParallelDeepcopy, TwoDimensionsWithExecutionSpace)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    ddc::Chunk chk(dom, ddc::HostAllocator<int>());
    chk(dom.front() + DVectXY(0, 0)) = 1;
    chk(dom.front() + DVectXY(1, 0)) = 2;
    chk(dom.front() + DVectXY(0, 1)) = 3;
    chk(dom.front() + DVectXY(1, 1)) = 4;
    ddc::Chunk chk_copy(dom, ddc::HostAllocator<int>());
    Kokkos::DefaultHostExecutionSpace const exec_space;
    ddc::parallel_deepcopy(exec_space, chk_copy, chk);
    exec_space.fence();
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(0, 0)), 1);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(1, 0)), 2);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(0, 1)), 3);
    EXPECT_EQ(chk_copy(dom.front() + DVectXY(1, 1)), 4);
}
