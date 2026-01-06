// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

inline namespace anonymous_namespace_workaround_parallel_fill_cpp {

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

} // namespace anonymous_namespace_workaround_parallel_fill_cpp

TEST(ParallelFill, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_fill(view, 1);
    EXPECT_EQ(Kokkos::Experimental::count(Kokkos::DefaultExecutionSpace(), storage, 1), dom.size());
}

TEST(ParallelFill, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_fill(view, 1);
    EXPECT_EQ(Kokkos::Experimental::count(Kokkos::DefaultExecutionSpace(), storage, 1), dom.size());
}

TEST(ParallelFill, OneDimensionWithExecutionSpace)
{
    Kokkos::DefaultExecutionSpace const exec_space;
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*> const storage(Kokkos::view_alloc("storage", exec_space), dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_fill(exec_space, view, 1);
    EXPECT_EQ(Kokkos::Experimental::count(exec_space, storage, 1), dom.size());
}

TEST(ParallelFill, TwoDimensionsWithExecutionSpace)
{
    Kokkos::DefaultExecutionSpace const exec_space;
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*> const storage(Kokkos::view_alloc("storage", exec_space), dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_fill(exec_space, view, 1);
    EXPECT_EQ(Kokkos::Experimental::count(exec_space, storage, 1), dom.size());
}
