// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "ddc/kokkos_allocator.hpp"

inline namespace anonymous_namespace_workaround_parallel_copy_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

template <class Datatype>
using Chunk0D = ddc::Chunk<Datatype, DDom0D>;
template <class Datatype>
using ChunkSpan0D = ddc::ChunkSpan<Datatype, DDom0D>;


struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

template <class Datatype>
using ChunkX = ddc::Chunk<Datatype, DDomX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

template <class Datatype>
using ChunkY = ddc::Chunk<Datatype, DDomY>;


struct DDimZ
{
};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;
using DDomZ = ddc::DiscreteDomain<DDimZ>;


using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

template <class Datatype>
using ChunkXY = ddc::Chunk<Datatype, DDomXY>;


using DElemYX = ddc::DiscreteElement<DDimY, DDimX>;
using DVectYX = ddc::DiscreteVector<DDimY, DDimX>;
using DDomYX = ddc::DiscreteDomain<DDimY, DDimX>;

template <class Datatype>
using ChunkYX = ddc::Chunk<Datatype, DDomYX>;


using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;
using DDomXYZ = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;

DElem0D constexpr lbound_0d {};
DVect0D constexpr nelems_0d {};
DDom0D constexpr dom_0d(lbound_0d, nelems_0d);

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(3);
DDomX constexpr dom_x(lbound_x, nelems_x);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(12);

DElemZ constexpr lbound_z = ddc::init_trivial_half_bounded_space<DDimZ>();
DVectZ constexpr nelems_z(5);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);
DDomXY constexpr dom_x_y(lbound_x_y, nelems_x_y);

DElemXYZ constexpr lbound_x_y_z(lbound_x, lbound_y, lbound_z);
DVectXYZ constexpr nelems_x_y_z(nelems_x, nelems_y, nelems_z);
DDomXYZ constexpr dom_x_y_z(lbound_x_y_z, nelems_x_y_z);

} // namespace anonymous_namespace_workaround_parallel_copy_cpp

TEST(ParallelCopy, BroadcastScalar2XY)
{
    Kokkos::DefaultExecutionSpace const exec_space;

    Kokkos::View<int*> const storage(Kokkos::view_alloc("storage", exec_space), dom_x_y.size());
    ddc::ChunkSpan const
            chunk_x_y(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom_x_y);
    ddc::parallel_fill(exec_space, chunk_x_y, 0);

    ddc::Chunk chunk(dom_0d, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk, 1);

    ddc::parallel_copy(exec_space, chunk_x_y, chunk);

    EXPECT_EQ(Kokkos::Experimental::count(exec_space, storage, 1), dom_x_y.size());
}

TEST(ParallelCopy, BroadcastX2XY)
{
    Kokkos::DefaultExecutionSpace const exec_space;

    Kokkos::View<int*> const storage(Kokkos::view_alloc("storage", exec_space), dom_x_y.size());
    ddc::ChunkSpan const
            chunk_x_y(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom_x_y);
    ddc::parallel_fill(exec_space, chunk_x_y, 0);

    ddc::Chunk chunk_x(dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_x, 1);

    ddc::parallel_copy(exec_space, chunk_x_y, chunk_x);

    EXPECT_EQ(Kokkos::Experimental::count(exec_space, storage, 1), dom_x_y.size());
}

TEST(ParallelCopy, TransposeXY2XY)
{
    Kokkos::DefaultExecutionSpace const exec_space;

    Kokkos::View<int*> const storage(Kokkos::view_alloc("storage", exec_space), dom_x_y.size());
    ddc::ChunkSpan const
            chunk_x_y_out(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom_x_y);
    ddc::parallel_fill(exec_space, chunk_x_y_out, 0);

    ddc::Chunk chunk_x_y_in(dom_x_y, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_x_y_in, 1);

    ddc::parallel_copy(exec_space, chunk_x_y_out, chunk_x_y_in);

    EXPECT_EQ(Kokkos::Experimental::count(exec_space, storage, 1), dom_x_y.size());
}

TEST(ParallelCopy, BroadcastAndTransposeYX2XYZ)
{
    Kokkos::DefaultExecutionSpace const exec_space;

    Kokkos::View<int*> const storage(Kokkos::view_alloc("storage", exec_space), dom_x_y_z.size());
    ddc::ChunkSpan const chunk_x_y_z(
            Kokkos::View<int***>(storage.data(), nelems_x, nelems_y, nelems_z),
            dom_x_y_z);
    ddc::parallel_fill(exec_space, chunk_x_y_z, 0);

    DDomYX const dom_y_x(dom_x_y);
    ddc::Chunk chunk_y_x(dom_y_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(exec_space, chunk_y_x, 1);

    ddc::parallel_copy(exec_space, chunk_x_y_z, chunk_y_x);

    EXPECT_EQ(Kokkos::Experimental::count(exec_space, storage, 1), dom_x_y_z.size());
}
