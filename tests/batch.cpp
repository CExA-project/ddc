// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_mean_cpp {

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


DElem0D constexpr lbound_0d {};
DVect0D constexpr nelems_0d {};
DDom0D constexpr dom_0d(lbound_0d, nelems_0d);

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(3);
DDomX constexpr dom_x(lbound_x, nelems_x);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(12);

DElemZ constexpr lbound_z = ddc::init_trivial_half_bounded_space<DDimZ>();

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);
DDomXY constexpr dom_x_y(lbound_x_y, nelems_x_y);

} // namespace anonymous_namespace_workaround_mean_cpp

template <
        typename Tin,
        typename Tout,
        typename DDomIn,
        typename DDomOut,
        typename ExecSpace,
        typename MemorySpace,
        typename LayoutIn,
        typename LayoutOut>
void sum(
        ExecSpace const& exec_space,
        ddc::ChunkSpan<Tout, DDomOut, LayoutOut, MemorySpace> const& out,
        ddc::ChunkSpan<Tin const, DDomIn, LayoutIn, MemorySpace> const& in)
{
    assert(out.domain() == DDomOut(in.domain()));
    ddc::host_for_each(out.domain(), [&](typename DDomOut::discrete_element_type iout) {
        ddc::ChunkSpan const a = in[iout];
        out(iout) = ddc::
                parallel_transform_reduce(exec_space, a.domain(), 0, ddc::reducer::sum<Tin>(), a);
    });
}

TEST(Algorithms, Sum)
{
    ChunkXY<int> chunk(dom_x_y);
    ddc::parallel_fill(chunk, 1);

    ChunkX<int> chunk2(dom_x);
    ddc::parallel_fill(chunk2, 0);

    sum(Kokkos::DefaultExecutionSpace(), chunk2.span_view(), chunk.span_cview());

    EXPECT_EQ(chunk2.domain(), dom_x);
    for (DElemX const ix : chunk2.domain()) {
        EXPECT_EQ(chunk2(ix), nelems_y.value());
    }
    ddc::print(std::cout, chunk2.span_cview());
    std::cout << '\n';
}

namespace ddc {

template <
        typename Tin,
        typename Tout,
        typename DDomIn,
        typename DDomOut,
        typename ExecSpace,
        typename MemorySpace,
        typename LayoutIn,
        typename LayoutOut>
void parallel_broadcast(
        ExecSpace const& exec_space,
        ddc::ChunkSpan<Tout, DDomOut, LayoutOut, MemorySpace> const& out,
        ddc::ChunkSpan<Tin const, DDomIn, LayoutIn, MemorySpace> const& in)
{
    assert(DDomIn(out.domain()) == in.domain());
    auto ddom_batch = ddc::remove_dims_of(out.domain(), in.domain());
    using DDomBatch = decltype(ddom_batch);
    ddc::parallel_for_each(
            exec_space,
            in.domain(),
            KOKKOS_LAMBDA(typename DDomIn::discrete_element_type iin) {
                Tin const val = in(iin);
                auto slice_out = out[iin];
                ddc::device_for_each(ddom_batch, [&](typename DDomBatch::discrete_element_type ib) {
                    slice_out(ib) = val;
                });
            });
    // ddc::parallel_for_each(
    //         exec_space,
    //         in.domain(),
    //         KOKKOS_LAMBDA(typename DDomIn::discrete_element_type iin) {
    //             ddc::deepcopy(out[iin], in[iin]);
    //         });
    ddc::parallel_for_each(
            exec_space,
            out.domain(),
            KOKKOS_LAMBDA(typename DDomOut::discrete_element_type iout) {
                out(iout) = in(typename DDomIn::discrete_element_type(iout));
            });
    ddc::host_for_each(ddom_batch, [=](typename DDomBatch::discrete_element_type ib) {
        ddc::parallel_deepcopy(out[ib], in);
    });
}

} // namespace ddc

TEST(Algorithms, Broadcast0dTo2d)
{
    ChunkXY<int> chunk(dom_x_y);
    ddc::parallel_fill(chunk, 0);

    Chunk0D<int> chunk2(dom_0d);
    ddc::parallel_fill(chunk2, 1);

    ddc::parallel_broadcast(
            Kokkos::DefaultExecutionSpace(),
            chunk.span_view(),
            chunk2.span_cview());

    ddc::print(std::cout, chunk.span_cview());
    std::cout << '\n';
}

TEST(Algorithms, Broadcast1dTo2d)
{
    ChunkXY<int> chunk(dom_x_y);
    ddc::parallel_fill(chunk, 0);

    ChunkX<int> chunk2(dom_x);
    ddc::parallel_fill(chunk2, 1);

    ddc::parallel_broadcast(
            Kokkos::DefaultExecutionSpace(),
            chunk.span_view(),
            chunk2.span_cview());

    ddc::print(std::cout, chunk.span_cview());
    std::cout << '\n';
}
