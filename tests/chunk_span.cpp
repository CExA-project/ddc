// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_chunk_span_cpp {

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

template <class Datatype>
using ChunkX = ddc::Chunk<Datatype, DDomX>;

template <class Datatype>
using ChunkSpanX = ddc::ChunkSpan<Datatype, DDomX>;

} // namespace anonymous_namespace_workaround_chunk_span_cpp

TEST(ChunkSpan1DTest, ConstructionFromChunk)
{
    EXPECT_FALSE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double>>));
    EXPECT_FALSE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double> const>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double>&>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<double const>, ChunkX<double>&>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<double const>, ChunkX<double> const&>));
}

TEST(ChunkSpan1DTest, CtadFromChunk)
{
    using LvalueRefChunkType = ChunkX<int>&;
    EXPECT_TRUE((std::is_same_v<
                 decltype(ddc::ChunkSpan(std::declval<LvalueRefChunkType>())),
                 ddc::ChunkSpan<int, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));

    using ConstLvalueRefChunkType = ChunkX<int> const&;
    EXPECT_TRUE((std::is_same_v<
                 decltype(ddc::ChunkSpan(std::declval<ConstLvalueRefChunkType>())),
                 ddc::ChunkSpan<int const, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));
}

TEST(ChunkSpan1DTest, CtadFromKokkosView)
{
    using ViewType = Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace>;
    EXPECT_TRUE((std::is_same_v<
                 decltype(ddc::ChunkSpan(std::declval<ViewType>(), std::declval<DDomX>())),
                 ddc::ChunkSpan<int, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));

    using ConstViewType = Kokkos::View<int const*, Kokkos::LayoutRight, Kokkos::HostSpace>;
    EXPECT_TRUE((std::is_same_v<
                 decltype(ddc::ChunkSpan(std::declval<ConstViewType>(), std::declval<DDomX>())),
                 ddc::ChunkSpan<int const, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));
}

inline namespace anonymous_namespace_workaround_chunk_span_cpp {

void TestChunkSpan1DTestCtadOnDevice()
{
    Kokkos::View<int*, Kokkos::LayoutRight> const view("view", 3);
    Kokkos::deep_copy(view, 1);
    ddc::DiscreteDomain<DDimX> const ddom_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(view.extent(0)));
    ddc::DiscreteElement<DDimX> const ix = ddom_x.front();
    int sum;
    Kokkos::parallel_reduce(
            view.extent(0),
            KOKKOS_LAMBDA(int i, int& local_sum) {
                ddc::ChunkSpan const chk_span(view, ddom_x);
                local_sum += chk_span(ix + i);
            },
            sum);
    EXPECT_EQ(sum, view.size());
}

// Aliases to clarify tensor notations
using DDimMu = DDimX;
using DDimNu = DDimY;
using DDomMuNu = DDomXY;
using DVectMuNu = DVectXY;
using DElemMuNu = DElemXY;

void TestChunkSpan2DTestCtorStaticStorageFromLayoutRightExtents()
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using chunk_type = ddc::ChunkSpan<double, DDomMuNu, Kokkos::layout_right, memory_space, 4>;

    Kokkos::View<double*, memory_space> sum_d("sum_d", 1);
    Kokkos::deep_copy(sum_d, 0.0);

    ddc::parallel_for_each(
            execution_space(),
            ddc::DiscreteDomain<>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<>) {
                chunk_type chunk(2, 2);

                chunk(DElemMuNu(0, 0)) = 1.0;
                chunk(DElemMuNu(0, 1)) = 2.0;
                chunk(DElemMuNu(1, 0)) = 3.0;
                chunk(DElemMuNu(1, 1)) = 4.0;

                sum_d(0) += chunk(DElemMuNu(0, 0));
                sum_d(0) += chunk(DElemMuNu(0, 1));
                sum_d(0) += chunk(DElemMuNu(1, 0));
                sum_d(0) += chunk(DElemMuNu(1, 1));
            });

    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace::memory_space> const sum_h
            = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), sum_d);
    EXPECT_EQ(sum_h(0), 10.0);
}

void TestChunkSpan2DTestCtorStaticStorageFromLayoutStrideMapping()
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using chunk_type = ddc::ChunkSpan<double, DDomMuNu, Kokkos::layout_stride, memory_space, 6>;

    Kokkos::View<double*, memory_space> sum_d("sum_d", 1);
    Kokkos::deep_copy(sum_d, 0.0);

    DElemX const delem_mu = ddc::init_trivial_half_bounded_space<DDimMu>();
    DElemY const delem_nu = ddc::init_trivial_half_bounded_space<DDimNu>();
    DDomXY const domain_munu(DElemMuNu(delem_mu, delem_nu), DVectMuNu(2, 2));

    ddc::parallel_for_each(
            execution_space(),
            ddc::DiscreteDomain<>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<>) {
                typename chunk_type::extents_type const extents(2, 2);
                typename chunk_type::mapping_type const
                        layout_mapping(extents, std::array<std::size_t, 2> {3, 1});
                chunk_type chunk(layout_mapping, domain_munu);

                chunk(DElemMuNu(0, 0)) = 1.0;
                chunk(DElemMuNu(0, 1)) = 2.0;
                chunk(DElemMuNu(1, 0)) = 3.0;
                chunk(DElemMuNu(1, 1)) = 4.0;

                sum_d(0) += chunk(DElemMuNu(0, 0));
                sum_d(0) += chunk(DElemMuNu(0, 1));
                sum_d(0) += chunk(DElemMuNu(1, 0));
                sum_d(0) += chunk(DElemMuNu(1, 1));
            });

    Kokkos::View<double*, Kokkos::DefaultHostExecutionSpace::memory_space> const sum_h
            = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), sum_d);
    EXPECT_EQ(sum_h(0), 10.0);
}

} // namespace anonymous_namespace_workaround_chunk_span_cpp

TEST(ChunkSpan1DTest, CtadOnDevice)
{
    TestChunkSpan1DTestCtadOnDevice();
}

TEST(ChunkSpan2DTest, CtorContiguousLayoutRightKokkosView)
{
    Kokkos::View<int**, Kokkos::LayoutRight> const view("view", 133, 189);
    ddc::DiscreteElement<DDimX> const delem_x = ddc::init_trivial_half_bounded_space<DDimX>();
    ddc::DiscreteElement<DDimY> const delem_y = ddc::init_trivial_half_bounded_space<DDimY>();
    ddc::DiscreteDomain<DDimX, DDimY> const
            ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(delem_x, delem_y),
                    ddc::DiscreteVector<DDimX, DDimY>(view.extent(0), view.extent(1)));
    EXPECT_NO_FATAL_FAILURE(ddc::ChunkSpan(view, ddom_xy));
}

TEST(ChunkSpan2DTest, CtorNonContiguousLayoutRightKokkosView)
{
    Kokkos::View<int**, Kokkos::LayoutRight> const
            view(Kokkos::view_alloc("view", Kokkos::AllowPadding), 133, 189);
    if (!view.span_is_contiguous()) {
        ddc::DiscreteElement<DDimX> const delem_x = ddc::init_trivial_half_bounded_space<DDimX>();
        ddc::DiscreteElement<DDimY> const delem_y = ddc::init_trivial_half_bounded_space<DDimY>();
        ddc::DiscreteDomain<DDimX, DDimY> const
                ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(delem_x, delem_y),
                        ddc::DiscreteVector<DDimX, DDimY>(view.extent(0), view.extent(1)));
        EXPECT_DEBUG_DEATH(ddc::ChunkSpan(view, ddom_xy), ".*is_kokkos_layout_compatible.*");
    } else {
        GTEST_SKIP() << "The view does not use padding";
    }
}

TEST(ChunkSpan2DTest, CtorLayoutStrideKokkosView)
{
    Kokkos::View<int***, Kokkos::LayoutRight> const view("view", 3, 4, 5);
    Kokkos::View<int**, Kokkos::LayoutStride> const subview
            = Kokkos::subview(view, Kokkos::ALL, Kokkos::ALL, 3);
    ddc::DiscreteElement<DDimX> const delem_x = ddc::init_trivial_half_bounded_space<DDimX>();
    ddc::DiscreteElement<DDimY> const delem_y = ddc::init_trivial_half_bounded_space<DDimY>();
    ddc::DiscreteDomain<DDimX, DDimY> const
            ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(delem_x, delem_y),
                    ddc::DiscreteVector<DDimX, DDimY>(subview.extent(0), subview.extent(1)));
    ASSERT_TRUE((std::is_same_v<decltype(subview)::array_layout, Kokkos::LayoutStride>));
    EXPECT_NO_FATAL_FAILURE(ddc::ChunkSpan(subview, ddom_xy));
}

TEST(ChunkSpan2DTest, CtorStaticStorageFromLayoutRightExtents)
{
    TestChunkSpan2DTestCtorStaticStorageFromLayoutRightExtents();
}

TEST(ChunkSpan2DTest, CtorStaticStorageFromLayoutStrideMapping)
{
    TestChunkSpan2DTestCtorStaticStorageFromLayoutStrideMapping();
}
