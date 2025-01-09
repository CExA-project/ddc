// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(CHUNK_SPAN_CPP) {

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

template <class Datatype>
using ChunkX = ddc::Chunk<Datatype, DDomX>;

template <class Datatype>
using ChunkSpanX = ddc::ChunkSpan<Datatype, DDomX>;

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(CHUNK_SPAN_CPP)

TEST(ChunkSpan1DTest, ConstructionFromChunk)
{
    EXPECT_FALSE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double>>));
    EXPECT_FALSE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double> const>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double>&>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<const double>, ChunkX<double>&>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<const double>, ChunkX<double> const&>));
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
                 ddc::ChunkSpan<const int, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));
}

TEST(ChunkSpan1DTest, CtadFromKokkosView)
{
    using ViewType = Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::HostSpace>;
    EXPECT_TRUE((std::is_same_v<
                 decltype(ddc::ChunkSpan(std::declval<ViewType>(), std::declval<DDomX>())),
                 ddc::ChunkSpan<int, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));

    using ConstViewType = Kokkos::View<const int*, Kokkos::LayoutRight, Kokkos::HostSpace>;
    EXPECT_TRUE((std::is_same_v<
                 decltype(ddc::ChunkSpan(std::declval<ConstViewType>(), std::declval<DDomX>())),
                 ddc::ChunkSpan<const int, DDomX, Kokkos::layout_right, Kokkos::HostSpace>>));
}

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(CHUNK_SPAN_CPP) {

void TestChunkSpan1DTestCtadOnDevice()
{
    Kokkos::View<int*, Kokkos::LayoutRight> const view("view", 3);
    Kokkos::deep_copy(view, 1);
    ddc::DiscreteElement<DDimX> const ix(0);
    ddc::DiscreteDomain<DDimX> const ddom_x(ix, ddc::DiscreteVector<DDimX>(view.extent(0)));
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

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(CHUNK_SPAN_CPP)

TEST(ChunkSpan1DTest, CtadOnDevice)
{
    TestChunkSpan1DTestCtadOnDevice();
}

TEST(ChunkSpan2DTest, CtorContiguousLayoutRightKokkosView)
{
    Kokkos::View<int**, Kokkos::LayoutRight> const view("view", 133, 189);
    ddc::DiscreteDomain<DDimX, DDimY> const
            ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                    ddc::DiscreteVector<DDimX, DDimY>(view.extent(0), view.extent(1)));
    EXPECT_NO_FATAL_FAILURE(ddc::ChunkSpan(view, ddom_xy));
}

TEST(ChunkSpan2DTest, CtorNonContiguousLayoutRightKokkosView)
{
    Kokkos::View<int**, Kokkos::LayoutRight> const
            view(Kokkos::view_alloc("view", Kokkos::AllowPadding), 133, 189);
    if (!view.span_is_contiguous()) {
        ddc::DiscreteDomain<DDimX, DDimY> const
                ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
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
    ddc::DiscreteDomain<DDimX, DDimY> const
            ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(0, 0),
                    ddc::DiscreteVector<DDimX, DDimY>(subview.extent(0), subview.extent(1)));
    ASSERT_TRUE((std::is_same_v<decltype(subview)::array_layout, Kokkos::LayoutStride>));
    EXPECT_NO_FATAL_FAILURE(ddc::ChunkSpan(subview, ddom_xy));
}
