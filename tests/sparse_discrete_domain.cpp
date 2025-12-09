// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

inline namespace anonymous_namespace_workaround_sparse_discrete_domain_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::SparseDiscreteDomain<DDimX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::SparseDiscreteDomain<DDimY>;


using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::SparseDiscreteDomain<DDimX, DDimY>;


using DElemYX = ddc::DiscreteElement<DDimY, DDimX>;
using DVectYX = ddc::DiscreteVector<DDimY, DDimX>;
using DDomYX = ddc::SparseDiscreteDomain<DDimY, DDimX>;


DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();

} // namespace anonymous_namespace_workaround_sparse_discrete_domain_cpp

TEST(SparseDiscreteDomainTest, Constructor)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;
    DDomXY const dom_xy(view_x, view_y);
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 0, lbound_y + 0), DVectXY(0, 0));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 0, lbound_y + 2), DVectXY(0, 1));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 0, lbound_y + 5), DVectXY(0, 2));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 2, lbound_y + 0), DVectXY(1, 0));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 2, lbound_y + 2), DVectXY(1, 1));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 2, lbound_y + 5), DVectXY(1, 2));
    EXPECT_FALSE(dom_xy.contains(lbound_x + 1, lbound_y + 0));
}

void TestAnnotatedForEachSparseDevice2D(ddc::ChunkSpan<
                                        int,
                                        DDomXY,
                                        Kokkos::layout_right,
                                        typename Kokkos::DefaultExecutionSpace::memory_space> view)
{
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            DDom0D(),
            KOKKOS_LAMBDA(DElem0D) {
                ddc::annotated_for_each(view.domain(), [=](DVectXY const ixy) {
                    view(ixy) = 1;
                    });
            });
}

TEST(AnnotatedForEachSparseDevice, TwoDimensions)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;

    DDomXY const dom(view_x, view_y);
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> const
            storage("", dom.size());
    ddc::ChunkSpan<
            int,
            DDomXY,
            Kokkos::layout_right,
            typename Kokkos::DefaultExecutionSpace::memory_space> const view(storage.data(), dom);

    TestAnnotatedForEachSparseDevice2D(view);
    EXPECT_EQ(
            Kokkos::Experimental::
                    count(Kokkos::DefaultExecutionSpace(),
                          Kokkos::Experimental::begin(storage),
                          Kokkos::Experimental::end(storage),
                          1),
            dom.size());
}

int TestAnnotatedTransformReduceSparse(
        ddc::ChunkSpan<
                int,
                DDomXY,
                Kokkos::layout_right,
                typename Kokkos::DefaultExecutionSpace::memory_space> chunk)
{
    Kokkos::View<int, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> const count("");
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            DDom0D(),
            KOKKOS_LAMBDA(DElem0D) {
                count() = ddc::annotated_transform_reduce(
                        chunk.domain(),
                        0,
                        ddc::reducer::sum<int>(),
                        chunk);
            });
    Kokkos::fence();
    auto const count_host = Kokkos::create_mirror_view(count);
    Kokkos::deep_copy(count_host, count);
    return count_host();
}

TEST(AnnotatedTransformReduceSparse, TwoDimensions)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;

    DDomXY const dom(view_x, view_y);
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> const
            storage("", dom.size());
    Kokkos::deep_copy(storage, 1);
    ddc::ChunkSpan<
            int,
            DDomXY,
            Kokkos::layout_right,
            typename Kokkos::DefaultExecutionSpace::memory_space> const chunk(storage.data(), dom);
    EXPECT_EQ(TestAnnotatedTransformReduceSparse(chunk), dom.size());
}
