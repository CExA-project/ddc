// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "helper_count.hpp"

inline namespace anonymous_namespace_workaround_sparse_discrete_domain_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

struct DDimX : ddc::DiscreteDimension
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::SparseDiscreteDomain<DDimX>;


struct DDimY : ddc::DiscreteDimension
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
    EXPECT_TRUE(dom_xy.contains(lbound_x + 0, lbound_y + 0));
    EXPECT_FALSE(dom_xy.contains(lbound_x + 1, lbound_y + 0));
    EXPECT_FALSE(dom_xy.contains(lbound_x + 0, lbound_y + 1));
    DDomXY const dom_xy_2(Kokkos::View<DElemX*, Kokkos::SharedSpace>("", 0), view_y);
    EXPECT_FALSE(dom_xy_2.contains(lbound_x, lbound_y));
}

TEST(SparseDiscreteDomainTest, CompareSameDomains)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;
    DDomXY const dom_x_y_1(view_x, view_y);
    EXPECT_TRUE(dom_x_y_1 == dom_x_y_1);
    EXPECT_TRUE(dom_x_y_1 == DDomYX(dom_x_y_1));
    EXPECT_FALSE(dom_x_y_1 != dom_x_y_1);
    EXPECT_FALSE(dom_x_y_1 != DDomYX(dom_x_y_1));
}

TEST(SparseDiscreteDomainTest, CompareDifferentDomains)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x_a("view_x", 2);
    view_x_a(0) = lbound_x + 0;
    view_x_a(1) = lbound_x + 2;
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x_b("view_x", 2);
    view_x_b(0) = lbound_x + 1;
    view_x_b(1) = lbound_x + 2;
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x_c("view_x", 1);
    view_x_b(0) = lbound_x + 1;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;
    DDomXY const dom_x_y_1(view_x_a, view_y);
    DDomXY const dom_x_y_2(view_x_b, view_y);
    DDomXY const dom_x_y_3(view_x_c, view_y);
    EXPECT_FALSE(dom_x_y_1 == dom_x_y_2);
    EXPECT_FALSE(dom_x_y_1 == DDomYX(dom_x_y_2));
    EXPECT_TRUE(dom_x_y_1 != dom_x_y_2);
    EXPECT_TRUE(dom_x_y_1 != DDomYX(dom_x_y_2));

    EXPECT_FALSE(dom_x_y_1 == dom_x_y_3);
    EXPECT_FALSE(dom_x_y_1 == DDomYX(dom_x_y_3));
    EXPECT_TRUE(dom_x_y_1 != dom_x_y_3);
    EXPECT_TRUE(dom_x_y_1 != DDomYX(dom_x_y_3));

    EXPECT_FALSE(dom_x_y_2 == dom_x_y_3);
    EXPECT_FALSE(dom_x_y_2 == DDomYX(dom_x_y_3));
    EXPECT_TRUE(dom_x_y_2 != dom_x_y_3);
    EXPECT_TRUE(dom_x_y_2 != DDomYX(dom_x_y_3));
}

TEST(SparseDiscreteDomainTest, CompareEmptyDomains)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;
    DDomXY const dom_x_y_1;
    DDomXY const dom_x_y_2(view_x, view_y);
    EXPECT_TRUE(dom_x_y_1.empty());
    EXPECT_FALSE(dom_x_y_2.empty());

    EXPECT_TRUE(dom_x_y_1 == dom_x_y_1);
    EXPECT_FALSE(dom_x_y_1 != dom_x_y_1);

    EXPECT_FALSE(dom_x_y_1 == dom_x_y_2);
    EXPECT_TRUE(dom_x_y_1 != dom_x_y_2);
}

TEST(SparseDiscreteDomainTest, Select)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;
    DDomX const dom_x(view_x);
    DDomY const dom_y(view_y);
    DDomXY const dom_x_y(dom_x, dom_y);
    EXPECT_EQ(ddc::select<DDimX>(dom_x_y), dom_x);
    EXPECT_EQ(ddc::select<DDimY>(dom_x_y), dom_y);
}

void TestDeviceForEachSparseDevice2D(
        ddc::ChunkSpan<
                int,
                DDomXY,
                Kokkos::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space> view)
{
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            DDom0D(),
            KOKKOS_LAMBDA(DElem0D) {
                ddc::device_for_each(view.domain(), [=](DElemXY const ixy) { view(ixy) = 1; });
            });
}

TEST(DeviceForEachSparseDevice, TwoDimensions)
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
            Kokkos::DefaultExecutionSpace::memory_space> const view(storage.data(), dom);

    TestDeviceForEachSparseDevice2D(view);
    EXPECT_EQ(ddc::testing::count(Kokkos::DefaultExecutionSpace(), storage, 1), dom.size());
}

int TestDeviceTransformReduceSparse(
        ddc::ChunkSpan<
                int,
                DDomXY,
                Kokkos::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space> chunk)
{
    Kokkos::View<int, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> const count("");
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            DDom0D(),
            KOKKOS_LAMBDA(DElem0D) {
                count() = ddc::
                        device_transform_reduce(chunk.domain(), 0, ddc::reducer::sum<int>(), chunk);
            });
    Kokkos::fence();
    auto const count_host = Kokkos::create_mirror_view(count);
    Kokkos::deep_copy(count_host, count);
    return count_host();
}

TEST(DeviceTransformReduceSparse, TwoDimensions)
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
            Kokkos::DefaultExecutionSpace::memory_space> const chunk(storage.data(), dom);
    EXPECT_EQ(TestDeviceTransformReduceSparse(chunk), dom.size());
}

TEST(SparseDiscreteDomainTest, Print)
{
    {
        std::stringstream ss;
        ss << ddc::SparseDiscreteDomain<>();
        EXPECT_EQ(ss.str(), "SparseDiscreteDomain(extents=(), elements=[])");
    }

    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 1;
    view_x(1) = lbound_x + 4;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 3;
    view_y(1) = lbound_y + 5;
    view_y(2) = lbound_y + 9;
    {
        std::stringstream ss;
        ss << DDomX(view_x);
        EXPECT_EQ(ss.str(), "SparseDiscreteDomain(extents=(2), elements=[[1, 4]])");
    }
    {
        std::stringstream ss;
        ss << DDomXY(view_x, view_y);
        EXPECT_EQ(ss.str(), "SparseDiscreteDomain(extents=(2, 3), elements=[[1, 4], [3, 5, 9]])");
    }
}

TEST(SparseDiscreteDomainTest, PrintElisionThreshold)
{
    // Up to 7 elements are printed in full, from 8 elements the middle ones are elided
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_7("view_7", 7);
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_8("view_8", 8);
    for (std::size_t i = 0; i < view_7.size(); ++i) {
        view_7(i) = lbound_x + 10 * i;
    }
    for (std::size_t i = 0; i < view_8.size(); ++i) {
        view_8(i) = lbound_x + 10 * i;
    }
    {
        std::stringstream ss;
        ss << DDomX(view_7);
        EXPECT_EQ(
                ss.str(),
                "SparseDiscreteDomain(extents=(7), elements=[[0, 10, 20, 30, 40, 50, 60]])");
    }
    {
        std::stringstream ss;
        ss << DDomX(view_8);
        EXPECT_EQ(
                ss.str(),
                "SparseDiscreteDomain(extents=(8), elements=[[0, 10, 20, ..., 50, 60, 70]])");
    }
}

TEST(SparseDiscreteDomainTest, PrintFrontAndBackOnly)
{
    // Even for a large domain, only the front and back elements are printed
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 10);
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 20);
    for (std::size_t i = 0; i < view_x.size(); ++i) {
        view_x(i) = lbound_x + 2 * i + 1;
    }
    for (std::size_t i = 0; i < view_y.size(); ++i) {
        view_y(i) = lbound_y + 3 * i + 2;
    }
    std::stringstream ss;
    ss << DDomXY(view_x, view_y);
    EXPECT_EQ(
            ss.str(),
            "SparseDiscreteDomain(extents=(10, 20), "
            "elements=[[1, 3, 5, ..., 15, 17, 19], [2, 5, 8, ..., 53, 56, 59]])");
}

TEST(SparseDiscreteDomainTest, PrintDimensionOrder)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 1;
    view_x(1) = lbound_x + 4;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 3;
    view_y(1) = lbound_y + 5;
    view_y(2) = lbound_y + 9;
    std::stringstream ss;
    ss << DDomYX(DDomXY(view_x, view_y));
    EXPECT_EQ(ss.str(), "SparseDiscreteDomain(extents=(3, 2), elements=[[3, 5, 9], [1, 4]])");
}

TEST(SparseDiscreteDomainTest, PrintEmpty)
{
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 3;
    view_y(1) = lbound_y + 5;
    view_y(2) = lbound_y + 9;
    std::stringstream ss;
    ss << DDomXY(Kokkos::View<DElemX*, Kokkos::SharedSpace>("", 0), view_y);
    EXPECT_EQ(ss.str(), "SparseDiscreteDomain(empty)");
}
