// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <ostream>
#include <vector>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_StdAlgorithms.hpp>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(FOR_EACH_CPP) {

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

DElemX constexpr lbound_x(0);
DVectX constexpr nelems_x(10);

DElemY constexpr lbound_y(0);
DVectY constexpr nelems_y(12);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(FOR_EACH_CPP)

TEST(ForEachSerialHost, Empty)
{
    DDomX const dom(lbound_x, DVectX(0));
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> const view(storage.data(), dom);
    ddc::for_each(dom, [=](DElemX const ix) { view(ix) += 1; });
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 1), dom.size())
            << std::count(storage.begin(), storage.end(), 1) << "\n";
}

TEST(ForEachSerialHost, ZeroDimension)
{
    DDom0D const dom;
    int storage = 0;
    ddc::ChunkSpan<int, DDom0D> const view(&storage, dom);
    ddc::for_each(dom, [=](DElem0D const ii) { view(ii) += 1; });
    EXPECT_EQ(storage, 1) << storage << "\n";
}

TEST(ForEachSerialHost, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> const view(storage.data(), dom);
    ddc::for_each(dom, [=](DElemX const ix) { view(ix) += 1; });
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 1), dom.size());
}

TEST(ForEachSerialHost, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomXY> const view(storage.data(), dom);
    ddc::for_each(dom, [=](DElemXY const ixy) { view(ixy) += 1; });
    EXPECT_EQ(std::count(storage.begin(), storage.end(), 1), dom.size());
}

void TestForEachSerialDevice(ddc::ChunkSpan<
                             int,
                             DDomXY,
                             Kokkos::layout_right,
                             typename Kokkos::DefaultExecutionSpace::memory_space> view)
{
    ddc::parallel_for_each(
            Kokkos::DefaultExecutionSpace(),
            DDom0D(),
            KOKKOS_LAMBDA([[maybe_unused]] DElem0D unused_elem) {
                ddc::for_each(view.domain(), [=](DElemXY const ixy) { view(ixy) = 1; });
            });
}

TEST(ForEachSerialDevice, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> const
            storage("", dom.size());
    ddc::ChunkSpan<
            int,
            DDomXY,
            Kokkos::layout_right,
            typename Kokkos::DefaultExecutionSpace::memory_space> const view(storage.data(), dom);
    TestForEachSerialDevice(view);
    EXPECT_EQ(
            Kokkos::Experimental::
                    count(Kokkos::DefaultExecutionSpace(),
                          Kokkos::Experimental::begin(storage),
                          Kokkos::Experimental::end(storage),
                          1),
            dom.size());
}
