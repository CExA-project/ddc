// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace {

struct DDimX;
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

static DElemX constexpr lbound_x(0);
static DVectX constexpr nelems_x(10);

} // namespace

void test_nested_algorithms_for_each()
{
    DDomX ddom(lbound_x, nelems_x);
    ddc::Chunk chk(ddom, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan chks = chk.span_view();
    ddc::parallel_for_each(
            ddc::DiscreteDomain<>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<>) {
                ddc::for_each(chks.domain(), [&](DElemX elem) { chks(elem) = 10; });
            });
    int res = ddc::parallel_transform_reduce(ddom, 0, ddc::reducer::sum<int>(), chks);
    EXPECT_EQ(res, 10 * ddom.size());
}

TEST(NestedAlgorithms, ForEach)
{
    test_nested_algorithms_for_each();
}

void test_nested_algorithms_transform_reduce()
{
    DDomX ddom(lbound_x, nelems_x);
    ddc::Chunk chk(ddom, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan chks = chk.span_view();
    ddc::parallel_for_each(
            ddom,
            KOKKOS_LAMBDA(DElemX elem) {
                chks(elem) = ddc::transform_reduce(
                        DDomX(lbound_x, DVectX(10)),
                        0,
                        ddc::reducer::sum<int>(),
                        [&](DElemX) { return 1; });
            });
    int res = ddc::parallel_transform_reduce(ddom, 0, ddc::reducer::sum<int>(), chks);
    EXPECT_EQ(res, 10 * ddom.size());
}

TEST(NestedAlgorithms, TransformReduce)
{
    test_nested_algorithms_transform_reduce();
}

void test_nested_algorithms_fill()
{
    DDomX ddom(lbound_x, nelems_x);
    ddc::Chunk chk(ddom, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan chks = chk.span_view();
    ddc::parallel_for_each(
            ddc::DiscreteDomain<>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<>) { ddc::fill(chks, 10); });
    int res = ddc::parallel_transform_reduce(ddom, 0, ddc::reducer::sum<int>(), chks);
    EXPECT_EQ(res, 10 * ddom.size());
}

TEST(NestedAlgorithms, Fill)
{
    test_nested_algorithms_fill();
}

void test_nested_algorithms_deepcopy()
{
    DDomX ddom(lbound_x, nelems_x);
    ddc::Chunk chk(ddom, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan chks = chk.span_view();
    ddc::parallel_fill(Kokkos::DefaultExecutionSpace(), chks, 10);
    ddc::Chunk chk2(ddom, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan chk2s = chk2.span_view();
    ddc::parallel_for_each(
            ddc::DiscreteDomain<>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<>) { ddc::deepcopy(chk2s, chks); });
    int res = ddc::parallel_transform_reduce(ddom, 0, ddc::reducer::sum<int>(), chk2s);
    EXPECT_EQ(res, 10 * ddom.size());
}

TEST(NestedAlgorithms, Deepcopy)
{
    test_nested_algorithms_deepcopy();
}
