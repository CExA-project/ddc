// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_create_mirror_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(3);
DDomX constexpr dom_x(lbound_x, nelems_x);

template <class ElementType, class Support, class Layout, class MemorySpace, class T>
[[nodiscard]] bool all_equal_to(
        ddc::ChunkSpan<ElementType, Support, Layout, MemorySpace> const& chunk_span,
        T const& value)
{
    return ddc::parallel_transform_reduce(
            "all_equal_to",
            typename MemorySpace::execution_space(),
            chunk_span.domain(),
            true,
            ddc::reducer::land<bool>(),
            KOKKOS_LAMBDA(DElemX elem_x) { return chunk_span(elem_x) == value; });
}

} // namespace anonymous_namespace_workaround_create_mirror_cpp

TEST(CreateMirror, Host)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror(chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
}

TEST(CreateMirror, Device)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror(chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
}

TEST(CreateMirrorWithExecutionSpace, HostToDevice)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror(Kokkos::DefaultExecutionSpace(), chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<
                 decltype(mirror)::memory_space,
                 Kokkos::DefaultExecutionSpace::memory_space>));
}

TEST(CreateMirrorAndCopy, Host)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_and_copy(chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
    EXPECT_TRUE(all_equal_to(mirror.span_cview(), 3));
}

TEST(CreateMirrorAndCopy, Device)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_and_copy(chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
    EXPECT_TRUE(all_equal_to(mirror.span_cview(), 3));
}

TEST(CreateMirrorAndCopyWithExecutionSpace, HostToDevice)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_and_copy(Kokkos::DefaultExecutionSpace(), chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<
                 decltype(mirror)::memory_space,
                 Kokkos::DefaultExecutionSpace::memory_space>));
    EXPECT_TRUE(all_equal_to(mirror.span_cview(), 3));
}

TEST(CreateMirrorView, Host)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_view(chunk.span_view());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_EQ(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
}

TEST(CreateMirrorView, Device)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_view(chunk.span_view());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    if (Kokkos::SpaceAccessibility<Kokkos::HostSpace, decltype(chunk)::memory_space>::accessible) {
        EXPECT_EQ(chunk.data_handle(), mirror.data_handle());
    } else {
        EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    }
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
}

TEST(CreateMirrorViewWithExecutionSpace, HostToDevice)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_view(Kokkos::DefaultExecutionSpace(), chunk.span_view());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    if (Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, decltype(chunk)::memory_space>::
                accessible) {
        EXPECT_EQ(chunk.data_handle(), mirror.data_handle());
    } else {
        EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    }
    EXPECT_TRUE((std::is_same_v<
                 decltype(mirror)::memory_space,
                 Kokkos::DefaultExecutionSpace::memory_space>));
}

TEST(CreateMirrorViewAndCopy, Host)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_view_and_copy(chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    EXPECT_EQ(chunk.data_handle(), mirror.data_handle());
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
    EXPECT_TRUE(all_equal_to(mirror.span_cview(), 3));
}

TEST(CreateMirrorViewAndCopy, Device)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::DeviceAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror = ddc::create_mirror_view_and_copy(chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    if (Kokkos::SpaceAccessibility<Kokkos::HostSpace, decltype(chunk)::memory_space>::accessible) {
        EXPECT_EQ(chunk.data_handle(), mirror.data_handle());
    } else {
        EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    }
    EXPECT_TRUE((std::is_same_v<decltype(mirror)::memory_space, Kokkos::HostSpace>));
    EXPECT_TRUE(all_equal_to(mirror.span_cview(), 3));
}

TEST(CreateMirrorViewAndCopyWithExecutionSpace, HostToDevice)
{
    ddc::Chunk chunk("chunk", dom_x, ddc::HostAllocator<int>());
    ddc::parallel_fill(chunk, 3);
    auto mirror
            = ddc::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), chunk.span_cview());
    EXPECT_EQ(chunk.domain(), mirror.domain());
    if (Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, decltype(chunk)::memory_space>::
                accessible) {
        EXPECT_EQ(chunk.data_handle(), mirror.data_handle());
    } else {
        EXPECT_NE(chunk.data_handle(), mirror.data_handle());
    }
    EXPECT_TRUE((std::is_same_v<
                 decltype(mirror)::memory_space,
                 Kokkos::DefaultExecutionSpace::memory_space>));
    EXPECT_TRUE(all_equal_to(mirror.span_cview(), 3));
}
