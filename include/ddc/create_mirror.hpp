// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "ddc/chunk_span.hpp"
#include "ddc/kokkos_allocator.hpp"

namespace ddc {

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space.
template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror(
        [[maybe_unused]] Space const& space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    static_assert(
            Kokkos::is_memory_space_v<Space> || Kokkos::is_execution_space_v<Space>,
            "DDC: parameter \"Space\" must be either a Kokkos execution space or a memory space");
    static_assert(
            std::is_same_v<Layout, std::experimental::layout_right>,
            "DDC: parameter \"Layout\" must be a `layout_right`");
    return Chunk(
            src.domain(),
            KokkosAllocator<std::remove_const_t<ElementType>, typename Space::memory_space>());
}

/// Equivalent to `create_mirror(Kokkos::HostSpace(), src)`.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space.
template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror(ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    return create_mirror(Kokkos::HostSpace(), src);
}

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space and operates a deep copy between the two.
template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_and_copy(
        Space const& space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    Chunk chunk = create_mirror(space, src);
    parallel_deepcopy(chunk, src);
    return chunk;
}

/// Equivalent to `create_mirror_and_copy(Kokkos::HostSpace(), src)`.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space and operates a deep copy between the two.
template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_and_copy(ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    return create_mirror_and_copy(Kokkos::HostSpace(), src);
}

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A non-const, layout right ChunkSpan.
/// @return If `MemorySpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space.
template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view(
        [[maybe_unused]] Space const& space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    static_assert(
            !std::is_const_v<ElementType>,
            "DDC: parameter \"ElementType\" must not be `const`");
    static_assert(
            Kokkos::is_memory_space_v<Space> || Kokkos::is_execution_space_v<Space>,
            "DDC: parameter \"Space\" must be either a Kokkos execution space or a memory space");
    static_assert(
            std::is_same_v<Layout, std::experimental::layout_right>,
            "DDC: parameter \"Layout\" must be a `layout_right`");
    if constexpr (Kokkos::SpaceAccessibility<Space, MemorySpace>::accessible) {
        return src;
    } else {
        return create_mirror(space, src);
    }
}

/// Equivalent to `create_mirror_view(Kokkos::HostSpace(), src)`.
/// @param[in] src A non-const, layout right ChunkSpan.
/// @return If `Kokkos::HostSpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space.
template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view(ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    create_mirror_view(Kokkos::HostSpace(), src);
}

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A layout right ChunkSpan.
/// @return If `MemorySpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space and operates a deep copy between the two.
template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view_and_copy(
        [[maybe_unused]] Space const& space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    static_assert(
            Kokkos::is_memory_space_v<Space> || Kokkos::is_execution_space_v<Space>,
            "DDC: parameter \"Space\" must be either a Kokkos execution space or a memory space");
    static_assert(
            std::is_same_v<Layout, std::experimental::layout_right>,
            "DDC: parameter \"Layout\" must be a `layout_right`");
    if constexpr (Kokkos::SpaceAccessibility<Space, MemorySpace>::accessible) {
        return src;
    } else {
        return create_mirror_and_copy(space, src);
    }
}

/// Equivalent to `create_mirror_view_and_copy(Kokkos::HostSpace(), src)`.
/// @param[in] src A layout right ChunkSpan.
/// @return If `Kokkos::HostSpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space and operates a deep copy between the two.
template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view_and_copy(ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    return create_mirror_view_and_copy(Kokkos::HostSpace(), src);
}

} // namespace ddc
