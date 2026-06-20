// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"

#include "chunk_span.hpp"
#include "kokkos_allocator.hpp"

namespace ddc {

namespace detail {

/**
 * @brief Ensure a layout_right view of a ChunkSpan, copying if necessary.
 *
 * If the input `src` already uses `Kokkos::layout_right`, it is returned as-is.
 * Otherwise, a new chunk with layout_right is allocated and a deep copy of
 * `src` is performed into it.
 *
 * @param[in] src Source ChunkSpan to adapt.
 * @return Either the original view (if already LayoutRight) or a copied chunk.
 */
template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_layout_right_view_and_copy(
        ChunkSpan<ElementType, Support, Layout, MemorySpace> const& src)
{
    if constexpr (std::is_same_v<Layout, Kokkos::layout_right>) {
        return src;
    } else {
        Chunk chunk(src.domain(), KokkosAllocator<std::remove_const_t<ElementType>, MemorySpace>());
        parallel_deepcopy(chunk, src);
        return chunk;
    }
}

} // namespace detail

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space.
template <class Space, class ElementType, class Support, class MemorySpace>
    requires detail::memory_space<Space> || detail::execution_space<Space>
auto create_mirror(
        [[maybe_unused]] Space const& space,
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    // `space` is always unused, but needed for Doxygen
    return Chunk(
            src.domain(),
            KokkosAllocator<std::remove_const_t<ElementType>, typename Space::memory_space>());
}

/// Equivalent to `create_mirror(Kokkos::HostSpace(), src)`.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space.
template <class ElementType, class Support, class MemorySpace>
auto create_mirror(ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    return create_mirror(Kokkos::HostSpace(), src);
}

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space and operates a deep copy between the two.
template <class Space, class ElementType, class Support, class MemorySpace>
    requires detail::memory_space<Space> || detail::execution_space<Space>
auto create_mirror_and_copy(
        Space const& space,
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    Chunk chunk = create_mirror(space, src);
    parallel_deepcopy(chunk, src);
    return chunk;
}

/// Equivalent to `create_mirror_and_copy(Kokkos::HostSpace(), src)`.
/// @param[in] src A layout right ChunkSpan.
/// @return a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space and operates a deep copy between the two.
template <class ElementType, class Support, class MemorySpace>
auto create_mirror_and_copy(
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    return create_mirror_and_copy(Kokkos::HostSpace(), src);
}

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A non-const, layout right ChunkSpan.
/// @return If `MemorySpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space.
template <class Space, class ElementType, class Support, class MemorySpace>
    requires(detail::memory_space<Space> || detail::execution_space<Space>)
            && (!std::is_const_v<ElementType>)
auto create_mirror_view(
        [[maybe_unused]] Space const& space,
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    if constexpr (Kokkos::SpaceAccessibility<Space, MemorySpace>::accessible) {
        return src;
    } else {
        return create_mirror(space, src);
    }
}

/// Equivalent to `create_mirror_view(Kokkos::HostSpace(), src)`.
/// @param[in] src A non-const, layout right ChunkSpan.
/// @return If `Kokkos::HostSpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space.
template <class ElementType, class Support, class MemorySpace>
auto create_mirror_view(
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    return create_mirror_view(Kokkos::HostSpace(), src);
}

/// @param[in] space A Kokkos memory space or execution space.
/// @param[in] src A layout right ChunkSpan.
/// @return If `MemorySpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Space::memory_space` memory space and operates a deep copy between the two.
template <class Space, class ElementType, class Support, class MemorySpace>
    requires detail::memory_space<Space> || detail::execution_space<Space>
auto create_mirror_view_and_copy(
        [[maybe_unused]] Space const& space,
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    if constexpr (Kokkos::SpaceAccessibility<Space, MemorySpace>::accessible) {
        return src;
    } else {
        return create_mirror_and_copy(space, src);
    }
}

/// Equivalent to `create_mirror_view_and_copy(Kokkos::HostSpace(), src)`.
/// @param[in] src A layout right ChunkSpan.
/// @return If `Kokkos::HostSpace` is accessible from `Space` then returns a copy of `src`, otherwise returns a `Chunk` with the same support and layout as `src` allocated on the `Kokkos::HostSpace` memory space and operates a deep copy between the two.
template <class ElementType, class Support, class MemorySpace>
auto create_mirror_view_and_copy(
        ChunkSpan<ElementType, Support, Kokkos::layout_right, MemorySpace> const& src)
{
    return create_mirror_view_and_copy(Kokkos::HostSpace(), src);
}

} // namespace ddc
