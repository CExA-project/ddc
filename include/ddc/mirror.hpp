// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#include "chunk_span.hpp"
#include "kokkos_allocator.hpp"

namespace ddc {

template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror(Space space, ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    return Chunk<ElementType, Support, KokkosAllocator<ElementType, typename Space::memory_space>>(
            chunk_span.domain());
}

template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror(ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    return create_mirror(Kokkos::DefaultHostExecutionSpace(), chunk_span);
}

template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_and_copy(
        Space space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    Chunk<ElementType, Support, KokkosAllocator<ElementType, typename Space::memory_space>> chunk(
            chunk_span.domain());
    deepcopy(chunk, chunk_span);
    return chunk;
}

template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_and_copy(ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    return create_mirror_and_copy(Kokkos::DefaultHostExecutionSpace(), chunk_span);
}

template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view(
        Space space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    if constexpr (Kokkos::SpaceAccessibility<Space, MemorySpace>::accessible) {
        return chunk_span;
    } else {
        return create_mirror(space, chunk_span);
    }
}

template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view(ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    return create_mirror_view(Kokkos::DefaultHostExecutionSpace(), chunk_span);
}

template <class Space, class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view_and_copy(
        Space space,
        ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    if constexpr (Kokkos::SpaceAccessibility<Space, MemorySpace>::accessible) {
        return chunk_span;
    } else {
        return create_mirror_and_copy(space, chunk_span);
    }
}

template <class ElementType, class Support, class Layout, class MemorySpace>
auto create_mirror_view_and_copy(ChunkSpan<ElementType, Support, Layout, MemorySpace> chunk_span)
{
    return create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), chunk_span);
}

} // namespace ddc
