// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include <experimental/mdspan>

#include "ddc/chunk.hpp"

#include "kokkos_allocator.hpp"

namespace ddc {

template <class ExecSpace, class ElementType, class SupportType, class Allocator>
Chunk<ElementType, SupportType, KokkosAllocator<ElementType, typename ExecSpace::memory_space>>
create_mirror(ExecSpace exec_space, Chunk<ElementType, SupportType, Allocator>& chunk)
{
    auto kokkos_view = chunk.allocation_kokkos_view();
    auto mirror_kokkos_view = Kokkos::create_mirror(exec_space, kokkos_view);
	/*
    auto mirror_chunkspan = ChunkSpan<
            ElementType,
            SupportType,
            std::experimental::layout_right,
            typename ExecSpace::memory_space>(mirror_kokkos_view, chunk.domain());
    auto mirror_chunk
            = Chunk<ElementType,
                    SupportType,
                    KokkosAllocator<ElementType, typename ExecSpace::memory_space>>(
                    mirror_chunkspan, KokkosAllocator<ElementType, typename ExecSpace::memory_space>()); // Not ok because deepcopy
	*/
	auto mirror_mdspan = detail::build_mdspan(mirror_kokkos_view, std::make_index_sequence<Chunk<ElementType, SupportType, Allocator>::rank()> {});
	Chunk<ElementType,
                    SupportType,
                    KokkosAllocator<ElementType, typename ExecSpace::memory_space>> mirror_chunk(mirror_mdspan, chunk.domain(), KokkosAllocator<ElementType, typename ExecSpace::memory_space>());
    return std::move(mirror_chunk);
}

template <class ExecSpace, class ElementType, class SupportType, class Layout, class MemorySpace>
//std::pair<auto,ChunkSpan<ElementType, SupportType, Layout, typename ExecSpace::memory_space>>
auto create_mirror(ExecSpace exec_space, const ChunkSpan<ElementType, SupportType, Layout, MemorySpace>& chunkspan)
{
    auto kokkos_view = chunkspan.allocation_kokkos_view();
    auto mirror_kokkos_view = Kokkos::create_mirror(exec_space, kokkos_view);
    auto mirror_chunkspan = ChunkSpan<
            ElementType,
            SupportType,
            Layout,
            typename ExecSpace::memory_space>(mirror_kokkos_view, chunkspan.domain());
    return std::pair(std::move(mirror_kokkos_view),std::move(mirror_chunkspan));
}
} // namespace ddc
