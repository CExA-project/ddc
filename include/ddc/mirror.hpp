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

template <
		class ExecSpace,
        class ElementType,
        class SupportType,
		class Allocator>
// Chunk<ElementType, SupportType, KokkosAllocator<ElementType, typename ExecSpace::memory_space>>&& create_mirror(ExecSpace exec_space, Chunk<ElementType, SupportType, Allocator> chunk) {
ChunkSpan<ElementType, SupportType, std::experimental::layout_right, typename ExecSpace::memory_space> create_mirror(ExecSpace exec_space, Chunk<ElementType, SupportType, Allocator>& chunk) {
	auto kokkos_view = chunk.allocation_kokkos_view();
    auto mirror_kokkos_view = Kokkos::create_mirror(exec_space, kokkos_view);
	auto mirror_chunkspan = ChunkSpan<ElementType, SupportType, std::experimental::layout_right, typename ExecSpace::memory_space>(mirror_kokkos_view,chunk.domain());
	auto mirror_chunk = Chunk<ElementType, SupportType, KokkosAllocator<ElementType, typename ExecSpace::memory_space>>(mirror_chunkspan);
	// return std::move(mirror_chunk);
	return mirror_chunkspan;
}

/*
template <
		class ExecSpace,
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = std::experimental::layout_right,
        class MemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
Chunk<ElementType, SupportType, LayoutStridedPolicy, ExecSpace::memory_space> create_mirror(ExecSpace exec_space, Chunk<ElementType, SupportType,LayoutStridedPolicy, MemorySpace>) {
	
}
*/

} // namespace ddc
