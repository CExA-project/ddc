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
		class Allocator = HostAllocator<ElementType>>
Chunk<ElementType, SupportType, KokkosAllocator<ElementType, typename ExecSpace::memory_space>> create_mirror(ExecSpace exec_space, Chunk<ElementType, SupportType, Allocator> chunk) {
	auto kokkos_view = chunk.allocation_kokkos_view();
    auto mirror_kokkos_view = Kokkos::create_mirror(exec_space, kokkos_view);
	return Chunk(detail::build_mdspan(mirror_kokkos_view, std::make_index_sequence<chunk.rank()> {}), chunk.domain());
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
