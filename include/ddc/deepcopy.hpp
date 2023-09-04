// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "ddc/chunk_common.hpp"
#include "ddc/chunk_span.hpp"

namespace ddc {

/** Copy the content of a chunk span into another
 * @param[out] dst the chunk span in which to copy
 * @param[in]  src the chunk span from which to copy
 * @return dst as a ChunkSpan
*/
template <
        class ElementTypeDst,
        class SupportDst,
        class LayoutDst,
        class MemorySpaceDst,
        class ElementTypeSrc,
        class SupportSrc,
        class LayoutSrc,
        class MemorySpaceSrc>
KOKKOS_FUNCTION auto deepcopy(
        ChunkSpan<ElementTypeDst, SupportDst, LayoutDst, MemorySpaceDst> const& dst,
        ChunkSpan<ElementTypeSrc, SupportSrc, LayoutSrc, MemorySpaceSrc> const& src) noexcept
{
    using ChunkDst = ChunkSpan<ElementTypeDst, SupportDst, LayoutDst, MemorySpaceDst>;
    using ChunkSrc = ChunkSpan<ElementTypeSrc, SupportSrc, LayoutSrc, MemorySpaceSrc>;
    static_assert(
            std::is_assignable_v<chunk_reference_t<ChunkDst>, chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    KOKKOS_ASSERT(dst.domain().extents() == src.domain().extents());
    KOKKOS_ASSERT(
            (Kokkos::SpaceAccessibility<DDC_CURRENT_KOKKOS_SPACE, MemorySpaceSrc>::accessible));
    KOKKOS_ASSERT(
            (Kokkos::SpaceAccessibility<DDC_CURRENT_KOKKOS_SPACE, MemorySpaceDst>::accessible));
    for_each(dst.domain(), [&dst, &src](typename SupportDst::discrete_element_type const& elem) {
        dst(elem) = src(elem);
    });
    return dst;
}

} // namespace ddc
