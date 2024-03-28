// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "ddc/chunk_span.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/for_each.hpp"

namespace ddc {

/** Fill a chunk span with a given value
 * @param[out] dst the chunk span in which to copy
 * @param[in]  value the value to fill `dst`
 * @return dst as a ChunkSpan
 */
template <class ElementType, class Support, class Layout, class MemorySpace, class T>
KOKKOS_FUNCTION auto fill(
        ChunkSpan<ElementType, Support, Layout, MemorySpace> const& dst,
        T const& value) noexcept
{
    static_assert(
            std::is_assignable_v<
                    chunk_reference_t<ChunkSpan<ElementType, Support, Layout, MemorySpace>>,
                    T>,
            "Not assignable");
    KOKKOS_ASSERT((Kokkos::SpaceAccessibility<DDC_CURRENT_KOKKOS_SPACE, MemorySpace>::accessible));
    for_each(dst.domain(), [&dst, &value](typename Support::discrete_element_type const& elem) {
        dst(elem) = value;
    });
    return dst;
}

} // namespace ddc
