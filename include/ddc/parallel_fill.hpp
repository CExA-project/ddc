// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "chunk_traits.hpp"

namespace ddc {

/** Fill a borrowed chunk with a given value
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  value the value to fill `dst`
 * @return dst as a ChunkSpan
 */
template <class ChunkDst, class T>
auto parallel_fill(ChunkDst&& dst, T const& value)
{
    static_assert(is_borrowed_chunk_v<ChunkDst>);
    static_assert(std::is_assignable_v<chunk_reference_t<ChunkDst>, T>, "Not assignable");
    Kokkos::deep_copy(dst.allocation_kokkos_view(), value);
    return dst.span_view();
}

/** Fill a borrowed chunk with a given value
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  value the value to fill `dst`
 * @return dst as a ChunkSpan
 */
template <class ExecSpace, class ChunkDst, class T>
auto parallel_fill(ExecSpace const& execution_space, ChunkDst&& dst, T const& value)
{
    static_assert(is_borrowed_chunk_v<ChunkDst>);
    static_assert(std::is_assignable_v<chunk_reference_t<ChunkDst>, T>, "Not assignable");
    Kokkos::deep_copy(execution_space, dst.allocation_kokkos_view(), value);
    return dst.span_view();
}

} // namespace ddc
