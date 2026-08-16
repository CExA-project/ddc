// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "chunk_traits.hpp"

namespace ddc {

/** Copy the content of a borrowed chunk into another
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  src the borrowed chunk from which to copy
 * @return dst as a ChunkSpan
*/
template <concepts::borrowed_chunk ChunkDst, concepts::borrowed_chunk ChunkSrc>
auto parallel_deepcopy(ChunkDst&& dst, ChunkSrc&& src)
{
    auto&& dst_ref = std::forward<ChunkDst>(dst);
    auto&& src_ref = std::forward<ChunkSrc>(src);
    static_assert(
            std::is_assignable_v<chunk_reference_t<ChunkDst>, chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    static_assert(std::is_same_v<decltype(dst_ref.domain()), decltype(src_ref.domain())>);
    assert(dst_ref.domain() == src_ref.domain());
    Kokkos::deep_copy(dst_ref.allocation_kokkos_view(), src_ref.allocation_kokkos_view());
    return dst_ref.span_view();
}

/** Copy the content of a borrowed chunk into another
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  src the borrowed chunk from which to copy
 * @return dst as a ChunkSpan
*/
template <class ExecSpace, concepts::borrowed_chunk ChunkDst, concepts::borrowed_chunk ChunkSrc>
auto parallel_deepcopy(ExecSpace const& execution_space, ChunkDst&& dst, ChunkSrc&& src)
{
    static_assert(
            std::is_assignable_v<chunk_reference_t<ChunkDst>, chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    static_assert(
            std::is_same_v<decltype(dst.domain()), decltype(src.domain())>,
            "ddc::parallel_deepcopy only supports domains whose dimensions are of the same order");
    auto&& dst_ref = std::forward<ChunkDst>(dst);
    auto&& src_ref = std::forward<ChunkSrc>(src);
    assert(dst_ref.domain() == src_ref.domain());
    Kokkos::deep_copy(
            execution_space,
            dst_ref.allocation_kokkos_view(),
            src_ref.allocation_kokkos_view());
    return dst_ref.span_view();
}

} // namespace ddc
