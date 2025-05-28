// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "chunk_span.hpp"
#include "chunk_traits.hpp"
#include "parallel_for_each.hpp"

namespace ddc {

namespace detail {

template <
        typename Tsrc,
        typename Tdst,
        typename DDomSrc,
        typename DDomDst,
        typename MemorySpace,
        typename LayoutSrc,
        typename LayoutDst>
class CopyKokkosLambdaAdapter
{
    ddc::ChunkSpan<Tdst, DDomDst, LayoutDst, MemorySpace> m_dst;

    ddc::ChunkSpan<Tsrc const, DDomSrc, LayoutSrc, MemorySpace> m_src;

public:
    explicit CopyKokkosLambdaAdapter(
            ddc::ChunkSpan<Tdst, DDomDst, LayoutDst, MemorySpace> const& dst,
            ddc::ChunkSpan<Tsrc const, DDomSrc, LayoutSrc, MemorySpace> const& src)
        : m_dst(dst)
        , m_src(src)
    {
    }

    KOKKOS_FUNCTION void operator()(DDomDst::discrete_element_type idst) const
    {
        m_dst(idst) = m_src(typename DDomSrc::discrete_element_type(idst));
    }
};

} // namespace detail

/** Copy the content of a borrowed chunk into another. It supports transposition and broadcasting at the same time.
 * The two arrays must be accessible from execution_space.
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in]  src the borrowed chunk from which to copy
 * @return dst as a ChunkSpan
*/
template <class ExecSpace, class ChunkDst, class ChunkSrc>
auto parallel_copy(ExecSpace const& execution_space, ChunkDst&& dst, ChunkSrc&& src)
{
    static_assert(is_borrowed_chunk_v<ChunkDst>);
    static_assert(is_borrowed_chunk_v<ChunkSrc>);
    static_assert(Kokkos::SpaceAccessibility<
                  ExecSpace,
                  typename std::remove_cvref_t<ChunkDst>::memory_space>::accessible);
    static_assert(Kokkos::SpaceAccessibility<
                  ExecSpace,
                  typename std::remove_cvref_t<ChunkSrc>::memory_space>::accessible);
    static_assert(
            std::is_assignable_v<chunk_reference_t<ChunkDst>, chunk_reference_t<ChunkSrc>>,
            "Not assignable");
    using DDomDst = decltype(dst.domain());
    using DDomSrc = decltype(src.domain());
    assert(DDomSrc(dst.domain()) == src.domain());
    if constexpr (std::is_same_v<DDomDst, DDomSrc>) {
        Kokkos::deep_copy(
                execution_space,
                dst.allocation_kokkos_view(),
                src.allocation_kokkos_view());
    } else {
        // The current implementation uses a loop over dst dimensions.
        // Alternative implementations:
        // - outer loop over src dimensions and inner loop over batch dimensions
        // - outer loop over batch dimensions and inner loop over src dimensions
        ddc::parallel_for_each(
                execution_space,
                dst.domain(),
                detail::CopyKokkosLambdaAdapter(dst.span_view(), src.span_cview()));
    }
    return dst.span_view();
}

} // namespace ddc
