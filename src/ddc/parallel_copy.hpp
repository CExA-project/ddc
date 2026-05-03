// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "chunk_traits.hpp"
#include "ddc_to_kokkos_execution_policy.hpp"

namespace ddc {

namespace detail {

template <typename ChunkSpanDst, typename ChunkSpanSrc, typename IndexSequence>
class CopyKokkosLambdaAdapter
{
};

template <typename ChunkSpanDst, typename ChunkSpanSrc, std::size_t... Idx>
class CopyKokkosLambdaAdapter<ChunkSpanDst, ChunkSpanSrc, std::index_sequence<Idx...>>
{
    template <std::size_t I>
    using index_type = DiscreteVectorElement;

    ChunkSpanDst m_dst;

    ChunkSpanSrc m_src;

public:
    explicit CopyKokkosLambdaAdapter(ChunkSpanDst const& dst, ChunkSpanSrc const& src)
        : m_dst(dst)
        , m_src(src)
    {
    }

    KOKKOS_FUNCTION void operator()(index_type<0> /*id*/) const
        requires(sizeof...(Idx) == 0)
    {
        m_dst() = m_src();
    }

    KOKKOS_FUNCTION void operator()(index_type<Idx>... ids) const
        requires(sizeof...(Idx) > 0)
    {
        using DVectDst = ChunkSpanDst::discrete_vector_type;
        using DVectSrc = ChunkSpanSrc::discrete_vector_type;
        DVectDst const ddst(ids...);
        m_dst(ddst) = m_src(DVectSrc(ddst));
    }
};

template <typename ChunkSpanDst, typename ChunkSpanSrc>
CopyKokkosLambdaAdapter(ChunkSpanDst const& dst, ChunkSpanSrc const& src)
        -> CopyKokkosLambdaAdapter<
                ChunkSpanDst,
                ChunkSpanSrc,
                std::make_index_sequence<ChunkSpanDst::rank()>>;

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
        Kokkos::parallel_for(
                "ddc_copy_default",
                detail::ddc_to_kokkos_execution_policy(
                        execution_space,
                        detail::array(dst.domain().extents())),
                detail::CopyKokkosLambdaAdapter(dst.span_view(), src.span_cview()));
    }
    return dst.span_view();
}

} // namespace ddc
