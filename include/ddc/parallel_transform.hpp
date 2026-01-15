// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <Kokkos_Macros.hpp>

#include "chunk_span.hpp"
#include "chunk_traits.hpp"
#include "parallel_for_each.hpp"

namespace ddc {

namespace detail {

template <
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy,
        class MemorySpace,
        class Functor>
class TransformKokkosLambdaAdapter
{
    ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> m_chunk;

    Functor m_functor;

public:
    explicit TransformKokkosLambdaAdapter(
            ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk,
            Functor const& functor)
        : m_chunk(chunk)
        , m_functor(functor)
    {
    }

    KOKKOS_FUNCTION void operator()(
            typename SupportType::discrete_element_type const i) const noexcept
    {
        ElementType& value = m_chunk(i);
        value = m_functor(static_cast<ElementType const&>(value));
    }
};

} // namespace detail

/** Transform a borrowed chunk with a given transform functor
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be assignable to dst
 * @return dst as a ChunkSpan
 */
template <class ChunkDst, class UnaryTransformOp>
auto parallel_transform(ChunkDst&& dst, UnaryTransformOp&& transform)
{
    static_assert(is_borrowed_chunk_v<ChunkDst>);
    parallel_for_each(
            "ddc_parallel_transform_default",
            dst.domain(),
            detail::TransformKokkosLambdaAdapter(
                    dst.span_view(),
                    std::forward<UnaryTransformOp>(transform)));
    return dst.span_view();
}

/** Transform a borrowed chunk with a given transform functor
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] dst the borrowed chunk in which to copy
 * @param[in] transform a unary FunctionObject that will be applied to each element of the input
 *            range. The return type must be acceptable as input to reduce
 * @return dst as a ChunkSpan
 */
template <class ExecSpace, class ChunkDst, class UnaryTransformOp>
auto parallel_transform(
        ExecSpace const& execution_space,
        ChunkDst&& dst,
        UnaryTransformOp&& transform)
{
    static_assert(is_borrowed_chunk_v<ChunkDst>);
    parallel_for_each(
            "ddc_parallel_transform_default",
            execution_space,
            dst.domain(),
            detail::TransformKokkosLambdaAdapter(
                    dst.span_view(),
                    std::forward<UnaryTransformOp>(transform)));
    return dst.span_view();
}

} // namespace ddc
