// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"

#include "chunk_traits.hpp"
#include "ddc_to_kokkos_execution_policy.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "parallel_transform_reduce.hpp"

namespace ddc::experimental {

template <class... Tags>
struct Dims
{
};

namespace detail {

enum class ScanType { inclusive, exclusive };

inline constexpr std::integral_constant<ScanType, ScanType::exclusive> exclusive_tag;
inline constexpr std::integral_constant<ScanType, ScanType::inclusive> inclusive_tag;

template <
        ScanType ScanValue,
        class Reducer,
        class Functor,
        class ChunkSpan,
        class Support,
        concepts::discrete_element DElem>
class TransformScanKokkosLambdaAdapter
{
    Reducer m_reducer;

    Functor m_functor;

    ChunkSpan m_chunk_span;

    Support m_support;

    DElem m_delem;

    using value_type = Reducer::value_type;

public:
    TransformScanKokkosLambdaAdapter(
            std::integral_constant<ScanType, ScanValue> /*scan_tag*/,
            Reducer const& r,
            Functor const& f,
            ChunkSpan const& chunk_span,
            Support const& support,
            DElem const& delem)
        : m_reducer(r)
        , m_functor(f)
        , m_chunk_span(chunk_span)
        , m_support(support)
        , m_delem(delem)
    {
    }

    KOKKOS_FUNCTION
    void join(value_type& dest, value_type const& src) const
    {
        value_type fake;
        ::ddc::detail::ddc_to_kokkos_reducer_t<Reducer> const kokkos_reducer(fake);
        kokkos_reducer.join(dest, src);
    }

    KOKKOS_FUNCTION
    void init(value_type& val) const
    {
        value_type fake;
        ::ddc::detail::ddc_to_kokkos_reducer_t<Reducer> const kokkos_reducer(fake);
        kokkos_reducer.init(val);
    }

    KOKKOS_FUNCTION void operator()(
            DiscreteVectorElement const id,
            Reducer::value_type& partial_reduction,
            bool const is_final) const
    {
        auto const delem_scan = m_support(typename Support::discrete_vector_type(id));
        if constexpr (ScanValue == ScanType::exclusive) {
            if (is_final) {
                m_chunk_span(m_delem, delem_scan) = partial_reduction;
            }
        }
        partial_reduction = m_reducer(partial_reduction, m_functor(m_delem, delem_scan));
        if constexpr (ScanValue == ScanType::inclusive) {
            if (is_final) {
                m_chunk_span(m_delem, delem_scan) = partial_reduction;
            }
        }
    }
};

template <
        ::ddc::detail::execution_space ExecSpace,
        detail::ScanType ScanValue,
        class DDim,
        concepts::borrowed_chunk ChunkDst,
        class BinaryReductionOp,
        class UnaryTransformOp>
void parallel_transform_scan(
        std::integral_constant<detail::ScanType, ScanValue> scan_tag,
        std::string const& label,
        ExecSpace const& execution_space,
        Dims<DDim> /*dim_tag*/,
        ChunkDst&& out,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    using DDomOut = std::remove_cvref_t<ChunkDst>::discrete_domain_type;
    using DDomScan = ::ddc::detail::Rebind<DDomOut, ::ddc::detail::TypeSeq<DDim>>::type;

    DDomOut const ddom_out = out.domain();
    DDomScan const ddom_scan(ddom_out);
    auto const ddom_batch = remove_dims_of(ddom_out, ddom_scan);

    host_for_each(ddom_batch, [&](auto ibatch) {
        Kokkos::parallel_scan(
                label,
                ::ddc::detail::ddc_to_kokkos_execution_policy(
                        execution_space,
                        ddc::detail::array(ddom_scan.extents())),
                detail::TransformScanKokkosLambdaAdapter(
                        scan_tag,
                        reduce,
                        transform,
                        out.span_view(),
                        ddom_scan,
                        ibatch));
    });
}

} // namespace detail

template <
        ::ddc::detail::execution_space ExecSpace,
        class DDim,
        concepts::borrowed_chunk ChunkDst,
        class BinaryReductionOp,
        class UnaryTransformOp>
void parallel_transform_inclusive_scan(
        std::string const& label,
        ExecSpace const& execution_space,
        Dims<DDim> dim_tag,
        ChunkDst&& out,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    detail::parallel_transform_scan(
            detail::inclusive_tag,
            label,
            execution_space,
            dim_tag,
            std::forward<ChunkDst>(out),
            reduce,
            transform);
}

template <
        ::ddc::detail::execution_space ExecSpace,
        class DDim,
        concepts::borrowed_chunk ChunkDst,
        class BinaryReductionOp,
        class UnaryTransformOp>
void parallel_transform_exclusive_scan(
        std::string const& label,
        ExecSpace const& execution_space,
        Dims<DDim> dim_tag,
        ChunkDst&& out,
        BinaryReductionOp const& reduce,
        UnaryTransformOp const& transform) noexcept
{
    detail::parallel_transform_scan(
            detail::exclusive_tag,
            label,
            execution_space,
            dim_tag,
            std::forward<ChunkDst>(out),
            reduce,
            transform);
}

} // namespace ddc::experimental
