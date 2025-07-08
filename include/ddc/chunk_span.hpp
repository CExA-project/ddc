// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"
#include "detail/type_seq.hpp"
#include "detail/type_traits.hpp"

#include "chunk_common.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"

namespace ddc {

template <class, class, class>
class Chunk;

template <
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = Kokkos::layout_right,
        class MemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
class ChunkSpan;

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool
        enable_chunk<ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
inline constexpr bool
        enable_borrowed_chunk<ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>>
        = true;

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
class ChunkSpan : public ChunkCommon<ElementType, SupportType, LayoutStridedPolicy>
{
    static_assert(
            std::is_same_v<LayoutStridedPolicy, Kokkos::layout_left>
                    || std::is_same_v<LayoutStridedPolicy, Kokkos::layout_right>
                    || std::is_same_v<LayoutStridedPolicy, Kokkos::layout_stride>,
            "ChunkSpan only supports layout_left, layout_right or layout_stride");

protected:
    using base_type = ChunkCommon<ElementType, SupportType, LayoutStridedPolicy>;

public:
    /// type of a span of this full chunk
    using span_type = ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace>;

    /// type of a view of this full chunk
    using view_type = ChunkSpan<ElementType const, SupportType, LayoutStridedPolicy, MemorySpace>;

    using discrete_domain_type = typename base_type::discrete_domain_type;

    using memory_space = MemorySpace;

    /// The dereferenceable part of the co-domain but with a different domain, starting at 0
    using allocation_mdspan_type = typename base_type::allocation_mdspan_type;

    using const_allocation_mdspan_type = typename base_type::const_allocation_mdspan_type;

    using discrete_element_type = typename discrete_domain_type::discrete_element_type;

    using discrete_vector_type = typename discrete_domain_type::discrete_vector_type;

    using extents_type = typename base_type::extents_type;

    using layout_type = typename base_type::layout_type;

    using accessor_type = typename base_type::accessor_type;

    using mapping_type = typename base_type::mapping_type;

    using element_type = typename base_type::element_type;

    using value_type = typename base_type::value_type;

    using size_type = typename base_type::size_type;

    using data_handle_type = typename base_type::data_handle_type;

    using reference = typename base_type::reference;

    template <class, class, class, class>
    friend class ChunkSpan;

protected:
    template <class QueryDDim, class... ODDims>
    KOKKOS_FUNCTION static constexpr auto get_slicer_for(DiscreteVector<ODDims...> const& c)
    {
        DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
        if constexpr (in_tags_v<QueryDDim, detail::TypeSeq<ODDims...>>) {
            return c.template get<QueryDDim>();
        } else {
            return Kokkos::full_extent;
        }
        DDC_IF_NVCC_THEN_POP
    }

    template <class QueryDDim, class... ODDims, class... OODDims>
    KOKKOS_FUNCTION static constexpr auto get_slicer_for(
            DiscreteDomain<ODDims...> const& c,
            DiscreteDomain<OODDims...> const& origin)
    {
        DDC_IF_NVCC_THEN_PUSH_AND_SUPPRESS(implicit_return_from_non_void_function)
        if constexpr (in_tags_v<QueryDDim, detail::TypeSeq<ODDims...>>) {
            DiscreteDomain<QueryDDim> const c_slice(c);
            DiscreteDomain<QueryDDim> const origin_slice(origin);
            return std::pair<std::size_t, std::size_t>(
                    c_slice.front() - origin_slice.front(),
                    c_slice.back() + 1 - origin_slice.front());
        } else {
            return Kokkos::full_extent;
        }
        DDC_IF_NVCC_THEN_POP
    }

    template <class TypeSeq>
    struct slicer;

    template <class... DDims>
    struct slicer<detail::TypeSeq<DDims...>>
    {
        template <class... ODDims>
        KOKKOS_FUNCTION constexpr auto operator()(
                allocation_mdspan_type const& span,
                DiscreteVector<ODDims...> const& c) const
        {
            return Kokkos::submdspan(span, get_slicer_for<DDims>(c)...);
        }

        template <class... ODDims, class... OODDims>
        KOKKOS_FUNCTION constexpr auto operator()(
                allocation_mdspan_type const& span,
                DiscreteDomain<ODDims...> const& c,
                DiscreteDomain<OODDims...> const& origin) const
        {
            return Kokkos::submdspan(span, get_slicer_for<DDims>(c, origin)...);
        }
    };

public:
    /// Empty ChunkSpan
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan() = default;

    /** Constructs a new ChunkSpan by copy, yields a new view to the same data
     * @param other the ChunkSpan to copy
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan(ChunkSpan const& other) = default;

    /** Constructs a new ChunkSpan by move
     * @param other the ChunkSpan to move
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan(ChunkSpan&& other) noexcept = default;

    /** Forbids to construct a ChunkSpan from a rvalue of type Chunk.
     */
    template <
            class OElementType,
            class Allocator,
            class = std::enable_if_t<std::is_same_v<typename Allocator::memory_space, MemorySpace>>>
    ChunkSpan(Chunk<OElementType, SupportType, Allocator>&& other) noexcept = delete;

    /** Constructs a new ChunkSpan from a Chunk, yields a new view to the same data
     * @param other the Chunk to view
     */
    template <
            class OElementType,
            class Allocator,
            class = std::enable_if_t<std::is_same_v<typename Allocator::memory_space, MemorySpace>>>
    KOKKOS_FUNCTION constexpr explicit ChunkSpan(
            Chunk<OElementType, SupportType, Allocator>& other) noexcept
        : base_type(other.m_allocation_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan from a Chunk, yields a new view to the same data
     * @param other the Chunk to view
     */
    // Disabled by SFINAE in the case of `ElementType` is not `const` to avoid write access
    template <
            class OElementType,
            class SFINAEElementType = ElementType,
            class = std::enable_if_t<std::is_const_v<SFINAEElementType>>,
            class Allocator,
            class = std::enable_if_t<std::is_same_v<typename Allocator::memory_space, MemorySpace>>>
    KOKKOS_FUNCTION constexpr explicit ChunkSpan(
            Chunk<OElementType, SupportType, Allocator> const& other) noexcept
        : base_type(other.m_allocation_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan by copy of a chunk, yields a new view to the same data
     * @param other the ChunkSpan to move
     */
    template <class OElementType>
    KOKKOS_FUNCTION constexpr explicit ChunkSpan(
            ChunkSpan<OElementType, SupportType, layout_type, MemorySpace> const& other) noexcept
        : base_type(other.m_allocation_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan from scratch
     * @param ptr the allocation pointer to the data
     * @param domain the domain that sustains the view
     */
    template <
            class Mapping = mapping_type,
            std::enable_if_t<std::is_constructible_v<Mapping, extents_type>, int> = 0>
    KOKKOS_FUNCTION constexpr ChunkSpan(ElementType* const ptr, SupportType const& domain)
        : base_type(ptr, domain)
    {
    }

    /** Constructs a new ChunkSpan from scratch
     * @param allocation_mdspan the allocation mdspan to the data
     * @param domain the domain that sustains the view
     */
    KOKKOS_FUNCTION constexpr ChunkSpan(
            allocation_mdspan_type allocation_mdspan,
            SupportType const& domain)
        : base_type(allocation_mdspan, domain)
    {
        for (std::size_t i = 0; i < SupportType::rank(); ++i) {
            assert(allocation_mdspan.extent(i)
                   == static_cast<std::size_t>(detail::array(domain.extents())[i]));
        }
    }

    /** Constructs a new ChunkSpan from scratch
     * @param view the Kokkos view
     * @param domain the domain that sustains the view
     */
    template <class KokkosView, class = std::enable_if_t<Kokkos::is_view_v<KokkosView>>>
    KOKKOS_FUNCTION constexpr ChunkSpan(KokkosView const& view, SupportType const& domain) noexcept
        : ChunkSpan(
                  detail::build_mdspan(view, std::make_index_sequence<SupportType::rank()> {}),
                  domain)
    {
    }

    KOKKOS_DEFAULTED_FUNCTION ~ChunkSpan() noexcept = default;

    /** Copy-assigns a new value to this ChunkSpan, yields a new view to the same data
     * @param other the ChunkSpan to copy
     * @return *this
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan& operator=(ChunkSpan const& other) = default;

    /** Move-assigns a new value to this ChunkSpan
     * @param other the ChunkSpan to move
     * @return *this
     */
    KOKKOS_DEFAULTED_FUNCTION constexpr ChunkSpan& operator=(ChunkSpan&& other) noexcept = default;

    /** Slice out some dimensions
     */
    template <class... QueryDDims>
    KOKKOS_FUNCTION constexpr auto operator[](
            DiscreteElement<QueryDDims...> const& slice_spec) const
    {
        assert(select<QueryDDims...>(this->m_domain).contains(slice_spec));
        slicer<to_type_seq_t<SupportType>> const slicer;
        auto subview = slicer(
                this->allocation_mdspan(),
                ddc::DiscreteDomain<QueryDDims...>(this->m_domain).distance_from_front(slice_spec));
        using layout_type = typename decltype(subview)::layout_type;
        using extents_type = typename decltype(subview)::extents_type;
        using detail::TypeSeq;
        using OutTypeSeqDDims
                = type_seq_remove_t<to_type_seq_t<SupportType>, TypeSeq<QueryDDims...>>;
        using OutDDom = typename detail::RebindDomain<SupportType, OutTypeSeqDDims>::type;
        if constexpr (
                std::is_same_v<layout_type, Kokkos::Experimental::layout_left_padded<>>
                || std::is_same_v<layout_type, Kokkos::Experimental::layout_right_padded<>>) {
            Kokkos::layout_stride::mapping<extents_type> const mapping_stride(subview.mapping());
            Kokkos::mdspan<ElementType, extents_type, Kokkos::layout_stride> const
                    a(subview.data_handle(), mapping_stride);
            return ChunkSpan<
                    ElementType,
                    OutDDom,
                    Kokkos::layout_stride,
                    memory_space>(a, OutDDom(this->m_domain));
        } else {
            return ChunkSpan<
                    ElementType,
                    OutDDom,
                    layout_type,
                    memory_space>(subview, OutDDom(this->m_domain));
        }
    }

    /** Restrict to a subdomain, only valid when SupportType is a DiscreteDomain
     */
    template <
            class... QueryDDims,
            class SFINAESupportType = SupportType,
            std::enable_if_t<is_discrete_domain_v<SFINAESupportType>, std::nullptr_t> = nullptr>
    KOKKOS_FUNCTION constexpr auto operator[](DiscreteDomain<QueryDDims...> const& odomain) const
    {
        assert(odomain.empty()
               || (DiscreteDomain<QueryDDims...>(this->m_domain).contains(odomain.front())
                   && DiscreteDomain<QueryDDims...>(this->m_domain).contains(odomain.back())));
        slicer<to_type_seq_t<SupportType>> const slicer;
        auto subview = slicer(this->allocation_mdspan(), odomain, this->m_domain);
        using layout_type = typename decltype(subview)::layout_type;
        using extents_type = typename decltype(subview)::extents_type;
        if constexpr (
                std::is_same_v<layout_type, Kokkos::Experimental::layout_left_padded<>>
                || std::is_same_v<layout_type, Kokkos::Experimental::layout_right_padded<>>) {
            Kokkos::layout_stride::mapping<extents_type> const mapping_stride(subview.mapping());
            Kokkos::mdspan<ElementType, extents_type, Kokkos::layout_stride> const
                    a(subview.data_handle(), mapping_stride);
            return ChunkSpan<
                    ElementType,
                    decltype(this->m_domain.restrict_with(odomain)),
                    Kokkos::layout_stride,
                    memory_space>(a, this->m_domain.restrict_with(odomain));
        } else {
            return ChunkSpan<
                    ElementType,
                    decltype(this->m_domain.restrict_with(odomain)),
                    layout_type,
                    memory_space>(subview, this->m_domain.restrict_with(odomain));
        }
    }

    /** Element access using a list of DiscreteElement
     * @param delems discrete elements
     * @return reference to this element
     */
    template <
            class... DElems,
            std::enable_if_t<detail::all_of_v<is_discrete_element_v<DElems>...>, int> = 0>
    KOKKOS_FUNCTION constexpr reference operator()(DElems const&... delems) const noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DElems::size()),
                "Invalid number of dimensions");
        assert(this->m_domain.contains(delems...));
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(this->m_domain.distance_from_front(delems...)));
    }

    /** Element access using a list of DiscreteVector
     * @param dvects discrete vectors
     * @return reference to this element
     */
    template <
            class... DVects,
            std::enable_if_t<detail::all_of_v<is_discrete_vector_v<DVects>...>, int> = 0,
            std::enable_if_t<sizeof...(DVects) != 0, int> = 0>
    KOKKOS_FUNCTION constexpr reference operator()(DVects const&... dvects) const noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DVects::size()),
                "Invalid number of dimensions");
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(discrete_vector_type(dvects...)));
    }

    /** Access to the underlying allocation pointer
     * @return allocation pointer
     */
    KOKKOS_FUNCTION constexpr ElementType* data_handle() const
    {
        return base_type::data_handle();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    KOKKOS_FUNCTION constexpr allocation_mdspan_type allocation_mdspan() const
    {
        return base_type::allocation_mdspan();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    KOKKOS_FUNCTION constexpr auto allocation_kokkos_view() const
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<SupportType::rank()> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_t<ElementType, SupportType::rank()>,
                decltype(kokkos_layout),
                MemorySpace>(s.data_handle(), kokkos_layout);
    }

    KOKKOS_FUNCTION constexpr view_type span_cview() const
    {
        return view_type(*this);
    }

    KOKKOS_FUNCTION constexpr span_type span_view() const
    {
        return *this;
    }
};

template <class DataType, class... Properties, class SupportType>
KOKKOS_DEDUCTION_GUIDE ChunkSpan(
        Kokkos::View<DataType, Properties...> const& view,
        SupportType domain)
        -> ChunkSpan<
                detail::kokkos_to_mdspan_element_t<
                        typename Kokkos::View<DataType, Properties...>::data_type>,
                SupportType,
                detail::kokkos_to_mdspan_layout_t<
                        typename Kokkos::View<DataType, Properties...>::array_layout>,
                typename Kokkos::View<DataType, Properties...>::memory_space>;

template <class ElementType, class SupportType, class Allocator>
ChunkSpan(Chunk<ElementType, SupportType, Allocator>& other) -> ChunkSpan<
        ElementType,
        SupportType,
        Kokkos::layout_right,
        typename Allocator::memory_space>;

template <class ElementType, class SupportType, class Allocator>
ChunkSpan(Chunk<ElementType, SupportType, Allocator> const& other) -> ChunkSpan<
        ElementType const,
        SupportType,
        Kokkos::layout_right,
        typename Allocator::memory_space>;

template <
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = Kokkos::layout_right,
        class MemorySpace = Kokkos::HostSpace>
using ChunkView = ChunkSpan<ElementType const, SupportType, LayoutStridedPolicy, MemorySpace>;

} // namespace ddc
