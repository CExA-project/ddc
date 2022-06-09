// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include <experimental/mdspan>

#include "ddc/chunk_common.hpp"
#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"

template <class, class, class>
class Chunk;

template <
        class ElementType,
        class SupportType,
        class MemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space,
        class LayoutStridedPolicy = std::experimental::layout_right>
class ChunkSpan;

template <class ElementType, class SupportType, class MemorySpace, class LayoutStridedPolicy>
inline constexpr bool
        enable_chunk<ChunkSpan<ElementType, SupportType, MemorySpace, LayoutStridedPolicy>> = true;

template <class ElementType, class SupportType, class MemorySpace, class LayoutStridedPolicy>
inline constexpr bool enable_borrowed_chunk<
        ChunkSpan<ElementType, SupportType, MemorySpace, LayoutStridedPolicy>> = true;

template <class ElementType, class... DDims, class MemorySpace, class LayoutStridedPolicy>
class ChunkSpan<ElementType, DiscreteDomain<DDims...>, MemorySpace, LayoutStridedPolicy>
    : public ChunkCommon<ElementType, DiscreteDomain<DDims...>, LayoutStridedPolicy>
{
protected:
    /// the raw mdspan underlying this, with the same indexing (0 might no be dereferenceable)
    using internal_mdspan_type = std::experimental::mdspan<
            ElementType,
            std::experimental::dextents<sizeof...(DDims)>,
            std::experimental::layout_stride>;

    using base_type = ChunkCommon<ElementType, DiscreteDomain<DDims...>, LayoutStridedPolicy>;

public:
    /// type of a span of this full chunk
    using span_type
            = ChunkSpan<ElementType, DiscreteDomain<DDims...>, MemorySpace, LayoutStridedPolicy>;

    /// type of a view of this full chunk
    using view_type = ChunkSpan<
            ElementType const,
            DiscreteDomain<DDims...>,
            MemorySpace,
            LayoutStridedPolicy>;

    using mdomain_type = DiscreteDomain<DDims...>;

    using memory_space = MemorySpace;

    /// The dereferenceable part of the co-domain but with a different domain, starting at 0
    using allocation_mdspan_type = std::experimental::
            mdspan<ElementType, std::experimental::dextents<sizeof...(DDims)>, LayoutStridedPolicy>;

    using mcoord_type = typename mdomain_type::mcoord_type;

    using extents_type = typename base_type::extents_type;

    using layout_type = typename base_type::layout_type;

    using accessor_type = typename base_type::accessor_type;

    using mapping_type = typename base_type::mapping_type;

    using element_type = typename base_type::element_type;

    using value_type = typename base_type::value_type;

    using size_type = typename base_type::size_type;

    using difference_type = typename base_type::difference_type;

    using pointer = typename base_type::pointer;

    using reference = typename base_type::reference;

    template <class, class, class, class>
    friend class ChunkSpan;

protected:
    template <class QueryDDim, class... ODDims>
    auto get_slicer_for(DiscreteCoordinate<ODDims...> const& c) const
    {
#if defined(DDC_INTERNAL_FIX_NVCC_IF_CONSTEXPR)
        DDC_NV_DIAG_SUPPRESS(implicit_return_from_non_void_function)
        return [&]() {
#endif
            if constexpr (in_tags_v<QueryDDim, detail::TypeSeq<ODDims...>>) {
                return (uid<QueryDDim>(c) - front<QueryDDim>(this->m_domain).uid());
            } else {
                return std::experimental::full_extent;
            }
#if defined(DDC_INTERNAL_FIX_NVCC_IF_CONSTEXPR)
        }();
        DDC_NV_DIAG_DEFAULT(implicit_return_from_non_void_function)
#endif
    }

    template <class QueryDDim, class... ODDims>
    auto get_slicer_for(DiscreteDomain<ODDims...> const& c) const
    {
#if defined(DDC_INTERNAL_FIX_NVCC_IF_CONSTEXPR)
        DDC_NV_DIAG_SUPPRESS(implicit_return_from_non_void_function)
        return [&]() {
#endif
            if constexpr (in_tags_v<QueryDDim, detail::TypeSeq<ODDims...>>) {
                return std::pair<std::size_t, std::size_t>(
                        front<QueryDDim>(c) - front<QueryDDim>(this->m_domain),
                        back<QueryDDim>(c) + 1 - front<QueryDDim>(this->m_domain));
            } else {
                return std::experimental::full_extent;
            }
#if defined(DDC_INTERNAL_FIX_NVCC_IF_CONSTEXPR)
        }();
        DDC_NV_DIAG_DEFAULT(implicit_return_from_non_void_function)
#endif
    }

public:
    /// Empty ChunkSpan
    constexpr ChunkSpan() = default;

    /** Constructs a new ChunkSpan by copy, yields a new view to the same data
     * @param other the ChunkSpan to copy
     */
    constexpr ChunkSpan(ChunkSpan const& other) = default;

    /** Constructs a new ChunkSpan by move
     * @param other the ChunkSpan to move
     */
    constexpr ChunkSpan(ChunkSpan&& other) = default;

    /** Constructs a new ChunkSpan from a Chunk, yields a new view to the same data
     * @param other the Chunk to view
     */
    template <class OElementType, class Allocator>
    constexpr ChunkSpan(Chunk<OElementType, mdomain_type, Allocator>& other) noexcept
        : base_type(other.m_internal_mdspan, other.m_domain)
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
            class Allocator>
    constexpr ChunkSpan(Chunk<OElementType, mdomain_type, Allocator> const& other) noexcept
        : base_type(other.m_internal_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan by copy of a chunk, yields a new view to the same data
     * @param other the ChunkSpan to move
     */
    template <class OElementType>
    constexpr ChunkSpan(
            ChunkSpan<OElementType, mdomain_type, MemorySpace, layout_type> const& other) noexcept
        : base_type(other.m_internal_mdspan, other.m_domain)
    {
    }

    /** Constructs a new ChunkSpan from scratch
     * @param ptr the allocation pointer to the data
     * @param domain the domain that sustains the view
     */
    template <
            class Mapping = mapping_type,
            std::enable_if_t<std::is_constructible_v<Mapping, extents_type>, int> = 0>
    constexpr ChunkSpan(ElementType* const ptr, mdomain_type const& domain) : base_type(ptr, domain)
    {
    }

    /** Constructs a new ChunkSpan from scratch
     * @param allocation_mdspan the allocation mdspan to the data
     * @param domain the domain that sustains the view
     */
    constexpr ChunkSpan(allocation_mdspan_type allocation_mdspan, mdomain_type const& domain)
    {
        namespace stdex = std::experimental;
        extents_type extents_s((front<DDims>(domain) + ::extents<DDims>(domain)).uid()...);
        std::array<std::size_t, sizeof...(DDims)> strides_s {allocation_mdspan.mapping().stride(
                type_seq_rank_v<DDims, detail::TypeSeq<DDims...>>)...};
        stdex::layout_stride::mapping<extents_type> mapping_s(extents_s, strides_s);
        this->m_internal_mdspan = internal_mdspan_type(
                allocation_mdspan.data() - mapping_s(front<DDims>(domain).uid()...),
                mapping_s);
        this->m_domain = domain;
    }

    /** Constructs a new ChunkSpan from scratch
     * @param view the Kokkos view
     * @param domain the domain that sustains the view
     */
    template <class KokkosView, class = std::enable_if_t<Kokkos::is_view<KokkosView>::value>>
    constexpr ChunkSpan(KokkosView const& view, mdomain_type const& domain) noexcept
        : ChunkSpan(
                detail::build_mdspan(view, std::make_index_sequence<sizeof...(DDims)> {}),
                domain)
    {
    }

    /** Copy-assigns a new value to this ChunkSpan, yields a new view to the same data
     * @param other the ChunkSpan to copy
     * @return *this
     */
    constexpr ChunkSpan& operator=(ChunkSpan const& other) = default;

    /** Move-assigns a new value to this ChunkSpan
     * @param other the ChunkSpan to move
     * @return *this
     */
    constexpr ChunkSpan& operator=(ChunkSpan&& other) = default;

    /** Slice out some dimensions
     */
    template <class... QueryDDims>
    constexpr auto operator[](DiscreteCoordinate<QueryDDims...> const& slice_spec) const
    {
        auto subview = std::experimental::
                submdspan(allocation_mdspan(), get_slicer_for<DDims>(slice_spec)...);
        using detail::TypeSeq;
        using selected_meshes = type_seq_remove_t<TypeSeq<DDims...>, TypeSeq<QueryDDims...>>;
        return ::ChunkSpan<
                ElementType,
                decltype(select_by_type_seq<selected_meshes>(this->m_domain)),
                memory_space,
                typename decltype(subview)::
                        layout_type>(subview, select_by_type_seq<selected_meshes>(this->m_domain));
    }

    /** Slice out some dimensions
     */
    template <class... QueryDDims>
    constexpr auto operator[](DiscreteDomain<QueryDDims...> const& odomain) const
    {
        auto subview = std::experimental::
                submdspan(allocation_mdspan(), get_slicer_for<DDims>(odomain)...);
        return ::ChunkSpan<
                ElementType,
                decltype(this->m_domain.restrict(odomain)),
                memory_space,
                typename decltype(subview)::layout_type>(subview, this->m_domain.restrict(odomain));
    }

    /** Element access using a list of DiscreteCoordinate
     * @param mcoords 1D discrete coordinates
     * @return reference to this element
     */
    template <class... ODDims>
    constexpr reference operator()(DiscreteCoordinate<ODDims> const&... mcoords) const noexcept
    {
        static_assert(sizeof...(ODDims) == sizeof...(DDims), "Invalid number of dimensions");
        assert(((mcoords >= front<ODDims>(this->m_domain)) && ...));
        assert(((mcoords <= back<ODDims>(this->m_domain)) && ...));
        return this->m_internal_mdspan(uid(take<DDims>(mcoords...))...);
    }

    /** Element access using a multi-dimensional DiscreteCoordinate
     * @param mcoord discrete coordinates
     * @return reference to this element
     */
    template <class... ODDims, class = std::enable_if_t<sizeof...(ODDims) != 1>>
    constexpr reference operator()(DiscreteCoordinate<ODDims...> const& mcoord) const noexcept
    {
        static_assert(sizeof...(ODDims) == sizeof...(DDims), "Invalid number of dimensions");
        assert(((select<ODDims>(mcoord) >= front<ODDims>(this->m_domain)) && ...));
        assert(((select<ODDims>(mcoord) <= back<ODDims>(this->m_domain)) && ...));
        return this->m_internal_mdspan(uid<DDims>(mcoord)...);
    }

    /** Access to the underlying allocation pointer
     * @return allocation pointer
     */
    constexpr ElementType* data() const
    {
        return base_type::data();
    }

    /// @deprecated
    [[deprecated]] constexpr internal_mdspan_type internal_mdspan() const
    {
        return base_type::internal_mdspan();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    constexpr allocation_mdspan_type allocation_mdspan() const
    {
        return base_type::allocation_mdspan();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    constexpr auto allocation_kokkos_view() const
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<sizeof...(DDims)> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_type_t<ElementType, sizeof...(DDims)>,
                decltype(kokkos_layout),
                MemorySpace>(s.data(), kokkos_layout);
    }

    constexpr view_type span_cview() const
    {
        return view_type(*this);
    }

    constexpr span_type span_view() const
    {
        return *this;
    }
};

template <class ElementType, class... DDims, class Extents, class StridedLayout>
ChunkSpan(
        std::experimental::mdspan<ElementType, Extents, StridedLayout> allocation_view,
        DiscreteDomain<DDims...> domain)
        -> ChunkSpan<ElementType, DiscreteDomain<DDims...>, StridedLayout>;

template <
        class KokkosView,
        class... DDims,
        class = std::enable_if_t<Kokkos::is_view<KokkosView>::value>>
ChunkSpan(KokkosView const& view, DiscreteDomain<DDims...> domain) -> ChunkSpan<
        detail::kokkos_to_mdspan_element_type_t<typename KokkosView::data_type>,
        DiscreteDomain<DDims...>,
        typename KokkosView::memory_space,
        detail::mdspan_layout_t<typename KokkosView::array_layout>>;

template <
        class ElementType,
        class SupportType,
        class MemorySpace,
        class LayoutStridedPolicy = std::experimental::layout_right>
using ChunkView = ChunkSpan<ElementType const, SupportType, MemorySpace, LayoutStridedPolicy>;
