// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <type_traits>
#include <vector>

#include <experimental/mdspan>

#include "ddc/chunk_common.hpp"
#include "ddc/discrete_domain.hpp"

template <class, class, class>
class Chunk;

template <
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = std::experimental::layout_right>
class ChunkSpan;

template <class ElementType, class SupportType, class LayoutStridedPolicy>
inline constexpr bool enable_chunk<ChunkSpan<ElementType, SupportType, LayoutStridedPolicy>> = true;

template <class ElementType, class SupportType, class LayoutStridedPolicy>
inline constexpr bool
        enable_borrowed_chunk<ChunkSpan<ElementType, SupportType, LayoutStridedPolicy>> = true;

template <class ElementType, class... DDims, class LayoutStridedPolicy>
class ChunkSpan<ElementType, DiscreteDomain<DDims...>, LayoutStridedPolicy>
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
    using span_type = ChunkSpan<ElementType, DiscreteDomain<DDims...>, LayoutStridedPolicy>;

    /// type of a view of this full chunk
    using view_type = ChunkSpan<ElementType const, DiscreteDomain<DDims...>, LayoutStridedPolicy>;

    using mdomain_type = DiscreteDomain<DDims...>;

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

    template <class, class, class>
    friend class ChunkSpan;

protected:
    template <class QueryDDim, class... ODDims>
    auto get_slicer_for(DiscreteCoordinate<ODDims...> const& c) const
    {
        if constexpr (in_tags_v<QueryDDim, detail::TypeSeq<ODDims...>>) {
            return get<QueryDDim>(c) - front<QueryDDim>(this->m_domain);
        } else {
            return std::experimental::full_extent;
        }
    }

    template <class QueryDDim, class... ODDims>
    auto get_slicer_for(DiscreteDomain<ODDims...> const& c) const
    {
        if constexpr (in_tags_v<QueryDDim, detail::TypeSeq<ODDims...>>) {
            return std::pair<std::size_t, std::size_t>(
                    front<QueryDDim>(c) - front<QueryDDim>(this->m_domain),
                    back<QueryDDim>(c) + 1 - front<QueryDDim>(this->m_domain));
        } else {
            return std::experimental::full_extent;
        }
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
    constexpr ChunkSpan(ChunkSpan<OElementType, mdomain_type, layout_type> const& other) noexcept
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
        extents_type extents_s((front<DDims>(domain) + ::extents<DDims>(domain))...);
        std::array<std::size_t, sizeof...(DDims)> strides_s {allocation_mdspan.mapping().stride(
                type_seq_rank_v<DDims, detail::TypeSeq<DDims...>>)...};
        stdex::layout_stride::mapping<extents_type> mapping_s(extents_s, strides_s);
        this->m_internal_mdspan = internal_mdspan_type(
                allocation_mdspan.data() - mapping_s(front<DDims>(domain)...),
                mapping_s);
        this->m_domain = domain;
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
        return ::ChunkSpan(subview, select_by_type_seq<selected_meshes>(this->m_domain));
    }

    /** Slice out some dimensions
     */
    template <class... QueryDDims>
    constexpr auto operator[](DiscreteDomain<QueryDDims...> const& odomain) const
    {
        auto subview = std::experimental::
                submdspan(allocation_mdspan(), get_slicer_for<DDims>(odomain)...);
        return ::ChunkSpan(subview, this->m_domain.restrict(odomain));
    }

    /** Element access using a list of DiscreteCoordinate
     * @param mcoords 1D discrete coordinates
     * @return reference to this element
     */
    // Warning: Do not use DiscreteCoordinate because of template deduction issue with clang 12
    template <class... ODDims>
    constexpr reference operator()(
            detail::TaggedVector<DiscreteCoordElement, ODDims> const&... mcoords) const noexcept
    {
        static_assert(sizeof...(ODDims) == sizeof...(DDims), "Invalid number of dimensions");
        assert(((mcoords >= front<ODDims>(this->m_domain)) && ...));
        assert(((mcoords <= back<ODDims>(this->m_domain)) && ...));
        return this->m_internal_mdspan(take<DDims>(mcoords...)...);
    }

    /** Element access using a multi-dimensional DiscreteCoordinate
     * @param mcoord discrete coordinates
     * @return reference to this element
     */
    template <class... ODDims, class = std::enable_if_t<sizeof...(ODDims) != 1>>
    constexpr reference operator()(
            detail::TaggedVector<DiscreteCoordElement, ODDims...> const& mcoord) const noexcept
    {
        static_assert(sizeof...(ODDims) == sizeof...(DDims), "Invalid number of dimensions");
        assert(((get<ODDims>(mcoord) >= front<ODDims>(this->m_domain)) && ...));
        assert(((get<ODDims>(mcoord) <= back<ODDims>(this->m_domain)) && ...));
        return this->m_internal_mdspan(get<DDims>(mcoord)...);
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
        class ElementType,
        class SupportType,
        class LayoutStridedPolicy = std::experimental::layout_right>
using ChunkView = ChunkSpan<ElementType const, SupportType, LayoutStridedPolicy>;
