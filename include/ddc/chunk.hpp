// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"
#include "detail/type_traits.hpp"

#include "chunk_common.hpp"
#include "chunk_span.hpp"
#include "chunk_traits.hpp"
#include "kokkos_allocator.hpp"

namespace ddc {

template <class ElementType, class, class Allocator = HostAllocator<ElementType>>
class Chunk;

template <class ElementType, class SupportType, class Allocator>
inline constexpr bool enable_chunk<Chunk<ElementType, SupportType, Allocator>> = true;

template <class ElementType, class SupportType, class Allocator>
class Chunk : public ChunkCommon<ElementType, SupportType, Kokkos::layout_right>
{
protected:
    using base_type = ChunkCommon<ElementType, SupportType, Kokkos::layout_right>;

public:
    /// type of a span of this full chunk
    using span_type = ChunkSpan<
            ElementType,
            SupportType,
            Kokkos::layout_right,
            typename Allocator::memory_space>;

    /// type of a view of this full chunk
    using view_type = ChunkSpan<
            ElementType const,
            SupportType,
            Kokkos::layout_right,
            typename Allocator::memory_space>;

    /// The dereferenceable part of the co-domain but with indexing starting at 0
    using allocation_mdspan_type = typename base_type::allocation_mdspan_type;

    using const_allocation_mdspan_type = typename base_type::const_allocation_mdspan_type;

    using discrete_domain_type = typename base_type::discrete_domain_type;

    using memory_space = typename Allocator::memory_space;

    using discrete_element_type = typename base_type::discrete_element_type;

    using discrete_vector_type = typename base_type::discrete_vector_type;

    using extents_type = typename base_type::extents_type;

    using layout_type = typename base_type::layout_type;

    using mapping_type = typename base_type::mapping_type;

    using element_type = typename base_type::element_type;

    using value_type = typename base_type::value_type;

    using size_type = typename base_type::size_type;

    using data_handle_type = typename base_type::data_handle_type;

    using reference = typename base_type::reference;

    template <class, class, class>
    friend class Chunk;

private:
    Allocator m_allocator;

    std::string m_label;

public:
    /// Empty Chunk
    Chunk() = default;

    /// Construct a labeled Chunk on a domain with uninitialized values
    explicit Chunk(
            std::string const& label,
            SupportType const& domain,
            Allocator allocator = Allocator())
        : base_type(allocator.allocate(label, domain.size()), domain)
        , m_allocator(std::move(allocator))
        , m_label(label)
    {
    }

    /// Construct a Chunk on a domain with uninitialized values
    explicit Chunk(SupportType const& domain, Allocator allocator = Allocator())
        : Chunk("no-label", domain, std::move(allocator))
    {
    }

    /// Deleted: use deepcopy instead
    Chunk(Chunk const& other) = delete;

    /** Constructs a new Chunk by move
     * @param other the Chunk to move
     */
    Chunk(Chunk&& other) noexcept
        : base_type(std::move(static_cast<base_type&>(other)))
        , m_allocator(std::move(other.m_allocator))
        , m_label(std::move(other.m_label))
    {
        other.m_allocation_mdspan
                = allocation_mdspan_type(nullptr, other.m_allocation_mdspan.mapping());
    }

    ~Chunk() noexcept
    {
        if (this->m_allocation_mdspan.data_handle()) {
            m_allocator.deallocate(this->data_handle(), this->size());
        }
    }

    /// Deleted: use deepcopy instead
    Chunk& operator=(Chunk const& other) = delete;

    /** Move-assigns a new value to this field
     * @param other the Chunk to move
     * @return *this
     */
    Chunk& operator=(Chunk&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }
        if (this->m_allocation_mdspan.data_handle()) {
            m_allocator.deallocate(this->data_handle(), this->size());
        }
        static_cast<base_type&>(*this) = std::move(static_cast<base_type&>(other));
        m_allocator = std::move(other.m_allocator);
        m_label = std::move(other.m_label);
        other.m_allocation_mdspan
                = allocation_mdspan_type(nullptr, other.m_allocation_mdspan.mapping());

        return *this;
    }

    /// Slice out some dimensions
    template <class... QueryDDims>
    auto operator[](DiscreteElement<QueryDDims...> const& slice_spec) const
    {
        return view_type(*this)[slice_spec];
    }

    /// Slice out some dimensions
    template <class... QueryDDims>
    auto operator[](DiscreteElement<QueryDDims...> const& slice_spec)
    {
        return span_view()[slice_spec];
    }

    /// Slice out some dimensions
    template <class... QueryDDims>
    auto operator[](DiscreteDomain<QueryDDims...> const& odomain) const
    {
        return span_view()[odomain];
    }

    /// Slice out some dimensions
    template <class... QueryDDims>
    auto operator[](DiscreteDomain<QueryDDims...> const& odomain)
    {
        return span_view()[odomain];
    }

    /** Element access using a list of DiscreteElement
     * @param delems discrete coordinates
     * @return const-reference to this element
     */
    template <
            class... DElems,
            std::enable_if_t<detail::all_of_v<is_discrete_element_v<DElems>...>, int> = 0>
    element_type const& operator()(DElems const&... delems) const noexcept
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
    element_type const& operator()(DVects const&... dvects) const noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DVects::size()),
                "Invalid number of dimensions");
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(discrete_vector_type(dvects...)));
    }

    /** Element access using a list of DiscreteElement
     * @param delems discrete coordinates
     * @return reference to this element
     */
    template <
            class... DElems,
            std::enable_if_t<detail::all_of_v<is_discrete_element_v<DElems>...>, int> = 0>
    element_type& operator()(DElems const&... delems) noexcept
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
    element_type& operator()(DVects const&... dvects) noexcept
    {
        static_assert(
                SupportType::rank() == (0 + ... + DVects::size()),
                "Invalid number of dimensions");
        return DDC_MDSPAN_ACCESS_OP(
                this->m_allocation_mdspan,
                detail::array(discrete_vector_type(dvects...)));
    }

    /** Returns the label of the Chunk
     * @return c-string
     */
    char const* label() const
    {
        return m_label.c_str();
    }

    /** Access to the underlying allocation pointer
     * @return read-only allocation pointer
     */
    ElementType const* data_handle() const
    {
        return base_type::data_handle();
    }

    /** Access to the underlying allocation pointer
     * @return allocation pointer
     */
    ElementType* data_handle()
    {
        return base_type::data_handle();
    }

    /** Provide a mdspan on the memory allocation
     * @return read-only allocation mdspan
     */
    const_allocation_mdspan_type allocation_mdspan() const
    {
        return base_type::allocation_mdspan();
    }

    /** Provide a mdspan on the memory allocation
     * @return allocation mdspan
     */
    allocation_mdspan_type allocation_mdspan()
    {
        return base_type::allocation_mdspan();
    }

    /** Provide an unmanaged `Kokkos::View` on the memory allocation
     * @return allocation `Kokkos::View`
     */
    auto allocation_kokkos_view()
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<SupportType::rank()> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_t<ElementType, SupportType::rank()>,
                decltype(kokkos_layout),
                typename Allocator::memory_space>(s.data_handle(), kokkos_layout);
    }

    /** Provide an unmanaged `Kokkos::View` on the memory allocation
     * @return read-only allocation `Kokkos::View`
     */
    auto allocation_kokkos_view() const
    {
        auto s = this->allocation_mdspan();
        auto kokkos_layout = detail::build_kokkos_layout(
                s.extents(),
                s.mapping(),
                std::make_index_sequence<SupportType::rank()> {});
        return Kokkos::View<
                detail::mdspan_to_kokkos_element_t<ElementType const, SupportType::rank()>,
                decltype(kokkos_layout),
                typename Allocator::memory_space>(s.data_handle(), kokkos_layout);
    }

    view_type span_cview() const
    {
        return view_type(*this);
    }

    view_type span_view() const
    {
        return view_type(*this);
    }

    span_type span_view()
    {
        return span_type(*this);
    }
};

template <class SupportType, class Allocator>
Chunk(std::string const&, SupportType const&, Allocator)
        -> Chunk<typename Allocator::value_type, SupportType, Allocator>;

template <class SupportType, class Allocator>
Chunk(SupportType const&, Allocator)
        -> Chunk<typename Allocator::value_type, SupportType, Allocator>;

} // namespace ddc
