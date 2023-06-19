// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "ddc/chunk.hpp"
#include "ddc/chunk_span.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/distribution.hpp"

namespace ddc {

// similar to dash::Array
template <
        class ElementType,
        class Domain,
        class,
        class,
        class = std::experimental::layout_right,
        class = typename Chunk<ElementType, Domain>::allocator_type>
class DistributedField;

template <
        class ElementType,
        class Domain,
        class,
        class,
        class = std::experimental::layout_right,
        class = typename ChunkSpan<ElementType, Domain>::memory_space>
class DistributedFieldSpan;

template <class, class, class, class, class, class>
class DistributedFieldCommon;

template <
        class ElementType,
        class... DDims,
        class Chunking,
        class Distribution,
        class LayoutPolicy,
        class MemorySpace>
class DistributedFieldCommon<
        ElementType,
        DiscreteDomain<DDims...>,
        Chunking,
        Distribution,
        LayoutPolicy,
        MemorySpace>
{
    using element_type = ElementType;

    using mdomain_type = DiscreteDomain<DDims...>;

    using layout_policy_type = LayoutPolicy;

    using chunking_type = Chunking;

    using distribution_type = Distribution;

    using memory_space = MemorySpace;

    using discrete_element_type = typename mdomain_type::discrete_element_type;

    using chunked_domain_type =
            typename chunking_type::template chunked_domain_type<DDims...>;

    using rank_id_type = typename distribution_type::rank_id_type;

    using mlength_type = typename chunked_domain_type::mlength_type;

    using chunk_id_type = typename chunked_domain_type::chunk_id_type;

    using span_type = DistributedFieldSpan<
            element_type,
            mdomain_type,
            chunking_type,
            distribution_type,
            layout_policy_type,
            memory_space>;

    using view_type = DistributedFieldSpan<
            element_type const,
            mdomain_type,
            chunking_type,
            distribution_type,
            layout_policy_type,
            memory_space>;

    using distributed_domain_type
            = DistributedDomain<mdomain_type, chunking_type, distribution_type>;

    using chunk_span_type
            = ddc::ChunkSpan<element_type, mdomain_type, layout_policy_type, memory_space>;

    using chunk_view_type = typename chunk_span_type::view_type;

    using extents_type = typename chunk_span_type::extents_type;

    using layout_type = typename chunk_span_type::layout_type;

    using accessor_type = typename chunk_span_type::accessor_type;

    using mapping_type = typename chunk_span_type::mapping_type;

    using value_type = typename chunk_span_type::value_type;

    using size_type = typename chunk_span_type::size_type;

    using data_handle_type = typename chunk_span_type::data_handle_type;

    using reference = typename chunk_span_type::reference;

    template <class, class, class, class, class, class>
    friend class DistributedFieldCommon;

    template <class, class, class, class, class, class>
    friend class DistributedField;

    template <class, class, class, class, class, class>
    friend class DistributedFieldSpan;

private:
    distributed_domain_type m_domain;

public:
    static constexpr int rank() noexcept
    {
        return extents_type::rank();
    }

    static constexpr int rank_dynamic() noexcept
    {
        return extents_type::rank_dynamic();
    }

    static constexpr size_type static_extent(std::size_t r) noexcept
    {
        return extents_type::static_extent(r);
    }

    static constexpr bool is_always_unique() noexcept
    {
        return mapping_type::is_always_unique();
    }

    static constexpr bool is_always_exhaustive() noexcept
    {
        return mapping_type::is_always_exhaustive();
    }

    static constexpr bool is_always_strided() noexcept
    {
        return mapping_type::is_always_strided();
    }

public:
    constexpr DiscreteVector<DDims...> extents() const noexcept
    {
        return domain().extents();
    }

    template <class QueryDDim>
    constexpr size_type extent() const noexcept
    {
        return domain().template extent<QueryDDim>();
    }

    constexpr size_type size() const noexcept
    {
        return domain().size();
    }

    /** Provide access to the domain on which this DistributedField is defined
     * @return the domain on which this DistributedField is defined
     */
    constexpr distributed_domain_type const& domain() const noexcept
    {
        return m_domain;
    }

    /** Provide access to the domain on which this DistributedField is defined
     * @return the domain on which this DistributedField is defined
     */
    template <class... QueryDDims>
    constexpr DiscreteDomain<QueryDDims...> domain() const noexcept
    {
        return select<QueryDDims...>(domain());
    }


protected:
    /// Empty DistributedFieldCommon
    constexpr DistributedFieldCommon() = default;

    /** Constructs a new DistributedFieldCommon from scratch
     * @param domain
     */
    constexpr DistributedFieldCommon(distributed_domain_type const& domain) noexcept
        : m_domain(domain)
    {
    }

    constexpr DistributedFieldCommon(
            mdomain_type const& global_domain,
            chunking_type const& chunking,
            distribution_type const& distribution)
        : m_domain(chunking(global_domain), distribution)
    {
    }

    constexpr DistributedFieldCommon(
            chunked_domain_type const& chunked_domain,
            distribution_type const& distribution)
        : m_domain(chunked_domain, distribution)
    {
    }

    /** Constructs a new DistributedFieldCommon by copy
     * @param other the DistributedFieldCommon to copy
     */
    constexpr DistributedFieldCommon(DistributedFieldCommon const& other) = default;

    /** Constructs a new DistributedFieldCommon by move
     * @param other the DistributedFieldCommon to move
     */
    constexpr DistributedFieldCommon(DistributedFieldCommon&& other) = default;

    /** Copy-assigns a new value to this DistributedFieldCommon
     * @param other the DistributedFieldCommon to copy
     * @return *this
     */
    constexpr DistributedFieldCommon& operator=(DistributedFieldCommon const& other) = default;

    /** Move-assigns a new value to this DistributedFieldCommon
     * @param other the DistributedFieldCommon to move
     * @return *this
     */
    constexpr DistributedFieldCommon& operator=(DistributedFieldCommon&& other) = default;
};

template <
        class ElementType,
        class... DDims,
        class Chunking,
        class Distribution,
        class LayoutPolicy,
        class Allocator>
class DistributedField<
        ElementType,
        DiscreteDomain<DDims...>,
        Chunking,
        Distribution,
        LayoutPolicy,
        Allocator>
    : public DistributedFieldCommon<
              ElementType,
              DiscreteDomain<DDims...>,
              Chunking,
              Distribution,
              LayoutPolicy,
              typename Allocator::memory_space>
{
public:
    using base_type = DistributedFieldCommon<
            ElementType,
            DiscreteDomain<DDims...>,
            Chunking,
            Distribution,
            LayoutPolicy,
            typename Allocator::memory_space>;

    using allocator_type = Allocator;

    using typename base_type::element_type;

    using typename base_type::mdomain_type;

    using typename base_type::layout_policy_type;

    using typename base_type::chunking_type;

    using typename base_type::distribution_type;

    using typename base_type::memory_space;

    using typename base_type::discrete_element_type;

    using typename base_type::chunked_domain_type;

    using typename base_type::rank_id_type;

    using typename base_type::mlength_type;

    using typename base_type::chunk_id_type;

    using typename base_type::span_type;

    using typename base_type::view_type;

    using typename base_type::distributed_domain_type;

    using typename base_type::chunk_span_type;

    using typename base_type::chunk_view_type;

    using typename base_type::extents_type;

    using typename base_type::layout_type;

    using typename base_type::accessor_type;

    using typename base_type::mapping_type;

    using typename base_type::value_type;

    using typename base_type::size_type;

    using typename base_type::data_handle_type;

    using typename base_type::reference;

    using chunk_type = ddc::Chunk<element_type, mdomain_type, allocator_type>;

private:
    chunk_type m_local_chunk;

public:
    static constexpr int rank() noexcept
    {
        return chunk_type::rank();
    }

public:
    /// Construct a DistributedField with uninitialized values
    explicit DistributedField(
            mdomain_type const& global_domain,
            chunking_type const& chunking,
            distribution_type const& distribution,
            Allocator allocator = Allocator())
        : base_type(global_domain, chunking, distribution)
        , m_local_chunk(this->domain().local_subdomain(), allocator)
    {
    }

    /// Construct a DistributedField with uninitialized values
    explicit DistributedField(
            chunked_domain_type const& chunked_domain,
            distribution_type const& distribution,
            Allocator allocator = Allocator())
        : base_type(chunked_domain, distribution)
        , m_local_chunk(this->domain().local_subdomain(), allocator)
    {
    }

    template <class... Args>
    inline element_type const& operator()(Args&&... delem) const noexcept
    {
        return m_local_chunk(std::forward<Args...>(delem...));
    }

    template <class... Args>
    inline element_type& operator()(Args&&... delem) noexcept
    {
        return m_local_chunk(std::forward<Args...>(delem...));
    }

    inline chunk_view_type local_chunk_cview() const noexcept
    {
        return m_local_chunk;
    }

    inline chunk_view_type local_chunk_view() const noexcept
    {
        return m_local_chunk;
    }

    inline chunk_span_type local_chunk_view() noexcept
    {
        return m_local_chunk;
    }

    constexpr inline view_type span_cview() const noexcept
    {
        return view_type(*this);
    }

    constexpr inline view_type span_view() const noexcept
    {
        return view_type(*this);
    }

    constexpr inline span_type span_view() noexcept
    {
        return span_type(*this);
    }
};


template <
        class ElementType,
        class LayoutPolicy = std::experimental::layout_right,
        class... DDims,
        class Chunking,
        class Distribution,
        class Allocator = typename Chunk<ElementType, DiscreteDomain<DDims...>>::allocator_type>
DistributedField<
        ElementType,
        DiscreteDomain<DDims...>,
        Chunking,
        Distribution,
        LayoutPolicy,
        Allocator>
distributed_field(
        DiscreteDomain<DDims...> const& global_domain,
        Chunking const& chunking,
        Distribution const& distribution,
        Allocator allocator = Allocator())
{
    return DistributedField<
            ElementType,
            DiscreteDomain<DDims...>,
            Chunking,
            Distribution,
            LayoutPolicy,
            Allocator>(global_domain, chunking, distribution, allocator);
}

template <
        class ElementType,
        class LayoutPolicy = std::experimental::layout_right,
        class... DDims,
        class Chunking,
        class Distribution,
        class Allocator = typename Chunk<ElementType, DiscreteDomain<DDims...>>::allocator_type>
DistributedField<
        ElementType,
        DiscreteDomain<DDims...>,
        Chunking,
        Distribution,
        LayoutPolicy,
        Allocator>
distributed_field(
        ChunkedDomain<DiscreteDomain<DDims...>, Chunking> const& chunked_domain,
        Distribution const& distribution,
        Allocator allocator = Allocator())
{
    return DistributedField<
            ElementType,
            DiscreteDomain<DDims...>,
            Chunking,
            Distribution,
            LayoutPolicy,
            Allocator>(chunked_domain, distribution, allocator);
}

template <
        class ElementType,
        class... DDims,
        class Chunking,
        class Distribution,
        class LayoutPolicy,
        class MemorySpace>
class DistributedFieldSpan<
        ElementType,
        DiscreteDomain<DDims...>,
        Chunking,
        Distribution,
        LayoutPolicy,
        MemorySpace>
    : public DistributedFieldCommon<
              ElementType,
              DiscreteDomain<DDims...>,
              Chunking,
              Distribution,
              LayoutPolicy,
              MemorySpace>
{
public:
    using base_type = DistributedFieldCommon<
            ElementType,
            DiscreteDomain<DDims...>,
            Chunking,
            Distribution,
            LayoutPolicy,
            MemorySpace>;

    using typename base_type::element_type;

    using typename base_type::mdomain_type;

    using typename base_type::layout_policy_type;

    using typename base_type::chunking_type;

    using typename base_type::distribution_type;

    using typename base_type::memory_space;

    using typename base_type::discrete_element_type;

    using typename base_type::chunked_domain_type;

    using typename base_type::rank_id_type;

    using typename base_type::mlength_type;

    using typename base_type::chunk_id_type;

    using typename base_type::span_type;

    using typename base_type::view_type;

    using typename base_type::distributed_domain_type;

    using typename base_type::chunk_span_type;

    using typename base_type::chunk_view_type;

    using typename base_type::extents_type;

    using typename base_type::layout_type;

    using typename base_type::accessor_type;

    using typename base_type::mapping_type;

    using typename base_type::value_type;

    using typename base_type::size_type;

    using typename base_type::data_handle_type;

    using typename base_type::reference;

private:
    chunk_span_type m_local_chunk;

public:
    static constexpr int rank() noexcept
    {
        return chunk_span_type::rank();
    }

    /// Empty DistributedFieldSpan
    constexpr DistributedFieldSpan() = default;

    /** Constructs a new DistributedFieldSpan by copy, yields a new view to the same data
     * @param other the DistributedFieldSpan to copy
     */
    constexpr DistributedFieldSpan(DistributedFieldSpan const& other) = default;

    /** Constructs a new DistributedFieldSpan by move
     * @param other the DistributedFieldSpan to move
     */
    constexpr DistributedFieldSpan(DistributedFieldSpan&& other) = default;

    /** Constructs a new DistributedFieldSpan from a modifiable DistributedField
     * @param other the DistributedField to view
     */
    template <class OElementType, class Allocator>
    constexpr DistributedFieldSpan(DistributedField<
                                   OElementType,
                                   mdomain_type,
                                   chunking_type,
                                   distribution_type,
                                   layout_policy_type,
                                   Allocator>& field) noexcept
        : base_type(field.domain())
        , m_local_chunk(field.local_chunk_view())
    {
    }

    /** Constructs a new non modifier (const ElementType) DistributedFieldSpan from a DistributedField
     * @param other the DistributedField to view
     */
    // Disabled by SFINAE in case `ElementType` is not `const` to avoid write access
    template <
            class OElementType,
            class Allocator,
            class ElementTypeSFINAE = ElementType,
            class = std::enable_if_t<std::is_const_v<ElementTypeSFINAE>>>
    constexpr DistributedFieldSpan(DistributedField<
                                   OElementType,
                                   mdomain_type,
                                   chunking_type,
                                   distribution_type,
                                   layout_policy_type,
                                   Allocator> const& field) noexcept
        : base_type(field.domain())
        , m_local_chunk(field.local_chunk_cview())
    {
    }

    /** Constructs a new DistributedFieldSpan by copy of a DistributedFieldSpan
     * @param other the DistributedFieldSpan to copy
     */
    template <class OElementType>
    constexpr DistributedFieldSpan(DistributedFieldSpan<
                                   OElementType,
                                   mdomain_type,
                                   chunking_type,
                                   distribution_type,
                                   layout_policy_type,
                                   memory_space> const& field) noexcept
        : base_type(field.domain())
        , m_local_chunk(field.local_chunk())
    {
    }

    /** Constructs a new DistributedFieldSpan from scratch
     * @param local_chunk
     * @param domain the domain that sustains the view
     */
    constexpr DistributedFieldSpan(chunk_span_type const& local_chunk, mdomain_type const& domain)
        : base_type(domain)
        , m_local_chunk(local_chunk)
    {
    }

    /** Slice out some dimensions
     */
    template <class... QueryDDims>
    constexpr auto operator[](DiscreteElement<QueryDDims...> const& slice_spec) const;

    /** Slice out some dimensions
     */
    template <class... QueryDDims>
    constexpr auto operator[](ChunkedDomain<QueryDDims...> const& odomain) const;

    template <class... Args>
    element_type& operator()(Args&&... delem) const noexcept
    {
        return m_local_chunk(std::forward<Args>(delem)...);
    }

    inline chunk_view_type local_chunk_cview() const noexcept
    {
        return m_local_chunk;
    }

    inline chunk_span_type const& local_chunk_view() noexcept
    {
        return m_local_chunk;
    }

    constexpr inline view_type span_cview() const noexcept
    {
        return view_type(*this);
    }

    constexpr inline span_type span_view() const noexcept
    {
        return *this;
    }
};

} // namespace ddc
