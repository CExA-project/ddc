// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>

#include "ddc/discrete_vector.hpp"
#include "ddc/for_each.hpp"

namespace ddc {

template <class, class>
class ChunkedDomain;

template <size_t NDim>
class BlockChunking;

/** A Chunking takes a DiscreteDomain and splits it into multiple chunks, a ChunkedDomain
 */
template <size_t NDim>
class BlockChunking
{
public:
    using nd_size_type = std::array<size_t, NDim>;

    template <class... DDims>
    using chunked_domain_type = ChunkedDomain<DiscreteDomain<DDims...>, BlockChunking>;

    nd_size_type m_nb_chunks;

    nd_size_type m_overlap_before;

    nd_size_type m_overlap_after;

    BlockChunking(nd_size_type nb_chunks, nd_size_type overlap_before, nd_size_type overlap_after)
        : m_nb_chunks(nb_chunks)
        , m_overlap_before(overlap_before)
        , m_overlap_after(overlap_after)
    {
    }

    BlockChunking(nd_size_type nb_chunks, nd_size_type overlap = {0})
        : m_nb_chunks(nb_chunks)
        , m_overlap_before(overlap)
        , m_overlap_after(overlap)
    {
    }

    template <size_t Idx, class... DDims>
    using ddim_for_t = std::tuple_element_t<Idx, std::tuple<DDims...>>;

    template <class... DDims, size_t... Idxs>
    static inline constexpr DiscreteVector<DDims...> array_to_vector(
            std::array<size_t, sizeof...(DDims)> a,
            std::index_sequence<Idxs...>)
    {
        static_assert(sizeof...(DDims) == sizeof...(Idxs), "");
        return DiscreteVector<DDims...>(DiscreteVector<ddim_for_t<Idxs, DDims...>>(a[Idxs])...);
    }

    template <class... DDims>
    chunked_domain_type<DDims...> operator()(DiscreteDomain<DDims...> const& global_domain) const
    {
        auto&& overlap_before
                = array_to_vector<DDims...>(m_overlap_before, std::index_sequence_for<DDims...>());
        auto&& overlap_after
                = array_to_vector<DDims...>(m_overlap_after, std::index_sequence_for<DDims...>());
        auto&& nb_chunks
                = array_to_vector<DDims...>(m_nb_chunks, std::index_sequence_for<DDims...>());
        return {global_domain, nb_chunks, overlap_before, overlap_after};
    }

    template <class DDom0, class... DDoms>
    auto operator()(DDom0 const& dom0, DDoms... doms) const
    {
        return (*this)(DiscreteDomain(dom0, doms...));
    }
};

template <size_t NDim>
BlockChunking(
        std::array<size_t, NDim> nb_chunks,
        std::array<size_t, NDim> overlap_before,
        std::array<size_t, NDim> overlap_after) -> BlockChunking<NDim>;

template <size_t NDim>
BlockChunking(std::array<size_t, NDim> nb_chunks, std::array<size_t, NDim> overlap = {0})
        -> BlockChunking<NDim>;

template <class... DDims>
class ChunkedDomain<DiscreteDomain<DDims...>, BlockChunking<sizeof...(DDims)>>
{
public:
    using mdomain_type = DiscreteDomain<DDims...>;

    using discrete_element_type = typename mdomain_type::discrete_element_type;

    using mlength_type = typename mdomain_type::mlength_type;

    using chunk_id_type = std::size_t;

    using chunk_mdid_type = DiscreteVector<DDims...>;

    template <size_t id>
    using dim_from_id_t = std::tuple_element_t<id, std::tuple<DDims...>>;

    template <class>
    struct ToDim;

    mdomain_type m_global_domain;

    mlength_type m_chunk_size;

    mlength_type m_overlap_before;

    mlength_type m_overlap_after;

    static constexpr std::size_t rank()
    {
        return sizeof...(DDims);
    }

    template <class Tag0, class... Tags, class IntegralType>
    static constexpr inline std::enable_if_t<
            std::is_integral_v<IntegralType>,
            std::tuple<DiscreteVector<Tag0, Tags...>, IntegralType>>
    modulo(IntegralType value, DiscreteVector<Tag0, Tags...> const& max)
    {
        if constexpr (sizeof...(Tags) == 0) {
            auto&& rest = value % get<Tag0>(max);
            auto&& res = DiscreteVector<Tag0>(value / get<Tag0>(max));
            return {res, rest};
        } else {
            auto&& [prev_res, prev_rest] = modulo(value, select<Tags...>(max));
            auto&& rest = prev_rest % get<Tag0>(max);
            auto&& res = DiscreteVector<Tag0>(prev_rest / get<Tag0>(max));
            return {DiscreteVector<Tag0, Tags...> {res, prev_res}, rest};
        }
    }

    ChunkedDomain(
            mdomain_type global_domain,
            mlength_type chunk_size,
            mlength_type overlap_before,
            mlength_type overlap_after)
        : m_global_domain(global_domain)
        , m_chunk_size(chunk_size)
        , m_overlap_before(overlap_before)
        , m_overlap_after(overlap_after)
    {
    }

    ChunkedDomain(mdomain_type global_domain, mlength_type chunk_size, mlength_type overlap = 0)
        : m_global_domain(global_domain)
        , m_chunk_size(chunk_size)
        , m_overlap_before(overlap)
        , m_overlap_after(overlap)
    {
    }

    inline constexpr chunk_mdid_type to_md(chunk_id_type chunk_id) const
    {
        auto&& nb_chunks = m_global_domain.extents() / m_chunk_size;
        auto&& [result, rest] = modulo(chunk_id, nb_chunks);
        if (rest != 0) {
            // ERROR
        }
        return result;
    }

    inline constexpr mlength_type first_elem(chunk_mdid_type chunk_id) const
    {
        return chunk_id * m_chunk_size;
    }

    inline constexpr mlength_type first_elem(chunk_id_type chunk_id) const
    {
        return first_elem(to_md(chunk_id));
    }

    inline constexpr mdomain_type domain_of(chunk_mdid_type chunk_id) const
    {
        auto&& elems_before = first_elem(chunk_id);
        auto&& elems_before_ghosts = elems_before - m_overlap_before;
        auto&& size_with_ghosts = m_chunk_size + m_overlap_after;
        return m_global_domain.remove_first(elems_before_ghosts).take_first(size_with_ghosts);
    }

    inline constexpr mdomain_type domain_of(chunk_id_type chunk_id) const
    {
        return domain_of(to_md(chunk_id));
    }
};

template <class, class, class>
class DistributedDomain;

template <class... Dims, class Chunking, class Distribution>
class DistributedDomain<DiscreteDomain<Dims...>, Chunking, Distribution>
{
public:
    using chunking_type = Chunking;

    using distribution_type = Distribution;

    using chunked_domain_type = typename chunking_type::template chunked_domain_type<Dims...>;

    using mdomain_type = typename chunked_domain_type::mdomain_type;

    using discrete_element_type = typename chunked_domain_type::discrete_element_type;

    using mlength_type = typename chunked_domain_type::mlength_type;

    using chunk_id_type = typename chunked_domain_type::chunk_id_type;

    using rank_id_type = typename distribution_type::rank_id_type;

    chunked_domain_type m_domain;

    distribution_type m_distribution;

    DistributedDomain(
            mdomain_type const& global_domain,
            chunking_type const& chunking,
            distribution_type const& distribution)
        : m_domain(chunking(global_domain))
        , m_distribution(distribution)
    {
    }

    DistributedDomain(
            chunked_domain_type const& chunked_domain,
            distribution_type const& distribution)
        : m_domain(chunked_domain)
        , m_distribution(distribution)
    {
    }

    mdomain_type subdomain_in(rank_id_type rank) const
    {
        return m_domain.domain_of(m_distribution.chunk_in(rank));
    }

    mdomain_type local_subdomain() const
    {
        return m_domain.domain_of(m_distribution.local_chunk_id());
    }
};

/** A DistributionPolicy takes a ChunkedDomain and distributes it over the various storage devices
 */
class DirectDistribution
{
public:
    using chunk_id_type = std::size_t;

    using rank_id_type = std::size_t;

    rank_id_type rank_of(chunk_id_type chunk_id) const
    {
        return chunk_id;
    }

    chunk_id_type chunk_in(rank_id_type rank) const
    {
        return rank;
    }

    rank_id_type local_rank() const
    {
        return 0;
    }

    chunk_id_type local_chunk_id() const
    {
        return chunk_in(local_rank());
    }
    using distribution_type = DirectDistribution;
};

/** iterates over a nD distributed domain using the provided execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class ExecutionPolicy, class... Dims, class Chunking, class Distribution, class Functor>
inline void for_each(
        ExecutionPolicy&& p,
        DistributedDomain<DiscreteDomain<Dims...>, Chunking, Distribution> const& domain,
        Functor&& f)
{
    for_each(std::forward<ExecutionPolicy>(p), domain.local_subdomain(), std::forward<Functor>(f));
}

/** iterates over a nD distributed domain using the default execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... Dims, class Chunking, class Distribution, class Functor>
inline void for_each(
        DistributedDomain<DiscreteDomain<Dims...>, Chunking, Distribution> const& domain,
        Functor&& f)
{
    for_each(default_policy(), domain, std::forward<Functor>(f));
}

} // namespace ddc
