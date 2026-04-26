// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <span>

#include <Kokkos_Core.hpp>

#include "detail/type_seq.hpp"

#include "discrete_element.hpp"
#include "discrete_vector.hpp"

namespace ddc {

namespace detail {

void distribute_blocks(
        std::size_t nb_blocks,
        std::span<DiscreteVectorElement const> sizes,
        std::span<DiscreteVectorElement> nb_blocks_per_dim);

class ComputeBlockFn
{
    DiscreteVectorElement m_quot;

    DiscreteVectorElement m_rem;

public:
    ComputeBlockFn(DiscreteVectorElement extent, DiscreteVectorElement nb_blocks) noexcept;

    DiscreteVectorElement operator()(DiscreteVectorElement i) const noexcept;
};

template <class Support, std::size_t N, class Functor, class... Doms1d>
void host_for_each_block(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& nb_blocks_per_dim,
        Functor const& f,
        Doms1d const&... doms1d) noexcept
{
    static constexpr std::size_t I = sizeof...(Doms1d);
    if constexpr (I == N) {
        f(Support(doms1d...));
    } else {
        using DDim = ddc::type_seq_element_t<I, ddc::to_type_seq_t<Support>>;
        ComputeBlockFn const compute_block(domain.template extent<DDim>(), nb_blocks_per_dim[I]);
        typename Rebind<Support, TypeSeq<DDim>>::type dom1d(domain);
        for (DiscreteVectorElement ib = 0; ib < nb_blocks_per_dim[I]; ++ib) {
            DiscreteVector<DDim> const block(compute_block(ib));
            host_for_each_block(domain, nb_blocks_per_dim, f, doms1d..., dom1d.take_first(block));
            dom1d = dom1d.remove_first(block);
        }
    }
}

} // namespace detail

template <class Support, class Functor>
void host_for_each_block(
        Support const& domain,
        typename Support::discrete_vector_type nb_blocks_per_dim,
        Functor const& f) noexcept
{
    detail::host_for_each_block(domain, detail::array(nb_blocks_per_dim), f);
}

template <class Support, class Functor>
void host_for_each_block(Support const& domain, std::size_t nb_blocks, Functor const& f) noexcept
{
    typename Support::discrete_vector_type nb_blocks_per_dim {};
    detail::distribute_blocks(
            nb_blocks,
            detail::array(domain.extents()),
            detail::array(nb_blocks_per_dim));
    host_for_each_block(domain, nb_blocks_per_dim, f);
}

} // namespace ddc
