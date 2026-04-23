// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <span>

#include <Kokkos_Core.hpp>

#include "detail/type_seq.hpp"

#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"

namespace ddc {

namespace detail {

void distribute_blocks(
        std::size_t nb_blocks,
        std::span<DiscreteVectorElement const> sizes,
        std::span<DiscreteVectorElement> nb_blocks_per_dim);

template <class Support, std::size_t N, class Functor, class... DDoms1d>
void host_for_each_block_impl(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& nb_blocks_per_dim,
        Functor const& f,
        DDoms1d const&... ddoms) noexcept
{
    static constexpr std::size_t I = sizeof...(DDoms1d);
    if constexpr (I == N) {
        f(Support(ddoms...));
    } else {
        using DDim = ddc::type_seq_element_t<I, ddc::to_type_seq_t<Support>>;
        DiscreteVectorElement const block = domain.template extent<DDim>() / nb_blocks_per_dim[I];
        DiscreteVectorElement const rem
                = domain.template extent<DDim>() - nb_blocks_per_dim[I] * block;
        DiscreteElement<DDim> front(domain.front());
        for (DiscreteVectorElement ib = 0; ib < nb_blocks_per_dim[I]; ++ib) {
            DiscreteVector<DDim> const size(block + (ib < rem ? 1 : 0));
            host_for_each_block_impl(
                    domain,
                    nb_blocks_per_dim,
                    f,
                    ddoms...,
                    DiscreteDomain<DDim>(front, size));
            front += size;
        }
    }
}

template <class Support, std::size_t N, class Functor>
void host_for_each_block(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& nb_blocks_per_dim,
        Functor const& f) noexcept
{
    host_for_each_block_impl(domain, nb_blocks_per_dim, f);
}

} // namespace detail

template <class Support, class Functor>
void host_for_each_block(Support const& domain, std::size_t nb_blocks, Functor const& f) noexcept
{
    std::array<DiscreteVectorElement, Support::rank()> nb_blocks_per_dim {};
    detail::distribute_blocks(nb_blocks, detail::array(domain.extents()), nb_blocks_per_dim);
    detail::host_for_each_block_impl(domain, nb_blocks_per_dim, f);
}

} // namespace ddc
