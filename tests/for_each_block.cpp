// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace ddc {

template <class DDim, class Functor>
void host_for_each_block(
        DiscreteDomain<DDim> const& domain,
        std::size_t const nb_blocks,
        Functor const& f) noexcept
{
    std::size_t const block = domain.size() / nb_blocks;
    std::size_t const rem = domain.size() - nb_blocks * block;
    DiscreteElement<DDim> front = domain.front();
    for (std::size_t ib = 0; ib < nb_blocks; ++ib) {
        DiscreteVector<DDim> const size(block + (ib < rem ? 1 : 0));
        f(DiscreteDomain<DDim>(front, size));
        front += size;
    }
}

template <class Support, std::size_t N, class Functor, class... DDoms1d>
void host_for_each_block(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& nb_blocks,
        Functor const& f,
        DDoms1d const&... ddoms) noexcept
{
    static constexpr std::size_t I = sizeof...(DDoms1d);
    if constexpr (I == N) {
        f(Support(ddoms...));
    } else {
        using DDim = ddc::type_seq_element_t<I, ddc::to_type_seq_t<Support>>;
        std::size_t const block = domain.template extent<DDim>() / nb_blocks[I];
        std::size_t const rem = domain.template extent<DDim>() - nb_blocks[I] * block;
        DiscreteElement<DDim> front(domain.front());
        for (std::size_t ib = 0; ib < nb_blocks[I]; ++ib) {
            DiscreteVector<DDim> const size(block + (ib < rem ? 1 : 0));
            host_for_each_block(domain, nb_blocks, f, ddoms..., DiscreteDomain<DDim>(front, size));
            front += size;
        }
    }
}

} // namespace ddc

inline namespace anonymous_namespace_workaround_for_each_block_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(10);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(12);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);

} // namespace anonymous_namespace_workaround_for_each_block_cpp

TEST(ForEachSerialBlockHost, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    ddc::Chunk elems_count("count", dom, ddc::KokkosAllocator<int, Kokkos::HostSpace>());

    for (ddc::DiscreteVectorElement const nb_blocks : {1, 2, 10, 15}) {
        ddc::parallel_fill(elems_count, 0);
        ddc::host_for_each_block(
                dom,
                std::array<ddc::DiscreteVectorElement, 1> {nb_blocks},
                [&](DDomX const domx) {
                    ddc::host_for_each(domx, [&](DElemX ix) { elems_count(ix)++; });
                });
        ddc::host_for_each(dom, [&](DElemX ix) { EXPECT_EQ(elems_count(ix), 1); });
    }
}

TEST(ForEachSerialBlockHost, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    ddc::Chunk elems_count("count", dom, ddc::KokkosAllocator<int, Kokkos::HostSpace>());

    for (ddc::DiscreteVectorElement const nb_blocks : {1, 2, 10, 15}) {
        ddc::parallel_fill(elems_count, 0);
        ddc::host_for_each_block(
                dom,
                std::array<ddc::DiscreteVectorElement, 2> {nb_blocks, nb_blocks},
                [&](DDomXY const domxy) {
                    ddc::host_for_each(domxy, [&](DElemXY ixy) { elems_count(ixy)++; });
                });
        ddc::host_for_each(dom, [&](DElemXY ixy) { EXPECT_EQ(elems_count(ixy), 1); });
    }
}
