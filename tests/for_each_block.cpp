// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace ddc {

namespace detail {

constexpr bool is_power_of_2(std::size_t const n) noexcept
{
    return n > 0 && !(n & (n - 1));
}

void distribute_blocks(
        std::size_t nb_blocks,
        std::span<DiscreteVectorElement const> const sizes,
        std::span<DiscreteVectorElement> const nb_blocks_per_dim)
{
    assert(sizes.size() == nb_blocks_per_dim.size());

    if (!is_power_of_2(nb_blocks)) {
        throw std::runtime_error("DDC distribute_blocks expects a power of 2.");
    }

    for (DiscreteVectorElement& blocks : nb_blocks_per_dim) {
        blocks = 1;
    }

    std::size_t dim = 0;
    while (nb_blocks != 1) {
        if (sizes[dim] >= nb_blocks_per_dim[dim] * 2) {
            nb_blocks_per_dim[dim] *= 2;
            nb_blocks /= 2;
        } else if (dim < sizes.size()) {
            ++dim;
        } else {
            throw std::runtime_error("what the hell");
        }
    }
}

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

template <class Support, std::size_t N, class Functor, class... DDoms1d>
void host_for_each_block(
        Support const& domain,
        std::array<DiscreteVectorElement, N> const& nb_blocks_per_dim,
        Functor const& f) noexcept
{
    host_for_each_block_impl(domain, nb_blocks_per_dim, f);
}

} // namespace detail

template <class Support, class Functor, class... DDoms1d>
void host_for_each_block(Support const& domain, std::size_t nb_blocks, Functor const& f) noexcept
{
    std::array<DiscreteVectorElement, Support::rank()> nb_blocks_per_dim {};
    detail::distribute_blocks(nb_blocks, detail::array(domain.extents()), nb_blocks_per_dim);
    detail::host_for_each_block_impl(domain, nb_blocks_per_dim, f);
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

TEST(ForEachSerialBlockHost, IsPowerOf2)
{
    EXPECT_FALSE(ddc::detail::is_power_of_2(0));
    EXPECT_TRUE(ddc::detail::is_power_of_2(1));
    EXPECT_TRUE(ddc::detail::is_power_of_2(2));
    EXPECT_FALSE(ddc::detail::is_power_of_2(3));
    EXPECT_TRUE(ddc::detail::is_power_of_2(4));
}

TEST(DistributeBlocks, D)
{
    {
        std::array<ddc::DiscreteVectorElement, 1> size {10};
        std::array<ddc::DiscreteVectorElement, 1> nb_blocks_per_dim;
        ddc::detail::distribute_blocks(2, size, nb_blocks_per_dim);
        EXPECT_EQ(nb_blocks_per_dim, (std::array<ddc::DiscreteVectorElement, 1> {2}));
    }
    {
        std::array<ddc::DiscreteVectorElement, 1> size {10};
        std::array<ddc::DiscreteVectorElement, 1> nb_blocks_per_dim;
        ddc::detail::distribute_blocks(8, size, nb_blocks_per_dim);
        EXPECT_EQ(nb_blocks_per_dim, (std::array<ddc::DiscreteVectorElement, 1> {8}));
    }
    {
        std::array<ddc::DiscreteVectorElement, 3> size {3, 4, 5};
        std::array<ddc::DiscreteVectorElement, 3> nb_blocks_per_dim;
        ddc::detail::distribute_blocks(32, size, nb_blocks_per_dim);
        EXPECT_EQ(nb_blocks_per_dim, (std::array<ddc::DiscreteVectorElement, 3> {2, 4, 4}));
    }
}

TEST(ForEachSerialBlockHost, OneDimension)
{
    for (ddc::DiscreteVectorElement const nb_blocks : {1, 2, 4, 8}) {
        DDomX const dom(lbound_x, nelems_x);
        ddc::Chunk elems_count("count", dom, ddc::KokkosAllocator<int, Kokkos::HostSpace>());
        int measured_nb_blocks = 0;
        ddc::parallel_fill(elems_count, 0);
        ddc::host_for_each_block(dom, nb_blocks, [&](DDomX const domx) {
            ddc::host_for_each(domx, [&](DElemX ix) { elems_count(ix)++; });
            ++measured_nb_blocks;
        });
        ddc::host_for_each(dom, [&](DElemX ix) { EXPECT_EQ(elems_count(ix), 1); });
        EXPECT_EQ(measured_nb_blocks, nb_blocks);
    }
}

TEST(ForEachSerialBlockHost, TwoDimensions)
{
    for (ddc::DiscreteVectorElement const nb_blocks : {1, 2, 4, 8}) {
        DDomXY const dom(lbound_x_y, nelems_x_y);
        ddc::Chunk elems_count("count", dom, ddc::KokkosAllocator<int, Kokkos::HostSpace>());
        int measured_nb_blocks = 0;
        ddc::parallel_fill(elems_count, 0);
        ddc::host_for_each_block(dom, nb_blocks, [&](DDomXY const domxy) {
            ddc::host_for_each(domxy, [&](DElemXY ixy) { elems_count(ixy)++; });
            ++measured_nb_blocks;
        });
        ddc::host_for_each(dom, [&](DElemXY ixy) { EXPECT_EQ(elems_count(ixy), 1); });
        EXPECT_EQ(measured_nb_blocks, nb_blocks);
    }
}
