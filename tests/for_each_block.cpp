// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <list>

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

} // namespace ddc

inline namespace anonymous_namespace_workaround_for_each_block_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(10);

} // namespace anonymous_namespace_workaround_for_each_block_cpp

TEST(ForEachSerialBlockHost, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> const view(storage.data(), dom);

    std::list<DElemX> list_elems_ref;
    ddc::host_for_each(dom, [&](DElemX ix) { list_elems_ref.emplace_back(ix); });

    for (std::size_t const block_size : {1, 2, 10, 15}) {
        std::list<DElemX> list_elems;
        ddc::host_for_each_block(dom, block_size, [&](DDomX const domx) {
            ddc::host_for_each(domx, [&](DElemX ix) { list_elems.emplace_back(ix); });
        });
        EXPECT_EQ(list_elems_ref, list_elems);
    }
}
