// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "discrete_vector.hpp"
#include "for_each_block.hpp"

namespace {

bool is_power_of_2(std::size_t const n) noexcept
{
    return n > 0 && !(n & (n - 1));
}

} // namespace

namespace ddc::detail {

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
        if (dim >= sizes.size()) {
            throw std::runtime_error("DDC expects a smaller number of blocks.");
        }

        if (sizes[dim] >= nb_blocks_per_dim[dim] * 2) {
            nb_blocks_per_dim[dim] *= 2;
            nb_blocks /= 2;
        } else {
            ++dim;
        }
    }
}

ComputeBlockFn::ComputeBlockFn(
        DiscreteVectorElement const extent,
        DiscreteVectorElement const nb_blocks) noexcept
    : m_quot(extent / nb_blocks)
    , m_rem(extent % nb_blocks)
{
}

DiscreteVectorElement ComputeBlockFn::operator()(DiscreteVectorElement const i) const noexcept
{
    if (i < m_rem) {
        return m_quot + 1;
    }
    return m_quot;
}

} // namespace ddc::detail
