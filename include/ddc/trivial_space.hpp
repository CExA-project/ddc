// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

/** Construct a bounded dimension without attributes.
 *
 * @param n number of elements
 * @return a DiscreteDomain of size `n`
 */
template <class DDim>
constexpr DiscreteDomain<DDim> init_trivial_bounded_space(DiscreteVector<DDim> const n) noexcept
{
    return DiscreteDomain<DDim>(DiscreteElement<DDim>(0), n);
}

/** Construct a half bounded dimension without attributes.
 *
 * @return the first DiscreteElement of the dimension
 */
template <class DDim>
constexpr DiscreteElement<DDim> init_trivial_half_bounded_space() noexcept
{
    return DiscreteElement<DDim>(0);
}

} // namespace ddc
