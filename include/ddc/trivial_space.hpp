// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

/** Construct a dimension without attributes.
 *
 * @param n number of elements
 * @return a DiscreteDomain of size `n`
 */
template <class DDim>
DiscreteDomain<DDim> init_trivial_space(DiscreteVector<DDim> const n) noexcept
{
    return DiscreteDomain<DDim>(DiscreteElement<DDim>(0), n);
}

} // namespace ddc
