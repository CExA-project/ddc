// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

namespace ddc {

/**
 * @brief A templated struct representing a discrete dimension storing
 * the derivatives of a function along a continuous dimension CDim.
 */
template <class CDim>
struct Deriv : ddc::DiscreteDimension
{
};

} // namespace ddc
