// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>

#include "coordinate.hpp"
#include "periodic_sampling.hpp"
#include "real_type.hpp"

namespace ddc::detail {

void print_periodic_sampling(std::ostream& os, CoordinateElement const origin, Real const step)
{
    os << "PeriodicSampling(origin=" << origin << ", step=" << step << ')';
}

} // namespace ddc::detail
