// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>

#include "coordinate.hpp"
#include "real_type.hpp"
#include "uniform_point_sampling.hpp"

namespace ddc::detail {

void print_uniform_point_sampling(std::ostream& os, CoordinateElement const origin, Real const step)
{
    os << "UniformPointSampling(origin=" << origin << ", step=" << step << ')';
}

} // namespace ddc::detail
