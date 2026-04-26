// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <ostream>

#include "non_uniform_point_sampling.hpp"

namespace ddc::detail {

void print_non_uniform_point_samplig(std::ostream& os, std::size_t const size)
{
    os << "NonUniformPointSampling(" << size << ')';
}

} // namespace ddc::detail
