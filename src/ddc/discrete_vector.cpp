// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <span>

#include "discrete_vector.hpp"

namespace ddc::detail {

void print_discrete_vector(std::ostream& os, std::span<DiscreteVectorElement const> const view)
{
    os << '(';
    if (!view.empty()) {
        os << view.front();
        for (DiscreteVectorElement const value : view.subspan(1)) {
            os << ", " << value;
        }
    }
    os << ')';
}

} // namespace ddc::detail
