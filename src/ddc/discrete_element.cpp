// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <ostream>

#include "discrete_element.hpp"

namespace ddc::detail {

void print_discrete_element(
        std::ostream& os,
        DiscreteElementType const* const data,
        std::size_t const n)
{
    os << '(';
    if (n > 0) {
        os << data[0];
        for (std::size_t i = 1; i < n; ++i) {
            os << ", " << data[i];
        }
    }
    os << ')';
}

} // namespace ddc::detail
