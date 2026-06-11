// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <span>

#include "discrete_element.hpp"

namespace ddc::detail {

void print_discrete_element(std::ostream& os, std::span<DiscreteElementType const> const view)
{
    os << '(';
    if (!view.empty()) {
        os << view.front();
        for (DiscreteElementType const value : view.subspan(1)) {
            os << ", " << value;
        }
    }
    os << ')';
}

} // namespace ddc::detail
