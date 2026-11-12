// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <span>

#include "discrete_domain.hpp"
#include "discrete_element.hpp"

namespace ddc::detail {

void print_discrete_domain(
        std::ostream& os,
        bool const empty,
        std::span<DiscreteElementType const> const front,
        std::span<DiscreteElementType const> const back)
{
    os << "DiscreteDomain(";
    if (empty) {
        os << "empty";
    } else {
        os << "front=";
        print_discrete_element(os, front);
        os << ", back=";
        print_discrete_element(os, back);
    }
    os << ')';
}

} // namespace ddc::detail
