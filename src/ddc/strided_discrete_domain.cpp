// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <span>

#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "strided_discrete_domain.hpp"

namespace ddc::detail {

void print_strided_discrete_domain(
        std::ostream& os,
        bool const empty,
        std::span<DiscreteElementType const> const front,
        std::span<DiscreteElementType const> const back,
        std::span<DiscreteVectorElement const> const strides)
{
    os << "StridedDiscreteDomain(";
    if (empty) {
        os << "empty";
    } else {
        os << "front=";
        print_discrete_element(os, front);
        os << ", back=";
        print_discrete_element(os, back);
        os << ", strides=";
        print_discrete_vector(os, strides);
    }
    os << ')';
}

} // namespace ddc::detail
