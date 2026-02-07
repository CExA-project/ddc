// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <ostream>

#include "discrete_vector.hpp"

namespace ddc::detail {

void discrete_vector_print(
        std::ostream& out,
        DiscreteVectorElement const* const data,
        std::size_t const n)
{
    out << '(';
    if (n > 0) {
        out << data[0];
        for (std::size_t i = 1; i < n; ++i) {
            out << ", " << data[i];
        }
    }
    out << ')';
}

} // namespace ddc::detail
