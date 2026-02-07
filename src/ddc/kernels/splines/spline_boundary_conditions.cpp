// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <stdexcept>

#include "spline_boundary_conditions.hpp"

namespace ddc {

std::ostream& operator<<(std::ostream& out, ddc::BoundCond const bc)
{
    if (bc == ddc::BoundCond::PERIODIC) {
        return out << "PERIODIC";
    }

    if (bc == ddc::BoundCond::HERMITE) {
        return out << "HERMITE";
    }

    if (bc == ddc::BoundCond::GREVILLE) {
        return out << "GREVILLE";
    }

    throw std::runtime_error("ddc::BoundCond not handled");
}

} // namespace ddc
