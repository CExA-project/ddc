// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <stdexcept>

#include "spline_boundary_conditions.hpp"

namespace ddc {

std::ostream& operator<<(std::ostream& os, ddc::BoundCond const bc)
{
    if (bc == ddc::BoundCond::PERIODIC) {
        return os << "PERIODIC";
    }

    if (bc == ddc::BoundCond::HERMITE) {
        return os << "HERMITE";
    }

    if (bc == ddc::BoundCond::HOMOGENEOUS_HERMITE) {
        return os << "HOMOGENEOUS_HERMITE";
    }

    if (bc == ddc::BoundCond::GREVILLE) {
        return os << "GREVILLE";
    }

    throw std::runtime_error("ddc::BoundCond not handled");
}

} // namespace ddc
