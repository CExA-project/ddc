// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ostream>
#include <stdexcept>

#include "spline_builder_closures.hpp"

namespace ddc {

std::ostream& operator<<(std::ostream& os, ddc::SplineBuilderClosure const sbc)
{
    if (sbc == ddc::SplineBuilderClosure::PERIODIC) {
        return os << "PERIODIC";
    }

    if (sbc == ddc::SplineBuilderClosure::HERMITE) {
        return os << "HERMITE";
    }

    if (sbc == ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE) {
        return os << "HOMOGENEOUS_HERMITE";
    }

    if (sbc == ddc::SplineBuilderClosure::GREVILLE) {
        return os << "GREVILLE";
    }

    throw std::runtime_error("ddc::SplineBuilderClosure not handled");
}

} // namespace ddc
