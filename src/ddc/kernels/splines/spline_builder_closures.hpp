// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iosfwd>
#include <stdexcept>

#include <ddc/ddc.hpp>

namespace ddc {

/** @brief An enum representing a spline closure relation. Please refer to
 * Emily Bourne's thesis (https://theses.fr/2022AIXM0412.pdf)
 */
enum class SplineBuilderClosure {
    PERIODIC, ///< Periodic closure relation u(1)=u(n)
    HERMITE, ///< Hermite closure relation
    HOMOGENEOUS_HERMITE, ///< Homogeneous Hermite closure relation (derivatives are 0)
    GREVILLE, ///< Use Greville points instead of conditions on derivative for B-Spline interpolation
};

/**
 * @brief Prints a closure relation in a std::ostream.
 *
 * @param os The stream in which the closure relation is printed.
 * @param sbc The closure relation.
 *
 * @return The stream in which the closure relation is printed.
 **/
std::ostream& operator<<(std::ostream& os, ddc::SplineBuilderClosure sbc);

/**
 * @brief Return the number of equations needed to describe a given closure relation.
 *
 * @param sbc The closure relation.
 * @param degree The degree of the spline.
 *
 * @return The number of equations.
 **/
constexpr int n_boundary_equations(ddc::SplineBuilderClosure const sbc, std::size_t const degree)
{
    if (sbc == ddc::SplineBuilderClosure::PERIODIC || sbc == ddc::SplineBuilderClosure::GREVILLE) {
        return 0;
    }

    if (sbc == ddc::SplineBuilderClosure::HERMITE
        || sbc == ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE) {
        return degree / 2;
    }

    throw std::runtime_error("ddc::SplineBuilderClosure not handled");
}

} // namespace ddc
