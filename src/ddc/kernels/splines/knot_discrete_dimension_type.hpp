// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "bsplines.hpp"
#include "bsplines_non_uniform.hpp"
#include "bsplines_uniform.hpp"

namespace ddc {

/// If the type `DDim` is a B-spline, defines `type` to the discrete dimension of the associated knots.
template <concepts::bsplines DDim>
struct KnotDiscreteDimension
{
    /// The type representing the discrete dimension of the knots.
    using type = std::conditional_t<
            is_uniform_bsplines_v<DDim>,
            UniformBsplinesKnots<DDim>,
            NonUniformBsplinesKnots<DDim>>;
};

/// Helper type to easily access `KnotDiscreteDimension<DDim>::type`
template <concepts::bsplines DDim>
using knot_discrete_dimension_t = KnotDiscreteDimension<DDim>::type;

} // namespace ddc
