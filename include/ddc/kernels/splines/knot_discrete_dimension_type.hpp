// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include "bsplines_non_uniform.hpp"
#include "bsplines_uniform.hpp"

namespace ddc {

/// If the type `DDim` is a B-spline, defines `type` to the discrete dimension of the associated knots.
template <class DDim>
struct KnotDiscreteDimension
{
    static_assert(is_uniform_bsplines_v<DDim> || is_non_uniform_bsplines_v<DDim>);

    /// The type representing the discrete dimension of the knots.
    using type = std::conditional_t<
            is_uniform_bsplines_v<DDim>,
            UniformBsplinesKnots<DDim>,
            NonUniformBsplinesKnots<DDim>>;
};

/// Helper type to easily access `KnotDiscreteDimension<DDim>::type`
template <class DDim>
using knot_discrete_dimension_t = typename KnotDiscreteDimension<DDim>::type;

} // namespace ddc
