// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include "bsplines_non_uniform.hpp"
#include "bsplines_uniform.hpp"

namespace ddc::concepts {

/// @brief Concept that is satisfied when the given type represents either uniform
///        or non-uniform B-splines discrete dimension.
template <class DDim>
concept bsplines = uniform_bsplines<DDim> || non_uniform_bsplines<DDim>;

} // namespace ddc::concepts
