// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/config.hpp>

#if !DDC_BUILD_DEPRECATED_CODE()
#    include <type_traits>
#endif

namespace ddc {

struct DiscreteDimension
{
};

namespace concepts {

#if DDC_BUILD_DEPRECATED_CODE()
template <class T>
concept discrete_dimension = true;
#else
template <class T>
concept discrete_dimension = std::is_base_of_v<DiscreteDimension, T>;
#endif

} // namespace concepts

} // namespace ddc
