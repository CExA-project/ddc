// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace ddc {

struct DiscreteDimension
{
};

namespace concepts {

template <class T>
concept discrete_dimension = std::is_base_of_v<DiscreteDimension, T>;

}

} // namespace ddc
