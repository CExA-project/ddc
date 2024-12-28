// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include "relocatable_device_code_initialization.hpp"

namespace rdc {

void initialize_ddimx(ddc::Coordinate<DimX> const origin, ddc::Real const step)
{
    ddc::create_uniform_point_sampling<DDimX>(origin, ddc::Coordinate<DimX>(step));
}

} // namespace rdc
