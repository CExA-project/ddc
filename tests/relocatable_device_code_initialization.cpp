// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include "relocatable_device_code_initialization.hpp"

import ddc.core;

namespace rdc {

void initialize_ddimx(ddc::Coordinate<DimX> const origin, ddc::Real const step)
{
    ddc::init_discrete_space<DDimX>(origin, step);
}

} // namespace rdc
