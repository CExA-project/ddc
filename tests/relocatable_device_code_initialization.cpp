// Copyright (C) 2021 - 2024 The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include "relocatable_device_code_initialization.hpp"

namespace rdc {

void initialize_ddimx(ddc::Coordinate<DimX> const origin, ddc::Real const step)
{
    ddc::init_discrete_space<DDimX>(origin, step);
}

} // namespace rdc
