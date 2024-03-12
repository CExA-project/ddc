// Copyright (C) 2021 - 2024 The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ddc {

#ifdef DDC_BUILD_DOUBLE_PRECISION

using Real = double;

#else

using Real = float;

#endif

} // namespace ddc
