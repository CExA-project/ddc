// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ddc::detail {

// This helper was introduced to workaround a parsing issue with msvc.
template <bool... Bs>
inline constexpr bool all_of_v = (Bs && ...);

} // namespace ddc::detail
