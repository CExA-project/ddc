// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

// Original copyright statement:
// SPDX-FileCopyrightText: Copyright (C) The CExA project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception

// NOLINTBEGIN(readability-identifier-naming)
#pragma once

#include <Kokkos_Macros.hpp>

namespace cexa {
namespace impl {
struct ignore_t
{
    template <class T>
    KOKKOS_INLINE_FUNCTION constexpr const ignore_t& operator=(T const&) const noexcept
    {
        return *this;
    }
};
} // namespace impl

// TODO: check if we need a host and a device version of this
inline constexpr impl::ignore_t ignore;
} // namespace cexa
// NOLINTEND(readability-identifier-naming)
