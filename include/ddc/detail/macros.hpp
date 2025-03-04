// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#if defined(__HIPCC__)
#define DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(NAME)                                           \
    DDC_NAMESPACE_##NAME {}                                                                        \
    using namespace DDC_NAMESPACE_##NAME;                                                          \
    namespace DDC_NAMESPACE_##NAME
#else
#define DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(NAME)
#endif
