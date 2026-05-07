// NOLINTBEGIN(readability-identifier-naming)
// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#pragma once

#include <Kokkos_Macros.hpp>

#if defined(_MSVC_LANG)
#    define CEXA_STD_VERSION _MSVC_LANG
#else
#    define CEXA_STD_VERSION __cplusplus
#endif

// GCC 13 defines __cplusplus to 202100L when in c++23
#if (defined(KOKKOS_COMPILER_GNU) && CEXA_STD_VERSION >= 202100L) || CEXA_STD_VERSION >= 202302L
#    define CEXA_HAS_CXX23
#endif

#if defined(KOKKOS_COMPILER_NVCC)
#    define CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE _Pragma("nv_exec_check_disable")
#else
#    define CEXA_NVCC_HOST_DEVICE_CHECK_DISABLE
#endif

// FIXME: As of cuda 13, support for operator<=> in device code is still brittle
#if !defined(KOKKOS_COMPILER_NVCC)
#    define CEXA_TUPLE_IMPL_USE_SPACESHIP_OPERATOR
#endif
// NOLINTEND(readability-identifier-naming)
