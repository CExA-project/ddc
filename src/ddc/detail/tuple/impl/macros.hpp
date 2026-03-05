// SPDX-FileCopyrightText: 2026 CExA-project
// SPDX-License-Identifier: MIT or Apache-2.0 with LLVM-exception
#pragma once

#if defined(_MSVC_LANG)
#define CEXA_STD_VERSION _MSVC_LANG
#else
#define CEXA_STD_VERSION __cplusplus
#endif

#if CEXA_STD_VERSION >= 202002L
#define CEXA_HAS_CXX20
#endif
#if CEXA_STD_VERSION >= 202302L
#define CEXA_HAS_CXX23
#endif
