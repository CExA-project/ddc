#pragma once

#include <Kokkos_Macros.hpp>

#if defined(KOKKOS_ENABLE_CXX20) || defined(KOKKOS_ENABLE_CXX23) || \
    defined(KOKKOS_ENABLE_CXX26)
#define CEXA_HAS_CXX20
#endif
