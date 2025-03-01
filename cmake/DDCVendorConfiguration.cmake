# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# This file provides different macros that will configure the dependencies with required options.
# They should be used only if DDC is responsible of handling the vendor dependency.

macro(ddc_configure_googletest)
    add_subdirectory(vendor/googletest)
endmacro()

macro(ddc_configure_kokkos)
    if("${Kokkos_ENABLE_CUDA}")
        option(
            Kokkos_ENABLE_CUDA_CONSTEXPR
            "Whether to activate experimental relaxed constexpr functions"
            ON
        )
    endif()
    add_subdirectory(vendor/kokkos)
endmacro()
