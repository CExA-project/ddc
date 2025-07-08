# Copyright (C) The DDC development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# This file provides different macros that will configure the dependencies with required options.
# They should be used only if DDC is responsible of handling the vendor dependency.

macro(ddc_configure_benchmark)
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    option(
        BENCHMARK_ENABLE_INSTALL
        "Enable installation of benchmark. (Projects embedding benchmark may want to turn this OFF.)"
        OFF
    )
    add_subdirectory(vendor/benchmark)
endmacro()

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
        option(
            Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
            "Whether to enable relocatable device code (RDC) for CUDA"
            ON
        )
    endif()
    if("${Kokkos_ENABLE_HIP}")
        option(
            Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE
            "Whether to enable relocatable device code (RDC) for HIP"
            ON
        )
    endif()
    add_subdirectory(vendor/kokkos)
endmacro()

macro(ddc_configure_kokkos_fft)
    option(KokkosFFT_ENABLE_FFTW "Enable fftw as a KokkosFFT backend on CPUs" ON)
    add_subdirectory(vendor/kokkos-fft)
endmacro()

macro(ddc_configure_kokkos_kernels)
    set(KokkosKernels_ENABLE_ALL_COMPONENTS OFF)
    set(KokkosKernels_ENABLE_COMPONENT_BLAS ON)
    set(KokkosKernels_ENABLE_COMPONENT_BATCHED ON)
    set(KokkosKernels_ENABLE_COMPONENT_LAPACK OFF)
    set(KokkosKernels_ENABLE_TPL_BLAS OFF)
    set(KokkosKernels_ENABLE_TPL_LAPACK OFF)
    add_subdirectory(vendor/kokkos-kernels)
endmacro()
