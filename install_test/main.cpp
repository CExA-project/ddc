// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#if defined(DDC_BUILD_KERNELS_FFT)
#include <ddc/kernels/fft.hpp>
#else
#error
#endif

#if defined(DDC_BUILD_KERNELS_SPLINES)
#include <ddc/kernels/splines.hpp>
#else
#error
#endif

#if defined(DDC_BUILD_PDI_WRAPPER)
#include <ddc/pdi.hpp>
#else
#error
#endif

int main()
{
    return 0;
}
