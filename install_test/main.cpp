// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#if defined(DDC_BUILD_DOUBLE_PRECISION)
static_assert(std::is_same_v<ddc::Real, double>);
#else
static_assert(std::is_same_v<ddc::Real, float>);
#endif

#if !__has_include(<ddc/kernels/fft.hpp>)
#    error
#else
#    include <ddc/kernels/fft.hpp>
#endif

#if !__has_include(<ddc/kernels/splines.hpp>)
#    error
#else
#    include <ddc/kernels/splines.hpp>
#endif

#if !__has_include(<ddc/pdi.hpp>)
#    error
#else
#    include <ddc/pdi.hpp>
#endif

int main()
{
    return 0;
}
