// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#if defined(DDC_BUILD_DOUBLE_PRECISION)
static_assert(std::is_same_v<ddc::Real, double>);
#else
static_assert(std::is_same_v<ddc::Real, float>);
#endif

#if __has_include(<ddc/kernels/fft.hpp>)
#    include <ddc/kernels/fft.hpp>
#else
#    error "The header <ddc/kernels/fft.hpp> cannot be found"
#endif

#if __has_include(<ddc/kernels/splines.hpp>)
#    include <ddc/kernels/splines.hpp>
#else
#    error "The header <ddc/kernels/splines.hpp> cannot be found"
#endif

#if __has_include(<ddc/pdi.hpp>)
#    include <ddc/pdi.hpp>
#else
#    error "The header <ddc/pdi.hpp> cannot be found"
#endif

int main()
{
    return 0;
}
