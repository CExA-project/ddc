// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

module;

#include <ddc/kernels/fft.hpp>

export module ddc.fft;

export namespace ddc {

using ::ddc::fft;
using ::ddc::FFT_Direction;
using ::ddc::FFT_Normalization;
using ::ddc::Fourier;
using ::ddc::fourier_mesh;
using ::ddc::ifft;
using ::ddc::init_fourier_space;
using ::ddc::kwArgs_fft;

namespace detail::fft {

using ::ddc::detail::fft::ddc_fft_normalization_to_kokkos_fft;
using ::ddc::detail::fft::is_complex_v;
using ::ddc::detail::fft::real_type_t;

} // namespace detail::fft

} // namespace ddc
