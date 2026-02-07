// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include <KokkosFFT.hpp>

#include "fft.hpp"

namespace ddc::detail::fft {

KokkosFFT::Normalization ddc_fft_normalization_to_kokkos_fft(
        FFT_Normalization const ddc_fft_normalization)
{
    if (ddc_fft_normalization == ddc::FFT_Normalization::OFF
        || ddc_fft_normalization == ddc::FFT_Normalization::FULL) {
        return KokkosFFT::Normalization::none;
    }

    if (ddc_fft_normalization == ddc::FFT_Normalization::FORWARD) {
        return KokkosFFT::Normalization::forward;
    }

    if (ddc_fft_normalization == ddc::FFT_Normalization::BACKWARD) {
        return KokkosFFT::Normalization::backward;
    }

    if (ddc_fft_normalization == ddc::FFT_Normalization::ORTHO) {
        return KokkosFFT::Normalization::ortho;
    }

    throw std::runtime_error("ddc::FFT_Normalization not handled");
}

} // namespace ddc::detail::fft
