// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <KokkosFFT.hpp>
#include <Kokkos_Core.hpp>

namespace ddc {
// TODO : maybe transfert this somewhere else because Fourier space is not specific to FFT
template <typename Dim>
struct Fourier;

// named arguments for FFT (and their default values)
enum class FFT_Direction { FORWARD, BACKWARD };
enum class FFT_Normalization { OFF, FORWARD, BACKWARD, ORTHO, FULL };
} // namespace ddc

namespace ddc::detail::fft {
template <typename T>
struct real_type
{
    using type = T;
};

template <typename T>
struct real_type<Kokkos::complex<T>>
{
    using type = T;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

// is_complex : trait to determine if type is Kokkos::complex<something>
template <typename T>
struct is_complex : std::false_type
{
};

template <typename T>
struct is_complex<Kokkos::complex<T>> : std::true_type
{
};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// LastSelector: returns a if Dim==Last, else b
template <typename T, typename Dim, typename Last>
KOKKOS_FUNCTION constexpr T LastSelector(const T a, const T b)
{
    return std::is_same_v<Dim, Last> ? a : b;
}

template <typename T, typename Dim, typename First, typename Second, typename... Tail>
KOKKOS_FUNCTION constexpr T LastSelector(const T a, const T b)
{
    return LastSelector<T, Dim, Second, Tail...>(a, b);
}

// transform_type : trait to determine the type of transformation (R2C, C2R, C2C...) <- no information about base type (float or double)
enum class TransformType { R2R, R2C, C2R, C2C };

template <typename T1, typename T2>
struct transform_type
{
    static constexpr TransformType value = TransformType::R2R;
};

template <typename T1, typename T2>
struct transform_type<T1, Kokkos::complex<T2>>
{
    static constexpr TransformType value = TransformType::R2C;
};

template <typename T1, typename T2>
struct transform_type<Kokkos::complex<T1>, T2>
{
    static constexpr TransformType value = TransformType::C2R;
};

template <typename T1, typename T2>
struct transform_type<Kokkos::complex<T1>, Kokkos::complex<T2>>
{
    static constexpr TransformType value = TransformType::C2C;
};

template <typename T1, typename T2>
constexpr TransformType transform_type_v = transform_type<T1, T2>::value;

struct kwArgs_impl
{
    ddc::FFT_Direction
            direction; // Only effective for C2C transform and for normalization BACKWARD and FORWARD
    ddc::FFT_Normalization normalization;
};

// N,a,b from x_mesh
template <typename DDim, typename... DDimX>
int N(ddc::DiscreteDomain<DDimX...> x_mesh)
{
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    return ddc::get<DDim>(x_mesh.extents());
}

template <typename DDim, typename... DDimX>
double a(ddc::DiscreteDomain<DDimX...> x_mesh)
{
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    return ((2 * N<DDim>(x_mesh) - 1) * coordinate(ddc::select<DDim>(x_mesh).front())
            - coordinate(ddc::select<DDim>(x_mesh).back()))
           / 2 / (N<DDim>(x_mesh) - 1);
}

template <typename DDim, typename... DDimX>
double b(ddc::DiscreteDomain<DDimX...> x_mesh)
{
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    return ((2 * N<DDim>(x_mesh) - 1) * coordinate(ddc::select<DDim>(x_mesh).back())
            - coordinate(ddc::select<DDim>(x_mesh).front()))
           / 2 / (N<DDim>(x_mesh) - 1);
}

template <typename... DDimX>
static constexpr KokkosFFT::axis_type<sizeof...(DDimX)> axes()
{
    return KokkosFFT::axis_type<sizeof...(DDimX)> {
            static_cast<int>(ddc::type_seq_rank_v<DDimX, ddc::detail::TypeSeq<DDimX...>>)...};
}

KokkosFFT::Normalization ddc_fft_normalization_to_kokkos_fft(
        FFT_Normalization const ddc_fft_normalization)
{
    KokkosFFT::Normalization kokkos_fft_normalization;
    switch (ddc_fft_normalization) {
    case ddc::FFT_Normalization::OFF:
        kokkos_fft_normalization = KokkosFFT::Normalization::none;
        break;
    case ddc::FFT_Normalization::FORWARD:
        kokkos_fft_normalization = KokkosFFT::Normalization::forward;
        break;
    case ddc::FFT_Normalization::BACKWARD:
        kokkos_fft_normalization = KokkosFFT::Normalization::backward;
        break;
    case ddc::FFT_Normalization::ORTHO:
        kokkos_fft_normalization = KokkosFFT::Normalization::ortho;
        break;
    // Last case is FULL which is mesh-dependant and thus handled by DDC.
    default:
        kokkos_fft_normalization = KokkosFFT::Normalization::none;
    }

    return kokkos_fft_normalization;
};

// impl
template <
        typename Tin,
        typename Tout,
        typename ExecSpace,
        typename MemorySpace,
        typename layout_in,
        typename layout_out,
        typename DomainOut,
        typename... DDimIn>
void impl(
        ExecSpace execSpace,
        ddc::ChunkSpan<Tin, ddc::DiscreteDomain<DDimIn...>, layout_in, MemorySpace> const in,
        ddc::ChunkSpan<Tout, DomainOut, layout_out, MemorySpace> out,
        const kwArgs_impl& kwargs)
{
    static_assert(
            std::is_same_v<real_type_t<Tin>, float> || std::is_same_v<real_type_t<Tin>, double>,
            "Base type of Tin (and Tout) must be float or double.");
    static_assert(
            std::is_same_v<real_type_t<Tin>, real_type_t<Tout>>,
            "Types Tin and Tout must be based on same type (float or double)");
    static_assert(
            Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible,
            "MemorySpace has to be accessible for ExecutionSpace.");

    Kokkos::View<
            ddc::detail::mdspan_to_kokkos_element_t<Tin, sizeof...(DDimIn)>,
            ddc::detail::mdspan_to_kokkos_layout_t<layout_in>,
            ExecSpace>
            in_view(in.allocation_kokkos_view());
    Kokkos::View<
            ddc::detail::mdspan_to_kokkos_element_t<Tout, sizeof...(DDimIn)>,
            ddc::detail::mdspan_to_kokkos_layout_t<layout_out>,
            ExecSpace>
            out_view(out.allocation_kokkos_view());
    KokkosFFT::Normalization kokkos_fft_normalization(
            ddc_fft_normalization_to_kokkos_fft(kwargs.normalization));

    // C2C
    if constexpr (std::is_same_v<Tin, Tout>) {
        if (kwargs.direction == ddc::FFT_Direction::FORWARD) {
            KokkosFFT::
                    fftn(execSpace, in_view, out_view, axes<DDimIn...>(), kokkos_fft_normalization);
        } else {
            KokkosFFT::
                    ifftn(execSpace,
                          in_view,
                          out_view,
                          axes<DDimIn...>(),
                          kokkos_fft_normalization);
        }
        // R2C & C2R
    } else {
        if constexpr (is_complex_v<Tout>) {
            assert(kwargs.direction == ddc::FFT_Direction::FORWARD);
            KokkosFFT::
                    rfftn(execSpace,
                          in_view,
                          out_view,
                          axes<DDimIn...>(),
                          kokkos_fft_normalization);
        } else {
            assert(kwargs.direction == ddc::FFT_Direction::BACKWARD);
            KokkosFFT::
                    irfftn(execSpace,
                           in_view,
                           out_view,
                           axes<DDimIn...>(),
                           kokkos_fft_normalization);
        }
    }
    execSpace.fence();

    // The FULL normalization is mesh-dependant and thus handled by DDC
    if (kwargs.normalization == ddc::FFT_Normalization::FULL) {
        const real_type_t<Tout> norm_coef
                = ((rlength(ddc::select<DDimIn>(in.domain()))
                    / (ddc::get<DDimIn>(in.domain().extents()) - 1))
                   * ...)
                  / Kokkos::
                          pow(2 * Kokkos::numbers::pi,
                              static_cast<real_type_t<Tout>>(sizeof...(DDimIn)) / 2);
        ddc::parallel_for_each(
                "ddc_fft_normalization",
                execSpace,
                out.domain(),
                KOKKOS_LAMBDA(typename DomainOut::discrete_element_type const i) {
                    out(i) = out(i) * norm_coef;
                });
    }
}
} // namespace ddc::detail::fft

namespace ddc {

template <typename DDimFx, typename DDimX>
typename DDimFx::template Impl<DDimFx, Kokkos::HostSpace> init_fourier_space(
        ddc::DiscreteDomain<DDimX> x_mesh)
{
    static_assert(
            is_uniform_point_sampling_v<DDimX>,
            "DDimX dimensions should derive from UniformPointSampling");
    static_assert(
            is_periodic_sampling_v<DDimFx>,
            "DDimFx dimensions should derive from PeriodicPointSampling");
    auto [impl, ddom] = DDimFx::template init<DDimFx>(
            ddc::Coordinate<typename DDimFx::continuous_dimension_type>(0),
            ddc::Coordinate<typename DDimFx::continuous_dimension_type>(
                    2 * (ddc::detail::fft::N<DDimX>(x_mesh) - 1)
                    / (ddc::detail::fft::b<DDimX>(x_mesh) - ddc::detail::fft::a<DDimX>(x_mesh))
                    * Kokkos::numbers::pi),
            ddc::DiscreteVector<DDimFx>(ddc::detail::fft::N<DDimX>(x_mesh)),
            ddc::DiscreteVector<DDimFx>(ddc::detail::fft::N<DDimX>(x_mesh)));
    return std::move(impl);
}

// FourierMesh, first element corresponds to mode 0
template <typename... DDimFx, typename... DDimX>
ddc::DiscreteDomain<DDimFx...> FourierMesh(ddc::DiscreteDomain<DDimX...> x_mesh, bool C2C)
{
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    static_assert(
            (is_periodic_sampling_v<DDimFx> && ...),
            "DDimFx dimensions should derive from PeriodicPointSampling");
    return ddc::DiscreteDomain<DDimFx...>(ddc::DiscreteDomain<DDimFx>(
            ddc::DiscreteElement<DDimFx>(0),
            ddc::DiscreteVector<DDimFx>(
                    (C2C ? ddc::detail::fft::N<DDimX>(x_mesh)
                         : ddc::detail::fft::LastSelector<double, DDimX, DDimX...>(
                                 ddc::detail::fft::N<DDimX>(x_mesh) / 2 + 1,
                                 ddc::detail::fft::N<DDimX>(x_mesh)))))...);
}

struct kwArgs_fft
{
    ddc::FFT_Normalization normalization;
};

// FFT
template <
        typename Tin,
        typename Tout,
        typename... DDimFx,
        typename... DDimX,
        typename ExecSpace,
        typename MemorySpace,
        typename layout_in,
        typename layout_out>
void fft(
        ExecSpace const& execSpace,
        ddc::ChunkSpan<Tout, ddc::DiscreteDomain<DDimFx...>, layout_out, MemorySpace> out,
        ddc::ChunkSpan<Tin, ddc::DiscreteDomain<DDimX...>, layout_in, MemorySpace> in,
        ddc::kwArgs_fft kwargs = {ddc::FFT_Normalization::OFF})
{
    static_assert(
            std::is_same_v<
                    layout_in,
                    std::experimental::
                            layout_right> && std::is_same_v<layout_out, std::experimental::layout_right>,
            "Layouts must be right-handed");
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    static_assert(
            (is_periodic_sampling_v<DDimFx> && ...),
            "DDimFx dimensions should derive from PeriodicPointSampling");

    ddc::detail::fft::impl(execSpace, in, out, {ddc::FFT_Direction::FORWARD, kwargs.normalization});
}

// iFFT (deduced from the fact that "in" is identified as a function on the Fourier space)
template <
        typename Tin,
        typename Tout,
        typename... DDimX,
        typename... DDimFx,
        typename ExecSpace,
        typename MemorySpace,
        typename layout_in,
        typename layout_out>
void ifft(
        ExecSpace const& execSpace,
        ddc::ChunkSpan<Tout, ddc::DiscreteDomain<DDimX...>, layout_out, MemorySpace> out,
        ddc::ChunkSpan<Tin, ddc::DiscreteDomain<DDimFx...>, layout_in, MemorySpace> in,
        ddc::kwArgs_fft kwargs = {ddc::FFT_Normalization::OFF})
{
    static_assert(
            std::is_same_v<
                    layout_in,
                    std::experimental::
                            layout_right> && std::is_same_v<layout_out, std::experimental::layout_right>,
            "Layouts must be right-handed");
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    static_assert(
            (is_periodic_sampling_v<DDimFx> && ...),
            "DDimFx dimensions should derive from PeriodicPointSampling");

    ddc::detail::fft::
            impl(execSpace, in, out, {ddc::FFT_Direction::BACKWARD, kwargs.normalization});
}
} // namespace ddc
