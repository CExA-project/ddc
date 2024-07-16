// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

#if fftw_serial_AVAIL || fftw_omp_AVAIL
#include <fftw3.h>
#endif

#if cufft_AVAIL
#include <functional>
#include <memory>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <cufft.h>
#endif

#if hipfft_AVAIL
#include <functional>
#include <memory>
#include <stdexcept>

#include <hip/hip_runtime_api.h>
#include <hipfft/hipfft.h>
#endif

#if fftw_serial_AVAIL || fftw_omp_AVAIL
static_assert(sizeof(fftwf_complex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(fftwf_complex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(fftw_complex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(fftw_complex) <= alignof(Kokkos::complex<double>));

static_assert(sizeof(fftwl_complex) == sizeof(Kokkos::complex<long double>));
static_assert(alignof(fftwl_complex) <= alignof(Kokkos::complex<long double>));
#endif

#if cufft_AVAIL
static_assert(sizeof(cufftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(cufftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(cufftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(cufftDoubleComplex) <= alignof(Kokkos::complex<double>));
#endif

#if hipfft_AVAIL
static_assert(sizeof(hipfftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(hipfftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(hipfftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(hipfftDoubleComplex) <= alignof(Kokkos::complex<double>));
#endif

namespace ddc {
/**
 * @brief A templated tag representing a continuous dimension in the Fourier space associated to the original continuous dimension.
 *
 * @tparam The tag representing the original dimension.
 */
template <typename Dim>
struct Fourier;

/**
 * @brief A named argument to choose the direction of the FFT.
 *
 * @see kwArgs_core, kwArgs_fft
 */
enum class FFT_Direction {
    FORWARD, ///< Forward, corresponds to direct FFT up to normalization
    BACKWARD ///< Backward, corresponds to inverse FFT up to normalization
};

/**
 * @brief A named argument to choose the type of normalization of the FFT.
 *
 * @see kwArgs_core, kwArgs_fft
 */
enum class FFT_Normalization {
    OFF, ///< No normalization
    FORWARD, ///< Multiply by 1/N for forward FFT, no normalization for backward FFT
    BACKWARD, ///< No normalization for forward FFT, multiply by 1/N for backward FFT
    ORTHO, ///< Multiply by 1/sqrt(N)
    FULL /**< 
          * Multiply by (b-a)/N/sqrt(2*pi) for forward FFT and (kb-ka)/N/sqrt(2*pi) for backward
          * FFT. It is aligned with the usual definition of the (continuous) Fourier transform
          * 1/sqrt(2*pi)*int f(x)*e^-ikx*dx, and thus preserves the gaussian function exp(-x^2/2) numerically.
          */
};
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

/**
 * @brief A trait to identify the type of transformation (R2C, C2R, C2C...).
 *
 * It does not contain the information about the base type (float or double).
 */
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

/**
 * @brief A trait to get the TransformType for the input and output types.
 *
 * Internally check if T1 and T2 are Kokkos::complex<something> or not.
 *
 * @tparam T1 The input type.
 * @tparam T2 The output type.
 */
template <typename T1, typename T2>
constexpr TransformType transform_type_v = transform_type<T1, T2>::value;

#if fftw_serial_AVAIL || fftw_omp_AVAIL
// _fftw_type : compatible with both single and double precision
template <typename T>
struct _fftw_type
{
    using type = T;
};

template <typename T>
struct _fftw_type<Kokkos::complex<T>>
{
    using type = std::
            conditional_t<std::is_same_v<real_type_t<T>, float>, fftwf_complex, fftw_complex>;
};

// _fftw_plan : compatible with both single and double precision
template <typename T>
using _fftw_plan = std::conditional_t<std::is_same_v<real_type_t<T>, float>, fftwf_plan, fftw_plan>;

// _fftw_plan_many_dft : templated function working for all types of transformation
template <typename Tin, typename Tout, typename... Args, typename PenultArg, typename LastArg>
_fftw_plan<Tin> _fftw_plan_many_dft(
        [[maybe_unused]] PenultArg penultArg,
        LastArg lastArg,
        Args... args)
{ // Ugly, penultArg and lastArg are passed before the rest because of a limitation of C++ (parameter packs must be last arguments)
    const TransformType transformType = transform_type_v<Tin, Tout>;
    if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, float>)
        return fftwf_plan_many_dft_r2c(args..., lastArg);
    else if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, double>)
        return fftw_plan_many_dft_r2c(args..., lastArg);
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, float>)
        return fftwf_plan_many_dft_c2r(args..., lastArg);
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, double>)
        return fftw_plan_many_dft_c2r(args..., lastArg);
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<float>>)
        return fftwf_plan_many_dft(args..., penultArg, lastArg);
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<double>>)
        return fftw_plan_many_dft(args..., penultArg, lastArg);
    // else constexpr
    //   static_assert(false, "Transform type not supported");
}

#endif
#if cufft_AVAIL
// _cufft_type : compatible with both single and double precision
template <typename T>
struct _cufft_type
{
    using type = std::conditional_t<std::is_same_v<T, float>, cufftReal, cufftDoubleReal>;
};

template <typename T>
struct _cufft_type<Kokkos::complex<T>>
{
    using type = std::
            conditional_t<std::is_same_v<real_type_t<T>, float>, cufftComplex, cufftDoubleComplex>;
};

// cufft_transform_type : argument passed in the cufftMakePlan function
template <typename Tin, typename Tout>
constexpr auto cufft_transform_type()
{
    const TransformType transformType = transform_type_v<Tin, Tout>;
    if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, float>)
        return CUFFT_R2C;
    else if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, double>)
        return CUFFT_D2Z;
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, float>)
        return CUFFT_C2R;
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, double>)
        return CUFFT_Z2D;
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<float>>)
        return CUFFT_C2C;
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<double>>)
        return CUFFT_Z2Z;
    // else constexpr
    //	static_assert(false, "Transform type not supported");
}

// cufftExec : argument passed in the cufftMakePlan function
// _fftw_plan_many_dft : templated function working for all types of transformation
template <typename Tin, typename Tout, typename... Args, typename LastArg>
cufftResult _cufftExec([[maybe_unused]] LastArg lastArg, Args... args)
{ // Ugly for same reason as fftw
    const TransformType transformType = transform_type_v<Tin, Tout>;
    if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, float>)
        return cufftExecR2C(args...);
    else if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, double>)
        return cufftExecD2Z(args...);
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, float>)
        return cufftExecC2R(args...);
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, double>)
        return cufftExecZ2D(args...);
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<float>>)
        return cufftExecC2C(args..., lastArg);
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<double>>)
        return cufftExecZ2Z(args..., lastArg);
    // else constexpr
    //   static_assert(false, "Transform type not supported");
}
#endif
#if hipfft_AVAIL
// _hipfft_type : compatible with both single and double precision
template <typename T>
struct _hipfft_type
{
    using type = std::conditional_t<std::is_same_v<T, float>, hipfftReal, hipfftDoubleReal>;
};

template <typename T>
struct _hipfft_type<Kokkos::complex<T>>
{
    using type = std::conditional_t<
            std::is_same_v<real_type_t<T>, float>,
            hipfftComplex,
            hipfftDoubleComplex>;
};

// hipfft_transform_type : argument passed in the hipfftMakePlan function
template <typename Tin, typename Tout>
constexpr auto hipfft_transform_type()
{
    const TransformType transformType = transform_type_v<Tin, Tout>;
    if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, float>)
        return HIPFFT_R2C;
    else if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, double>)
        return HIPFFT_D2Z;
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, float>)
        return HIPFFT_C2R;
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, double>)
        return HIPFFT_Z2D;
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<float>>)
        return HIPFFT_C2C;
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<double>>)
        return HIPFFT_Z2Z;
    // else constexpr
    //	static_assert(false, "Transform type not supported");
}

// hipfftExec : argument passed in the hipfftMakePlan function
// _fftw_plan_many_dft : templated function working for all types of transformation
template <typename Tin, typename Tout, typename... Args, typename LastArg>
hipfftResult _hipfftExec([[maybe_unused]] LastArg lastArg, Args... args)
{
    const TransformType transformType = transform_type_v<Tin, Tout>;
    if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, float>)
        return hipfftExecR2C(args...);
    else if constexpr (transformType == TransformType::R2C && std::is_same_v<Tin, double>)
        return hipfftExecD2Z(args...);
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, float>)
        return hipfftExecC2R(args...);
    else if constexpr (transformType == TransformType::C2R && std::is_same_v<Tout, double>)
        return hipfftExecZ2D(args...);
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<float>>)
        return hipfftExecC2C(args..., lastArg);
    else if constexpr (
            transformType == TransformType::C2C && std::is_same_v<Tin, Kokkos::complex<double>>)
        return hipfftExecZ2Z(args..., lastArg);
    // else constexpr
    //   static_assert(false, "Transform type not supported");
}
#endif

/* 
 * @brief A structure embedding the configuration of the core FFT function: direction and type of normalization.
 *
 * @see FFT_core 
 */
struct kwArgs_core
{
    ddc::FFT_Direction
            direction; // Only effective for C2C transform and for normalization BACKWARD and FORWARD
    ddc::FFT_Normalization normalization;
};

/**
 * @brief Get the mesh size along a given dimension.
 *
 * @tparam DDim The dimension along which the mesh size is returned.
 * @param x_mesh The mesh.
 *
 * @return The mesh size along the required dimension.
 */
template <typename DDim, typename... DDimX>
int N(ddc::DiscreteDomain<DDimX...> x_mesh)
{
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");
    return static_cast<int>(x_mesh.template extent<DDim>());
}

/**
 * @brief Get the lower boundary coordinate along a given dimension.
 *
 * The lower boundary of the domain (which appears in Nyquist-Shannon theorem) is not
 * xmin=ddc::coordinate(x_mesh.front()). Indeed, this coordinate identifies the lower cell, but
 * the lower boundary is the left side of this lower cell, which is a = xmin - cell_size/2, with
 * cell_size = (b-a)/N. It leads to a = xmin-(b-a)/2N. The same derivation for the
 * upper boundary coordinate gives b = xmax+(b-a)/2N. Inverting this linear system leads to:
 *
 * a = ((2N-1)*xmin-xmax)/2/(N-1)
 * b = ((2N-1)*xmax-xmin)/2/(N-1)
 *
 * The current function implements the first equation.
 *
 * @tparam DDim The dimension along which the lower cell coordinate of the Fourier mesh is returned.
 * @param x_mesh The mesh.
 *
 * @return The lower boundary along the required dimension.
 */
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

/**
 * @brief Get the upper boundary coordinate along a given dimension.
 *
 * The upper boundary of the domain (which appears in Nyquist-Shannon theorem) is not
 * xmax=ddc::coordinate(x_mesh.back()). Indeed, this coordinate identifies the upper cell, but
 * the upper boundary is the right side of this upper cell, which is b = xmax + cell_size/2, with
 * cell_size = (b-a)/N. It leads to b = xmax+(b-a)/2N. The same derivation for the
 * lower boundary coordinate gives a = xmin-(b-a)/2N. Inverting this linear system leads to:
 *
 * a = ((2N-1)*xmin-xmax)/2/(N-1)
 * b = ((2N-1)*xmax-xmin)/2/(N-1)
 *
 * The current function implements the second equation.
 *
 * @tparam DDim The dimension along which the upper cell coordinate of the Fourier mesh is returned.
 * @param x_mesh The mesh.
 *
 * @return The upper boundary along the required dimension.
 */
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

/// @brief Core internal function to perform the FFT.
template <typename Tin, typename Tout, typename ExecSpace, typename MemorySpace, typename... DDimX>
void core(
        ExecSpace const& execSpace,
        Tout* out_data,
        Tin* in_data,
        ddc::DiscreteDomain<DDimX...> mesh,
        const kwArgs_core& kwargs)
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
    static_assert(
            (is_uniform_point_sampling_v<DDimX> && ...),
            "DDimX dimensions should derive from UniformPointSampling");

    std::array<int, sizeof...(DDimX)> n = {static_cast<int>(ddc::get<DDimX>(mesh.extents()))...};
    int idist = 1;
    int odist = 1;
    for (std::size_t i = 0; i < sizeof...(DDimX); i++) {
        idist = transform_type_v<Tin, Tout> == TransformType::C2R && i == sizeof...(DDimX) - 1
                        ? idist * (n[i] / 2 + 1)
                        : idist * n[i];
        odist = transform_type_v<Tin, Tout> == TransformType::R2C && i == sizeof...(DDimX) - 1
                        ? odist * (n[i] / 2 + 1)
                        : odist * n[i];
    }

    if constexpr (false) {
    } // Trick to get only else if
#if fftw_serial_AVAIL
    else if constexpr (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        _fftw_plan<Tin> plan = _fftw_plan_many_dft<Tin, Tout>(
                kwargs.direction == ddc::FFT_Direction::FORWARD ? FFTW_FORWARD : FFTW_BACKWARD,
                FFTW_ESTIMATE,
                static_cast<int>(sizeof...(DDimX)),
                n.data(),
                1,
                reinterpret_cast<typename _fftw_type<Tin>::type*>(in_data),
                static_cast<int*>(nullptr),
                1,
                idist,
                reinterpret_cast<typename _fftw_type<Tout>::type*>(out_data),
                static_cast<int*>(nullptr),
                1,
                odist);
        if constexpr (std::is_same_v<real_type_t<Tin>, float>) {
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        } else {
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }
#endif
#if fftw_omp_AVAIL
    else if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        if constexpr (std::is_same_v<real_type_t<Tin>, float>) {
            fftwf_init_threads();
            fftwf_plan_with_nthreads(execSpace.concurrency());
        } else {
            fftw_init_threads();
            fftw_plan_with_nthreads(execSpace.concurrency());
        }
        _fftw_plan<Tin> plan = _fftw_plan_many_dft<Tin, Tout>(
                kwargs.direction == ddc::FFT_Direction::FORWARD ? FFTW_FORWARD : FFTW_BACKWARD,
                FFTW_ESTIMATE,
                static_cast<int>(sizeof...(DDimX)),
                n.data(),
                1,
                reinterpret_cast<typename _fftw_type<Tin>::type*>(in_data),
                static_cast<int*>(nullptr),
                1,
                idist,
                reinterpret_cast<typename _fftw_type<Tout>::type*>(out_data),
                static_cast<int*>(nullptr),
                1,
                odist);
        if constexpr (std::is_same_v<real_type_t<Tin>, float>) {
            fftwf_execute(plan);
            fftwf_destroy_plan(plan);
        } else {
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }
#endif
#if cufft_AVAIL
    else if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        cudaStream_t stream = execSpace.cuda_stream();

        cufftHandle unmanaged_plan = -1;
        cufftResult cufft_rt = cufftCreate(&unmanaged_plan);

        if (cufft_rt != CUFFT_SUCCESS)
            throw std::runtime_error("cufftCreate failed");

        std::unique_ptr<cufftHandle, std::function<void(cufftHandle*)>> const
                managed_plan(&unmanaged_plan, [](cufftHandle* handle) { cufftDestroy(*handle); });

        cufftSetStream(unmanaged_plan, stream);
        cufft_rt = cufftPlanMany(
                &unmanaged_plan, // plan handle
                sizeof...(DDimX),
                n.data(), // Nx, Ny...
                nullptr,
                1,
                idist,
                nullptr,
                1,
                odist,
                cufft_transform_type<Tin, Tout>(),
                1);

        if (cufft_rt != CUFFT_SUCCESS)
            throw std::runtime_error("cufftPlan failed");

        cufft_rt = _cufftExec<Tin, Tout>(
                kwargs.direction == ddc::FFT_Direction::FORWARD ? CUFFT_FORWARD : CUFFT_INVERSE,
                unmanaged_plan,
                reinterpret_cast<typename _cufft_type<Tin>::type*>(in_data),
                reinterpret_cast<typename _cufft_type<Tout>::type*>(out_data));
        if (cufft_rt != CUFFT_SUCCESS)
            throw std::runtime_error("cufftExec failed");
    }
#endif
#if hipfft_AVAIL
    else if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        hipStream_t stream = execSpace.hip_stream();

        hipfftHandle unmanaged_plan;
        hipfftResult hipfft_rt = hipfftCreate(&unmanaged_plan);

        if (hipfft_rt != HIPFFT_SUCCESS)
            throw std::runtime_error("hipfftCreate failed");

        std::unique_ptr<hipfftHandle, std::function<void(hipfftHandle*)>> const
                managed_plan(&unmanaged_plan, [](hipfftHandle* handle) { hipfftDestroy(*handle); });

        hipfftSetStream(unmanaged_plan, stream);
        hipfft_rt = hipfftPlanMany(
                &unmanaged_plan, // plan handle
                sizeof...(DDimX),
                n.data(), // Nx, Ny...
                nullptr,
                1,
                idist,
                nullptr,
                1,
                odist,
                hipfft_transform_type<Tin, Tout>(),
                1);

        if (hipfft_rt != HIPFFT_SUCCESS)
            throw std::runtime_error("hipfftPlan failed");

        hipfft_rt = _hipfftExec<Tin, Tout>(
                kwargs.direction == ddc::FFT_Direction::FORWARD ? HIPFFT_FORWARD : HIPFFT_BACKWARD,
                unmanaged_plan,
                reinterpret_cast<typename _hipfft_type<Tin>::type*>(in_data),
                reinterpret_cast<typename _hipfft_type<Tout>::type*>(out_data));
        if (hipfft_rt != HIPFFT_SUCCESS)
            throw std::runtime_error("hipfftExec failed");
    }
#endif

    if (kwargs.normalization != ddc::FFT_Normalization::OFF) {
        real_type_t<Tout> norm_coef = 1;
        switch (kwargs.normalization) {
        case ddc::FFT_Normalization::OFF:
            break;
        case ddc::FFT_Normalization::FORWARD:
            norm_coef = kwargs.direction == ddc::FFT_Direction::FORWARD
                                ? 1. / (ddc::get<DDimX>(mesh.extents()) * ...)
                                : 1.;
            break;
        case ddc::FFT_Normalization::BACKWARD:
            norm_coef = kwargs.direction == ddc::FFT_Direction::BACKWARD
                                ? 1. / (ddc::get<DDimX>(mesh.extents()) * ...)
                                : 1.;
            break;
        case ddc::FFT_Normalization::ORTHO:
            norm_coef = 1. / Kokkos::sqrt((ddc::get<DDimX>(mesh.extents()) * ...));
            break;
        case ddc::FFT_Normalization::FULL:
            norm_coef = kwargs.direction == ddc::FFT_Direction::FORWARD
                                ? (((coordinate(ddc::select<DDimX>(mesh).back())
                                     - coordinate(ddc::select<DDimX>(mesh).front()))
                                    / (ddc::get<DDimX>(mesh.extents()) - 1)
                                    / Kokkos::sqrt(2 * Kokkos::numbers::pi))
                                   * ...)
                                : ((Kokkos::sqrt(2 * Kokkos::numbers::pi)
                                    / (coordinate(ddc::select<DDimX>(mesh).back())
                                       - coordinate(ddc::select<DDimX>(mesh).front()))
                                    * (ddc::get<DDimX>(mesh.extents()) - 1)
                                    / ddc::get<DDimX>(mesh.extents()))
                                   * ...);
            break;
        }

        Kokkos::parallel_for(
                "ddc_fft_normalization",
                Kokkos::RangePolicy<ExecSpace>(
                        execSpace,
                        0,
                        is_complex_v<Tout> && transform_type_v<Tin, Tout> != TransformType::C2C
                                ? (LastSelector<double, DDimX, DDimX...>(
                                           ddc::get<DDimX>(mesh.extents()) / 2 + 1,
                                           ddc::get<DDimX>(mesh.extents()))
                                   * ...)
                                : (ddc::get<DDimX>(mesh.extents()) * ...)),
                KOKKOS_LAMBDA(const int& i) { out_data[i] = out_data[i] * norm_coef; });
    }
}
} // namespace ddc::detail::fft

namespace ddc {

/**
 * @brief Initialize a Fourier discrete dimension.
 *
 * Initialize the (1D) discrete space representing the Fourier discrete dimension associated
 * to the (1D) mesh passed as argument. It is a N-periodic PeriodicSampling defined between
 * ka=0 and kb=2*N/(b-a)*pi.
 *
 * This value for kb comes from the Nyquist-Shannon theorem: the period of the spectral domain
 * is kb-ka = 2*pi/cell_size = 2*pi*N/(b-a). The PeriodicSampling then contains cells between coordinates
 * k=0 and k=2*pi*(N-1)/(b-a), because the cell at coordinate k=2*pi*N/(b-a) is a periodic point (f(ka)=f(kb)).
 *  
 * @tparam DDimFx A PeriodicSampling representing the Fourier discrete dimension.
 * @tparam DDimX The type of the original discrete dimension.
 *
 * @param x_mesh The DiscreteDomain representing the (1D) original mesh.
 *
 * @return The initialized Impl representing the discrete Fourier space.
 *
 * @see PeriodicSampling
 */
template <typename DDimFx, typename DDimX>
typename DDimFx::template Impl<DDimFx, Kokkos::HostSpace> init_fourier_space(
        ddc::DiscreteDomain<DDimX> x_mesh)
{
    static_assert(
            is_uniform_point_sampling_v<DDimX>,
            "DDimX dimensions should derive from UniformPointSampling");
    static_assert(
            is_periodic_sampling_v<DDimFx>,
            "DDimFx dimensions should derive from PeriodicSampling");
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

/**
 * @brief Get the Fourier mesh.
 *
 * Compute the Fourier (or spectral) mesh on which the Discrete Fourier Transform of a
 * discrete function is defined.
 *
 * The uid identifies the mode (ie. ddc::DiscreteElement<DDimFx>(0) corresponds to mode 0).
 *
 * @param x_mesh The DiscreteDomain representing the original mesh.
 * @param C2C A flag indicating if a complex-to-complex DFT is going to be performed. Indeed, 
 * in this case the two meshes have same number of points, whereas for real-to-complex
 * or complex-to-real DFT, each complex value of the Fourier-transformed function contains twice more
 * information, and thus only half (actually Nx*Ny*(Nz/2+1) for 3D R2C FFT to take in account mode 0)
 * values are needed (cf. DFT conjugate symmetry property for more information about this).
 *
 * @return The domain representing the Fourier mesh.
 */
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

/** 
 * @brief A structure embedding the configuration of the exposed FFT function with the type of normalization.
 *
 * @see fft, ifft
 */
struct kwArgs_fft
{
    ddc::FFT_Normalization
            normalization; ///< Enum member to identify the type of normalization performed
};

/**
 * @brief Perform a direct Fast Fourier Transform.
 *
 * Compute the discrete Fourier transform of a function using the specialized implementation for the Kokkos::ExecutionSpace
 * of the FFT algorithm.
 *
 * @tparam Tin The type of the input elements (float, Kokkos::complex<float>, double or Kokkos::complex<double>).
 * @tparam Tout The type of the output elements (Kokkos::complex<float> or Kokkos::complex<double>).
 * @tparam DDimFx... The parameter pack of the Fourier discrete dimensions.
 * @tparam DDimX... The parameter pack of the original discrete dimensions.
 * @tparam ExecSpace The type of the Kokkos::ExecutionSpace on which the FFT is performed. It determines which specialized
 * backend is used (ie. fftw, cuFFT...).
 * @tparam MemorySpace The type of the Kokkos::MemorySpace on which are stored the input and output discrete functions.
 * @tparam layout_in The layout of the Chunkspan representing the input discrete function.
 * @tparam layout_out The layout of the Chunkspan representing the output discrete function.
 *
 * @param execSpace The Kokkos::ExecutionSpace on which the FFT is performed.
 * @param out The output discrete function, represented as a ChunkSpan storing values on a spectral mesh.
 * @param in The input discrete function, represented as a ChunkSpan storing values on a mesh.
 * @param kwargs The kwArgs_fft configuring the FFT.
 */
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

    ddc::detail::fft::core<Tin, Tout, ExecSpace, MemorySpace, DDimX...>(
            execSpace,
            out.data_handle(),
            in.data_handle(),
            in.domain(),
            {ddc::FFT_Direction::FORWARD, kwargs.normalization});
}

/**
 * @brief Perform an inverse Fast Fourier Transform.
 *
 * Compute the inverse discrete Fourier transform of a spectral function using the specialized implementation for the Kokkos::ExecutionSpace
 * of the iFFT algorithm.
 *
 * /!\ C2R iFFT does NOT preserve input !
 *
 * @tparam Tin The type of the input elements (Kokkos::complex<float> or Kokkos::complex<double>).
 * @tparam Tout The type of the output elements (float, Kokkos::complex<float>, double or Kokkos::complex<double>).
 * @tparam DDimX... The parameter pack of the original discrete dimensions.
 * @tparam DDimFx... The parameter pack of the Fourier discrete dimensions.
 * @tparam ExecSpace The type of the Kokkos::ExecutionSpace on which the iFFT is performed. It determines which specialized
 * backend is used (ie. fftw, cuFFT...).
 * @tparam MemorySpace The type of the Kokkos::MemorySpace on which are stored the input and output discrete functions.
 * @tparam layout_in The layout of the Chunkspan representing the input discrete function.
 * @tparam layout_out The layout of the Chunkspan representing the output discrete function.
 *
 * @param execSpace The Kokkos::ExecutionSpace on which the iFFT is performed.
 * @param out The output discrete function, represented as a ChunkSpan storing values on a mesh.
 * @param in The input discrete function, represented as a ChunkSpan storing values on a spectral mesh.
 * @param kwargs The kwArgs_fft configuring the iFFT.
 */
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

    ddc::detail::fft::core<Tin, Tout, ExecSpace, MemorySpace, DDimX...>(
            execSpace,
            out.data_handle(),
            in.data_handle(),
            out.domain(),
            {ddc::FFT_Direction::BACKWARD, kwargs.normalization});
}
} // namespace ddc
