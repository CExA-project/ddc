// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <ddc/ddc.hpp>
#include <ddc/kernels/fft.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#if !defined(KOKKOSFFT_ENABLE_SERIAL)
#if defined(KOKKOS_ENABLE_SERIAL) && defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#define KOKKOSFFT_ENABLE_SERIAL
#endif
#endif

#if !defined(KOKKOSFFT_ENABLE_OPENMP)
#if defined(KOKKOS_ENABLE_OPENMP) && defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#define KOKKOSFFT_ENABLE_OPENMP
#endif
#endif

template <typename X>
struct DDim : ddc::UniformPointSampling<X>
{
};

template <typename... DDim>
using DElem = ddc::DiscreteElement<DDim...>;

template <typename... DDim>
using DVect = ddc::DiscreteVector<DDim...>;

template <typename... DDim>
using DDom = ddc::DiscreteDomain<DDim...>;

template <typename Kx>
struct DFDim : ddc::PeriodicSampling<Kx>
{
};

template <typename X>
static void test_fourier_mesh(std::size_t Nx)
{
    double const a = -10;
    double const b = 10;

    DDom<DDim<X>> const x_mesh(ddc::init_discrete_space<DDim<X>>(DDim<X>::template init<DDim<X>>(
            ddc::Coordinate<X>(a + (b - a) / Nx / 2),
            ddc::Coordinate<X>(b - (b - a) / Nx / 2),
            DVect<DDim<X>>(Nx))));
    ddc::init_discrete_space<DFDim<ddc::Fourier<X>>>(
            ddc::init_fourier_space<DFDim<ddc::Fourier<X>>>(ddc::DiscreteDomain<DDim<X>>(x_mesh)));
    DDom<DFDim<ddc::Fourier<X>>> const k_mesh
            = ddc::fourier_mesh<DFDim<ddc::Fourier<X>>>(x_mesh, true);

    double const epsilon = 1e-14;
    ddc::DiscreteElement<DFDim<ddc::Fourier<X>>> const k_front = k_mesh.front();
    for (ddc::DiscreteElement<DFDim<ddc::Fourier<X>>> const k : k_mesh) {
        double const ka = -Kokkos::numbers::pi * Nx / (b - a);
        double const kb = Kokkos::numbers::pi * Nx / (b - a);
        double const k_pred = 2 * (k - k_front) * Kokkos::numbers::pi / (b - a);
        EXPECT_NEAR(
                ddc::coordinate(k),
                k_pred - (kb - ka) * Kokkos::floor((k_pred - ka) / (kb - ka)),
                epsilon);
    }
}

// TODO:
// - FFT multidim but according to a subset of dimensions
template <typename ExecSpace, typename MemorySpace, typename Tin, typename Tout, typename... X>
static void test_fft()
{
    ExecSpace const exec_space;
    bool const full_fft
            = ddc::detail::fft::is_complex_v<Tin> && ddc::detail::fft::is_complex_v<Tout>;
    double const a = -10;
    double const b = 10;
    std::size_t const Nx = 64; // Optimal value is (b-a)^2/(2*pi)

    DDom<DDim<X>...> const x_mesh(ddc::init_discrete_space<DDim<X>>(DDim<X>::template init<DDim<X>>(
            ddc::Coordinate<X>(a + (b - a) / Nx / 2),
            ddc::Coordinate<X>(b - (b - a) / Nx / 2),
            DVect<DDim<X>>(Nx)))...);
    (ddc::init_discrete_space<DFDim<ddc::Fourier<X>>>(
             ddc::init_fourier_space<DFDim<ddc::Fourier<X>>>(ddc::DiscreteDomain<DDim<X>>(x_mesh))),
     ...);
    DDom<DFDim<ddc::Fourier<X>>...> const k_mesh(
            ddc::fourier_mesh<DFDim<ddc::Fourier<X>>...>(x_mesh, full_fft));

    ddc::Chunk f_alloc(x_mesh, ddc::KokkosAllocator<Tin, MemorySpace>());
    ddc::ChunkSpan const f = f_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            f.domain(),
            KOKKOS_LAMBDA(DElem<DDim<X>...> const e) {
                ddc::Real const xn2 = (Kokkos::pow(ddc::coordinate(DElem<DDim<X>>(e)), 2) + ...);
                f(e) = Kokkos::exp(-xn2 / 2);
            });
    ddc::Chunk f_bis_alloc(f.domain(), ddc::KokkosAllocator<Tin, MemorySpace>());
    ddc::ChunkSpan const f_bis = f_bis_alloc.span_view();
    ddc::parallel_deepcopy(f_bis, f);

    ddc::Chunk Ff_alloc(k_mesh, ddc::KokkosAllocator<Tout, MemorySpace>());
    ddc::ChunkSpan const Ff = Ff_alloc.span_view();
    ddc::fft(exec_space, Ff, f_bis, {ddc::FFT_Normalization::FULL});
    Kokkos::fence();

    // deepcopy of Ff because FFT C2R overwrites the input
    ddc::Chunk Ff_bis_alloc(Ff.domain(), ddc::KokkosAllocator<Tout, MemorySpace>());
    ddc::ChunkSpan const Ff_bis = Ff_bis_alloc.span_view();
    ddc::parallel_deepcopy(Ff_bis, Ff);

    ddc::Chunk FFf_alloc(f.domain(), ddc::KokkosAllocator<Tin, MemorySpace>());
    ddc::ChunkSpan const FFf = FFf_alloc.span_view();
    ddc::ifft(exec_space, FFf, Ff_bis, {ddc::FFT_Normalization::FULL});

    ddc::Chunk f_host_alloc(f.domain(), ddc::HostAllocator<Tin>());
    ddc::ChunkSpan const f_host = f_host_alloc.span_view();
    ddc::parallel_deepcopy(f_host, f);

    ddc::Chunk Ff_host_alloc(Ff.domain(), ddc::HostAllocator<Tout>());
    ddc::ChunkSpan const Ff_host = Ff_host_alloc.span_view();
    ddc::parallel_deepcopy(Ff_host, Ff);

    ddc::Chunk FFf_host_alloc(FFf.domain(), ddc::HostAllocator<Tin>());
    ddc::ChunkSpan const FFf_host = FFf_host_alloc.span_view();
    ddc::parallel_deepcopy(FFf_host, FFf);

    auto const pow2 = KOKKOS_LAMBDA(double x)
    {
        return x * x;
    };

    double const criterion = Kokkos::sqrt(ddc::transform_reduce(
            Ff_host.domain(),
            0.,
            ddc::reducer::sum<double>(),
            [=](DElem<DFDim<ddc::Fourier<X>>...> const e) {
                double const xn2 = (pow2(ddc::coordinate(DElem<DFDim<ddc::Fourier<X>>>(e))) + ...);
                double const diff = Kokkos::abs(Ff_host(e)) - Kokkos::exp(-xn2 / 2);
                std::size_t const denom
                        = (ddc::detail::fft::LastSelector<std::size_t, X, X...>(Nx / 2, Nx) * ...);
                return pow2(diff) / denom;
            }));

    double const criterion2 = Kokkos::sqrt(ddc::transform_reduce(
            FFf_host.domain(),
            0.,
            ddc::reducer::sum<double>(),
            [=](DElem<DDim<X>...> const e) {
                double const diff = Kokkos::abs(FFf_host(e)) - Kokkos::abs(f_host(e));
                return pow2(diff) / Kokkos::pow(Nx, sizeof...(X));
            }));
    double const epsilon
            = std::is_same_v<ddc::detail::fft::real_type_t<Tin>, double> ? 1e-15 : 1e-7;
    EXPECT_LE(criterion, epsilon)
            << "Distance between analytical prediction and numerical result : " << criterion;
    EXPECT_LE(criterion2, epsilon)
            << "Distance between input and iFFT(FFT(input)) : " << criterion2;
}

template <typename ExecSpace, typename MemorySpace, typename Tin, typename Tout, typename X>
static void test_fft_norm(ddc::FFT_Normalization const norm)
{
    ExecSpace const exec_space;
    bool const full_fft
            = ddc::detail::fft::is_complex_v<Tin> && ddc::detail::fft::is_complex_v<Tout>;

    DDom<DDim<X>> const x_mesh(ddc::init_discrete_space<DDim<X>>(DDim<X>::template init<DDim<X>>(
            ddc::Coordinate<X>(-1. / 4),
            ddc::Coordinate<X>(1. / 4),
            DVect<DDim<X>>(2))));
    ddc::init_discrete_space<DFDim<ddc::Fourier<X>>>(
            ddc::init_fourier_space<DFDim<ddc::Fourier<X>>>(x_mesh));
    DDom<DFDim<ddc::Fourier<X>>> const k_mesh
            = ddc::fourier_mesh<DFDim<ddc::Fourier<X>>>(x_mesh, full_fft);

    ddc::Chunk f_alloc = ddc::Chunk(x_mesh, ddc::KokkosAllocator<Tin, MemorySpace>());
    ddc::ChunkSpan const f = f_alloc.span_view();
    ddc::parallel_fill(f, Tin(1));

    ddc::Chunk f_bis_alloc(f.domain(), ddc::KokkosAllocator<Tin, MemorySpace>());
    ddc::ChunkSpan const f_bis = f_bis_alloc.span_view();
    ddc::parallel_deepcopy(f_bis, f);

    ddc::Chunk Ff_alloc(k_mesh, ddc::KokkosAllocator<Tout, MemorySpace>());
    ddc::ChunkSpan const Ff = Ff_alloc.span_view();
    ddc::fft(exec_space, Ff, f_bis, {norm});
    Kokkos::fence();

    // deepcopy of Ff because FFT C2R overwrites the input
    ddc::Chunk Ff_bis_alloc(Ff.domain(), ddc::KokkosAllocator<Tout, MemorySpace>());
    ddc::ChunkSpan const Ff_bis = Ff_bis_alloc.span_view();
    ddc::parallel_deepcopy(Ff_bis, Ff);

    ddc::Chunk FFf_alloc(x_mesh, ddc::KokkosAllocator<Tin, MemorySpace>());
    ddc::ChunkSpan const FFf = FFf_alloc.span_view();
    ddc::ifft(exec_space, FFf, Ff_bis, {norm});

    double const f_sum = ddc::transform_reduce(f.domain(), 0., ddc::reducer::sum<double>(), f);

    double Ff0_expected;
    double FFf_expected;
    if (norm == ddc::FFT_Normalization::OFF) {
        Ff0_expected = f_sum;
        FFf_expected = f_sum;
    } else if (norm == ddc::FFT_Normalization::FORWARD) {
        Ff0_expected = 1;
        FFf_expected = 1;
    } else if (norm == ddc::FFT_Normalization::BACKWARD) {
        Ff0_expected = f_sum;
        FFf_expected = 1;
    } else if (norm == ddc::FFT_Normalization::ORTHO) {
        Ff0_expected = Kokkos::sqrt(f_sum);
        FFf_expected = 1;
    } else if (norm == ddc::FFT_Normalization::FULL) {
        Ff0_expected = 1 / Kokkos::sqrt(2 * Kokkos::numbers::pi);
        FFf_expected = 1;
    } else {
        throw std::runtime_error("ddc::FFT_Normalization not handled");
    }

    double const epsilon = 1e-6;
    EXPECT_NEAR(Kokkos::abs(Ff(Ff.domain().front())), Ff0_expected, epsilon);
    EXPECT_NEAR(FFf(FFf.domain().front()), FFf_expected, epsilon);
    EXPECT_NEAR(FFf(FFf.domain().back()), FFf_expected, epsilon);
}

struct RDimX;
struct RDimY;
struct RDimZ;

TEST(FourierMesh, Even)
{
    test_fourier_mesh<RDimX>(16);
}

TEST(FourierMesh, Odd)
{
    test_fourier_mesh<RDimX>(17);
}

#if defined(KOKKOSFFT_ENABLE_SERIAL)
TEST(FftNorm, Off)
{
    test_fft_norm<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX>(ddc::FFT_Normalization::OFF);
}

TEST(FftNorm, Backward)
{
    test_fft_norm<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX>(ddc::FFT_Normalization::BACKWARD);
}

TEST(FftNorm, Forward)
{
    test_fft_norm<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX>(ddc::FFT_Normalization::FORWARD);
}

TEST(FftNorm, Ortho)
{
    test_fft_norm<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX>(ddc::FFT_Normalization::ORTHO);
}

TEST(FftNorm, Full)
{
    test_fft_norm<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX>(ddc::FFT_Normalization::FULL);
}

TEST(FftSerialHost, R2cIn1d)
{
    test_fft<Kokkos::Serial, Kokkos::Serial::memory_space, float, Kokkos::complex<float>, RDimX>();
}

TEST(FftSerialHost, R2cIn2d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX,
            RDimY>();
}

TEST(FftSerialHost, R2cIn3d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

#if defined(KOKKOSFFT_ENABLE_OPENMP)
TEST(FftParallelHost, R2cIn1d)
{
    test_fft<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, float, Kokkos::complex<float>, RDimX>();
}

TEST(FftParallelHost, R2cIn2d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX,
            RDimY>();
}

TEST(FftParallelHost, R2cIn3d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

TEST(FftParallelDevice, R2cIn1d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX>();
}

TEST(FftParallelDevice, R2cIn2d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX,
            RDimY>();
}

TEST(FftParallelDevice, R2cIn3d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            float,
            Kokkos::complex<float>,
            RDimX,
            RDimY,
            RDimZ>();
}

#if defined(KOKKOSFFT_ENABLE_SERIAL)
TEST(FftSerialHost, C2cIn1d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX>();
}

TEST(FftSerialHost, C2cIn2d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX,
            RDimY>();
}

TEST(FftSerialHost, C2cIn3d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

#if defined(KOKKOSFFT_ENABLE_OPENMP)
TEST(FftParallelHost, C2cIn1d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX>();
}

TEST(FftParallelHost, C2cIn2d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX,
            RDimY>();
}

TEST(FftParallelHost, C2cIn3d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

TEST(FftParallelDevice, C2cIn1d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX>();
}

TEST(FftParallelDevice, C2cIn2d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX,
            RDimY>();
}

TEST(FftParallelDevice, C2cIn3d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::complex<float>,
            Kokkos::complex<float>,
            RDimX,
            RDimY,
            RDimZ>();
}

#if defined(KOKKOSFFT_ENABLE_SERIAL)
TEST(FftSerialHost, D2zIn1d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX>();
}

TEST(FftSerialHost, D2zIn2d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX,
            RDimY>();
}

TEST(FftSerialHost, D2zIn3d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

#if defined(KOKKOSFFT_ENABLE_OPENMP)
TEST(FftParallelHost, D2zIn1d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX>();
}

TEST(FftParallelHost, D2zIn2d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX,
            RDimY>();
}

TEST(FftParallelHost, D2zIn3d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

TEST(FftParallelDevice, D2zIn1d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX>();
}

TEST(FftParallelDevice, D2zIn2d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX,
            RDimY>();
}

TEST(FftParallelDevice, D2zIn3d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            double,
            Kokkos::complex<double>,
            RDimX,
            RDimY,
            RDimZ>();
}

#if defined(KOKKOSFFT_ENABLE_SERIAL)
TEST(FftSerialHost, Z2zIn1d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX>();
}

TEST(FftSerialHost, Z2zIn2d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX,
            RDimY>();
}

TEST(FftSerialHost, Z2zIn3d)
{
    test_fft<
            Kokkos::Serial,
            Kokkos::Serial::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

#if defined(KOKKOSFFT_ENABLE_OPENMP)
TEST(FftParallelHost, Z2zIn1d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX>();
}

TEST(FftParallelHost, Z2zIn2d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX,
            RDimY>();
}

TEST(FftParallelHost, Z2zIn3d)
{
    test_fft<
            Kokkos::OpenMP,
            Kokkos::OpenMP::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX,
            RDimY,
            RDimZ>();
}
#endif

TEST(FftParallelDevice, Z2zIn1d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX>();
}

TEST(FftParallelDevice, Z2zIn2d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX,
            RDimY>();
}

TEST(FftParallelDevice, Z2zIn3d)
{
    test_fft<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::complex<double>,
            Kokkos::complex<double>,
            RDimX,
            RDimY,
            RDimZ>();
}
