// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

# if hip_AVAIL
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
# endif

#include <fft.hpp>

struct RDimX;
struct RDimY;
struct RDimZ;

# if 0
__global__ void run_printf() { printf("Hello World\n"); }

static void TestGPUMathToolsParallelDeviceHipHelloWorld()
{
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    for (int i = 0; i < device_count; ++i) {
        hipSetDevice(i);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(run_printf), dim3(10), dim3(10), 0, 0);
        hipDeviceSynchronize();
    }
}

TEST(GPUMathToolsParallelDevice, HipHelloWorld)
{
    TestGPUMathToolsParallelDeviceHipHelloWorld();
}
# endif

TEST(FFTSerialHost, R2C_1D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, float, std::complex<float>, RDimX>();
}

TEST(FFTSerialHost, R2C_2D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, float, std::complex<float>, RDimX, RDimY>();
}

TEST(FFTSerialHost, R2C_3D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, float, std::complex<float>, RDimX, RDimY, RDimZ>();
}

#if fftw_omp_AVAIL
TEST(FFTParallelHost, R2C_1D)
{
	TestFFT<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, float, std::complex<float>, RDimX>();
}

TEST(FFTParallelHost, R2C_2D)
{
	TestFFT<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, float, std::complex<float>, RDimX, RDimY>();
}

TEST(FFTParallelHost, R2C_3D)
{
	TestFFT<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, float, std::complex<float>, RDimX, RDimY, RDimZ>();
}
#endif

TEST(FFTParallelDevice, R2C_1D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, float, std::complex<float>, RDimX>();
}

TEST(FFTParallelDevice, R2C_2D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, float, std::complex<float>, RDimX, RDimY>();
}

TEST(FFTParallelDevice, R2C_3D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, float, std::complex<float>, RDimX, RDimY, RDimZ>();
}

TEST(FFTSerialHost, D2Z_1D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, double, std::complex<double>, RDimX>();
}

TEST(FFTSerialHost, D2Z_2D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, double, std::complex<double>, RDimX, RDimY>();
}

TEST(FFTSerialHost, D2Z_3D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, double, std::complex<double>, RDimX, RDimY, RDimZ>();
}

#if fftw_omp_AVAIL
TEST(FFTParallelHost, D2Z_1D)
{
	TestFFT<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, double, std::complex<double>, RDimX>();
}

TEST(FFTParallelHost, D2Z_2D)
{
	TestFFT<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, double, std::complex<double>, RDimX, RDimY>();
}

TEST(FFTParallelHost, D2Z_3D)
{
	TestFFT<Kokkos::OpenMP, Kokkos::OpenMP::memory_space, double, std::complex<double>, RDimX, RDimY, RDimZ>();
}
#endif

TEST(FFTParallelDevice, D2Z_1D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, double, std::complex<double>, RDimX>();
}

TEST(FFTParallelDevice, D2Z_2D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, double, std::complex<double>, RDimX, RDimY>();
}

TEST(FFTParallelDevice, D2Z_3D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, double, std::complex<double>, RDimX, RDimY, RDimZ>();
}
