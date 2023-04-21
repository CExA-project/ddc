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

TEST(FFTSerialHost, 1D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, double, RDimX>();
}

TEST(FFTSerialHost, 2D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, double, RDimX, RDimY>();
}

TEST(FFTSerialHost, 3D)
{
	TestFFT<Kokkos::Serial, Kokkos::Serial::memory_space, double, RDimX, RDimY, RDimZ>();
}

TEST(FFTParallelDevice, 1D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, double, RDimX>();
}

TEST(FFTParallelDevice, 2D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, double, RDimX, RDimY>();
}

TEST(FFTParallelDevice, 3D)
{
	TestFFT<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, double, RDimX, RDimY, RDimZ>();
}
