// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipfft/hipfft.h>

struct X;
struct DDimX;
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct DDimY;
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

struct Kx;
struct DDimKx;

static DElemX constexpr lbound_x(0);
static DVectX constexpr nelems_x(10);

static DElemY constexpr lbound_y(0);
static DVectY constexpr nelems_y(12);

static DElemXY constexpr lbound_x_y {lbound_x, lbound_y};
static DVectXY constexpr nelems_x_y(nelems_x, nelems_y);
TEST(GPUMathToolsSerialHost, Empty)
{
    DDomX const dom(lbound_x, DVectX(0));
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> view(storage.data(), dom);
    ddc::for_each(ddc::policies::serial_host, dom, [=](DElemX const ix) { view(ix) += 1; });
    ASSERT_EQ(std::count(storage.begin(), storage.end(), 1), dom.size());
    std::cout << std::count(storage.begin(), storage.end(), 1) << std::endl;
}

TEST(GPUMathToolsParallelHost, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    std::vector<int> storage(dom.size(), 0);
    ddc::ChunkSpan<int, DDomX> view(storage.data(), dom);
    ddc::for_each(ddc::policies::parallel_host, dom, [=](DElemX const ix) { view(ix) += 1; });
    ASSERT_EQ(std::count(storage.begin(), storage.end(), 1), dom.size());
}

static void TestGPUMathToolsParallelDeviceOneDimension()
{
    DDomX const dom(lbound_x, nelems_x);
    ddc::Chunk<int, DDomX, ddc::DeviceAllocator<int>> storage(dom);
    Kokkos::deep_copy(storage.allocation_kokkos_view(), 0);
    ddc::ChunkSpan view(storage.span_view());
    ddc::for_each(
            ddc::policies::parallel_device,
            dom,
            DDC_LAMBDA(DElemX const ix) { view(ix) += 1; });
    int const* const ptr = storage.data();
    int sum;
    Kokkos::parallel_reduce(
            dom.size(),
            KOKKOS_LAMBDA(std::size_t i, int& local_sum) { local_sum += ptr[i]; },
            Kokkos::Sum<int>(sum));
    ASSERT_EQ(sum, dom.size());
}

TEST(GPUMathToolsParallelDevice, OneDimension)
{
    TestGPUMathToolsParallelDeviceOneDimension();
}

static void TestGPUMathToolsParallelDeviceTwoDimensions()
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    ddc::Chunk<int, DDomXY, ddc::DeviceAllocator<int>> storage(dom);
    Kokkos::deep_copy(storage.allocation_kokkos_view(), 0);
    ddc::ChunkSpan view(storage.span_view());
    ddc::for_each(
            ddc::policies::parallel_device,
            dom,
            DDC_LAMBDA(DElemXY const ixy) { view(ixy) += 1; });
    int const* const ptr = storage.data();
    int sum;
    Kokkos::parallel_reduce(
            dom.size(),
            KOKKOS_LAMBDA(std::size_t i, int& local_sum) { local_sum += ptr[i]; },
            Kokkos::Sum<int>(sum));
    ASSERT_EQ(sum, dom.size());
}

TEST(GPUMathToolsParallelDevice, TwoDimensions)
{
    TestGPUMathToolsParallelDeviceTwoDimensions();
}

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

static void TestGPUMathToolsFFT3Dz2z()
{
 	std::cout << "hipfft 3D double-precision complex-to-complex transform\n";

	const double a		= -10*M_PI;
	const double b		= 10*M_PI;
    const int Nx        = 201;
    const int Ny        = 4;
    const int Nz        = 4;
    int       direction = HIPFFT_FORWARD; // forward=-1, backward=1

    size_t complex_bytes = sizeof(std::complex<double>) * (Nx/2+1);

   	using DDimX = ddc::UniformPointSampling<X>;
	ddc::DiscreteDomain<DDimX> const x_mesh = ddc::init_discrete_space(
		DDimX::init(ddc::Coordinate<X>(a), ddc::Coordinate<X>(b), ddc::DiscreteVector<DDimX>(Nx)));
   	using DDimKx = ddc::UniformPointSampling<Kx>;
	ddc::DiscreteDomain<DDimKx> const k_mesh = ddc::init_discrete_space(
		DDimKx::init(ddc::Coordinate<Kx>(0), ddc::Coordinate<Kx>((Nx-1)/(b-a)*M_PI), ddc::DiscreteVector<DDimKx>(Nx/2+1)));
	ddc::ChunkSpan const f = ddc::Chunk(x_mesh, ddc::DeviceAllocator<double>()).span_view();
	ddc::for_each(
		ddc::policies::parallel_device,
		x_mesh,
		DDC_LAMBDA(ddc::DiscreteElement<DDimX> const Ex) {
			double const x = coordinate(ddc::select<DDimX>(Ex));
			f(Ex) = sin(x)/(x+1e-20);
			// f(Ex) = cos(4*x);
		}
	);
    hipfftHandle ddc_plan      = -1;
    hipfftResult hipfft_ddc_rt = hipfftCreate(&ddc_plan);
	hipfft_ddc_rt = hipfftPlan1d(&ddc_plan, // plan handle
                             Nx, // transform length
                             HIPFFT_D2Z, 1); // transform type (HIPFFT_C2C for single-precision)
	if(hipfft_ddc_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan1d failed");

	// Create HIP device object and copy data to device:
    // hipfftComplex for single-precision
    hipError_t hip_ddc_rt;
    hipfftDoubleComplex* Ff;
    hip_ddc_rt = hipMalloc(&Ff, complex_bytes);
    if(hip_ddc_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");
	
	// ddc::ChunkSpan const Ff = ddc::Chunk(k_mesh, ddc::DeviceAllocator<std::complex<double>>()).span_view();
	hipfft_ddc_rt = hipfftExecD2Z(ddc_plan, f.data(), Ff);
    if(hipfft_ddc_rt != HIPFFT_SUCCESS)
    	throw std::runtime_error("hipfftExecD2Z failed");
	

	std::vector<double> f_host(Nx);
	hip_ddc_rt = hipMemcpy(f_host.data(), f.data(), sizeof(double)*Nx, hipMemcpyDeviceToHost);
	if(hip_ddc_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
	std::cout << "input:\n";
    for(int i = 0; i < Nx; i++)
    {
       std::cout << coordinate(ddc::select<DDimX>(x_mesh[i])) << "->" << f_host[i] << " ";
    }
    std::cout << std::endl;
	std::vector<std::complex<double>> Ff_host(Nx);
    hip_ddc_rt = hipMemcpy(Ff_host.data(), Ff, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_ddc_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");	
	std::cout << "output:\n";
    for(int i = 0; i < Nx/2+1; i++)
    {
    	std::cout << coordinate(ddc::select<DDimKx>(k_mesh[i])) << "->" << abs(Ff_host[i]) << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(ddc_plan);
    hipFree(Ff);

	#if 0
	// Create HIP device object and copy data to device:
    // hipfftComplex for single-precision
    hipError_t           hip_rt;
    hipfftDoubleComplex* x;
    hip_rt = hipMalloc(&x, complex_bytes);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMalloc failed");

    hip_rt = hipMemcpy(x, cdata.data(), complex_bytes, hipMemcpyHostToDevice);

    std::cout << "Input:\n";
    for(size_t i = 0; i < Nx * Ny * Nz; i++)
    {
        cdata[i] = i;
    }
    for(int i = 0; i < Nx; i++)
    {
        for(int j = 0; j < Ny; j++)
        {
            for(int k = 0; k < Nz; k++)
            {
                int pos = (i * Ny + j) * Nz + k;
                std::cout << cdata[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
	// auto mdspan_rt = std::experimental::mdspan(cdata.data(), Nx, Ny, Nz);

    hip_rt = hipMemcpy(x, cdata.data(), complex_bytes, hipMemcpyHostToDevice);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");

	// Create plan
    hipfftHandle plan      = -1;
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("failed to create plan");

    hipfft_rt = hipfftPlan3d(&plan, // plan handle
                             Nx, // transform length
                             Ny, // transform length
                             Nz, // transform length
                             HIPFFT_Z2Z); // transform type (HIPFFT_C2C for single-precision)
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan3d failed");

    // Execute plan
    // hipfftExecZ2Z: double precision, hipfftExecC2C: for single-precision
    hipfft_rt = hipfftExecZ2Z(plan, x, x, direction);
    if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftExecZ2Z failed");

    std::cout << "output:\n";
    hip_rt = hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    if(hip_rt != hipSuccess)
        throw std::runtime_error("hipMemcpy failed");
    for(int i = 0; i < Nx; i++)
    {
        for(int j = 0; j < Ny; j++)
        {
            for(int k = 0; k < Nz; k++)
            {
                int pos = (i * Ny + j) * Nz + k;
                std::cout << cdata[pos] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);
    hipFree(x);
#endif
}

TEST(GPUMathToolsParallelDevice, FFT3Dz2z)
{
	TestGPUMathToolsFFT3Dz2z();
}
