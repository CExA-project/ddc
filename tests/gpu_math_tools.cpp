// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipfft/hipfft.h>

struct RDimX;
struct DDimX;
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct RDimY;
struct DDimY;
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

struct RDimKx;
struct DDimKx;

struct RDimKy;
struct DDimKy;

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

template<typename SpatialDom, typename SpectralDom>
void FFT(ddc::ChunkSpan<std::complex<double>, SpectralDom, std::experimental::layout_right, Kokkos::Cuda::memory_space> Ff, ddc::ChunkSpan<double, SpatialDom, std::experimental::layout_right, Kokkos::Cuda::memory_space> f)
{
	// const double a		= -16*M_PI;
	// const double b		= 16*M_PI;
	const int Nx = 32;
	const int Ny = 32;
	# if 0
	using DDimX_res = ddc::UniformPointSampling<RDimX>;
   	using DDimY_res = ddc::UniformPointSampling<RDimY>;
	using DDomX_restricted = ddc::DiscreteDomain<DDimX_res,DDimY_res>;
	DDomX_restricted const x_mesh_restricted = DDomX_restricted(
		ddc::init_discrete_space(DDimX_res::init(ddc::Coordinate<RDimX>(a), ddc::Coordinate<RDimX>(b-(b-a)/(Nx-1)), ddc::DiscreteVector<DDimX_res>(Nx-1))),
		ddc::init_discrete_space(DDimY_res::init(ddc::Coordinate<RDimY>(a), ddc::Coordinate<RDimY>(b-(b-a)/(Ny-1)), ddc::DiscreteVector<DDimY_res>(Ny-1)))
	);
	std::cout << "testztageazegazg"; 
	ddc::Chunk _f_restricted = ddc::Chunk(x_mesh_restricted, ddc::DeviceAllocator<double>());
	ddc::ChunkSpan f_restricted = _f_restricted.span_view();
	ddc::for_each(
		ddc::policies::parallel_device,
		ddc::get_domain<DDimX_res,DDimY_res>(f_restricted),
		DDC_LAMBDA(ddc::DiscreteElement<DDimX_res,DDimY_res> const e) {
			double const x = coordinate(ddc::select<DDimX_res>(e));
			double const y = coordinate(ddc::select<DDimY_res>(e));
			// f(e) = sin(x+1e-20)*sin(y+1e-20)/(x+1e-20)/(y+1e-20);
			// f(e) = cos(4*x)*cos(4*x);
			f_restricted(e) = exp(-(x*x+y*y)/2); //TODO : link to f
	});
	#endif
	hipfftHandle plan      = -1;
    hipfftResult hipfft_rt = hipfftCreate(&plan);
	hipfft_rt = hipfftPlan2d(&plan, // plan handle
                             Nx, 
                             Ny, 
                             HIPFFT_D2Z); // transform type (HIPFFT_C2C for single-precision)
	if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan1d failed");

	hipfft_rt = hipfftExecD2Z(plan, f.data(), (hipfftDoubleComplex*)Ff.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
    	throw std::runtime_error("hipfftExecD2Z failed");
	hipfftDestroy(plan);
}

template <typename... Dims>
struct SpectralSpace;

// TODO:
// - Remove input Kokkos::Cuda
// - Variadic with higher dimension
// - cuFFT+FFTW
template <typename... RDims>
static void TestGPUMathToolsFFT3Dz2z()
{
 	std::cout << "hipfft 3D double-precision complex-to-complex transform\n";

	const double a		= -2*M_PI;
	const double b		= 2*M_PI;
    const int Nx        = 32;
    const int Ny        = 32;

   	using DDimX = ddc::UniformPointSampling<RDimX>;
   	using DDimY = ddc::UniformPointSampling<RDimY>;
	using DDomX = ddc::DiscreteDomain<DDimX,DDimY>;
	DDomX const x_mesh = DDomX(
		ddc::init_discrete_space(DDimX::init(ddc::Coordinate<RDimX>(a+(b-a)/Nx/2), ddc::Coordinate<RDimX>(b-(b-a)/Nx/2), ddc::DiscreteVector<DDimX>(Nx))),
		ddc::init_discrete_space(DDimY::init(ddc::Coordinate<RDimY>(a+(b-a)/Nx/2), ddc::Coordinate<RDimY>(b-(b-a)/Nx/2), ddc::DiscreteVector<DDimY>(Ny)))
	);
	ddc::Chunk _f = ddc::Chunk(x_mesh, ddc::DeviceAllocator<double>());
	ddc::ChunkSpan f = _f.span_view();
	ddc::for_each(
		ddc::policies::parallel_device,
		ddc::get_domain<DDimX,DDimY>(f),
		DDC_LAMBDA(ddc::DiscreteElement<DDimX,DDimY> const e) {
			double const x = coordinate(ddc::select<DDimX>(e));
			double const y = coordinate(ddc::select<DDimY>(e));
			// f(e) = cos(4*x)*cos(4*y);
			// f(e) = sin(x+1e-20)*sin(y+1e-20)/(x+1e-20)/(y+1e-20);
			f(e) = exp(-(x*x+y*y)/2);
		}
	);

	using DDimKx = ddc::UniformPointSampling<RDimKx>;
   	using DDimKy = ddc::UniformPointSampling<RDimKy>;
	using DDomK = ddc::DiscreteDomain<DDimKx,DDimKy>;
	DDomK const k_mesh = DDomK(
		ddc::init_discrete_space(DDimKx::init(ddc::Coordinate<RDimKx>(0), ddc::Coordinate<RDimKx>((Nx-1)/(b-a)*M_PI), ddc::DiscreteVector<DDimKx>(Nx/2+1))),
		ddc::init_discrete_space(DDimKy::init(ddc::Coordinate<RDimKy>(0), ddc::Coordinate<RDimKy>((Ny-1)/(b-a)*M_PI), ddc::DiscreteVector<DDimKy>(Ny/2+1)))
	);
	ddc::Chunk _Ff = ddc::Chunk(k_mesh, ddc::DeviceAllocator<std::complex<double>>());
	ddc::ChunkSpan Ff = _Ff.span_view();
	FFT<DDomX, DDomK>(Ff, f);

	ddc::Chunk _f_host = ddc::Chunk(ddc::get_domain<DDimX,DDimY>(f), ddc::HostAllocator<double>());
    ddc::ChunkSpan f_host = _f_host.span_view();
	ddc::deepcopy(f_host, f);
	# if 1
	std::cout << "\n input:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DDimX,DDimY>(f_host),
        [=](ddc::DiscreteElement<DDimX,DDimY> const e) {
			std::cout << coordinate(ddc::select<DDimX>(e)) << coordinate(ddc::select<DDimY>(e)) << "->" << f_host(e) << ", ";
	});
    # endif

	ddc::Chunk _Ff_host = ddc::Chunk(ddc::get_domain<DDimKx,DDimKy>(Ff), ddc::HostAllocator<std::complex<double>>());
    ddc::ChunkSpan Ff_host = _Ff_host.span_view();
	ddc::deepcopy(Ff_host, Ff);
	# if 1
	std::cout << "\n output:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DDimKx,DDimKy>(Ff_host),
        [=](ddc::DiscreteElement<DDimKx,DDimKy> const e) {
			double const kx = coordinate(ddc::select<DDimKx>(e));
			double const ky = coordinate(ddc::select<DDimKy>(e));
			std::cout << "(" << kx << ", " << ky << ") ->" << abs(Ff_host(e))*1/2/M_PI*(b-a)*(b-a)/Nx/Ny << " " << exp(-(kx*kx+ky*ky)/2) << ", ";
	});
	# endif
	double error_squared = ddc::transform_reduce(
		ddc::get_domain<DDimKx,DDimKy>(Ff_host),
		0.,
		ddc::reducer::sum<double>(),
		[=](ddc::DiscreteElement<DDimKx,DDimKy> const e) {
			double const kx = coordinate(ddc::select<DDimKx>(e));
			double const ky = coordinate(ddc::select<DDimKy>(e));
			return pow((abs(Ff_host(e))*1/2/M_PI*(b-a)*(b-a)/Nx/Ny)-exp(-(kx*kx+ky*ky)/2),2)/Nx/Ny;
	
	});
	ASSERT_LE(sqrt(error_squared), 1e-2);
}

TEST(GPUMathToolsParallelDevice, FFT3Dz2z)
{
	TestGPUMathToolsFFT3Dz2z<RDimX, RDimY>();
}
