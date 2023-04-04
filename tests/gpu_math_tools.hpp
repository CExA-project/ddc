// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipfft/hipfft.h>

template <typename RDim>
using DDim = ddc::UniformPointSampling<RDim>;

template <typename ...DDim>
using DElem = ddc::DiscreteElement<DDim...>;

template <typename ...DDim>
using DVect = ddc::DiscreteVector<DDim...>;

template <typename ...DDim>
using DDom = ddc::DiscreteDomain<DDim...>;

template <typename Dims>
struct K;

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

// template <typename RDim>
// using SpectralDim = SpectralSpace<RDim>;


// TODO:
// - Remove input Kokkos::Cuda
// - Variadic with higher dimension
// - cuFFT+FFTW
template <typename... RDim>
static void TestGPUMathToolsFFT3Dz2z()
{
	const double a		= -2*M_PI;
	const double b		= 2*M_PI;
    const int Nx        = 32;
    const int Ny        = 32;

	DDom<DDim<RDim>...> const x_mesh = DDom<DDim<RDim>...>(
		ddc::init_discrete_space(DDim<RDim>::init(ddc::Coordinate<RDim>(a+(b-a)/Nx/2), ddc::Coordinate<RDim>(b-(b-a)/Nx/2), DVect<DDim<RDim>>(Nx)))...
	);
#if 0
	ddc::Chunk _f = ddc::Chunk(x_mesh, ddc::DeviceAllocator<double>());
	ddc::ChunkSpan f = _f.span_view();
	ddc::for_each(
		ddc::policies::parallel_device,
		ddc::get_domain<DDim<RDim>...>(f),
		DDC_LAMBDA(DElem<DDim<RDim>...> const e) {
			// double const x = coordinate(ddc::select<DDimX>(e));
			// double const y = coordinate(ddc::select<DDimY>(e));
			// f(e) = cos(4*x)*cos(4*y);
			// f(e) = sin(x+1e-20)*sin(y+1e-20)/(x+1e-20)/(y+1e-20);
			f(e) = exp(-(pow(coordinate(ddc::select<DDim<RDim>>(e)),2) + ...)/2);
		}
	);
	DDom<RDim...> const k_mesh = DDomK<RDim...>(
		ddc::init_discrete_space(DiscreteDim<Fourier<RDim>>::init(ddc::Coordinate<Fourier<RDim>>(0), ddc::Coordinate<Fourier<RDim>>((Nx-1)/(b-a)*M_PI), ddc::DiscreteVector<DiscreteDim<Fourier<RDim>>>(Nx/2+1))) ...
	);
	ddc::Chunk _Ff = ddc::Chunk(k_mesh, ddc::DeviceAllocator<std::complex<double>>());
	ddc::ChunkSpan Ff = _Ff.span_view();
	// FFT<DDomX, DDomK>(Ff, f);

	ddc::Chunk _f_host = ddc::Chunk(ddc::get_domain<DiscreteDim<RDim>...>(f), ddc::HostAllocator<double>());
    ddc::ChunkSpan f_host = _f_host.span_view();
	ddc::deepcopy(f_host, f);
	# if 1
	std::cout << "\n input:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DiscreteDim<RDim>...>(f_host),
        [=](ddc::DiscreteElement<DiscreteDim<RDim>...> const e) {
			std::cout << f_host(e) << ", ";
			// std::cout << coordinate(ddc::select<DDimX>(e)) << coordinate(ddc::select<DDimY>(e)) << "->" << f_host(e) << ", ";
	});
    # endif

	ddc::Chunk _Ff_host = ddc::Chunk(ddc::get_domain<DiscreteDim<Fourier<RDim>>...>(Ff), ddc::HostAllocator<std::complex<double>>());
    ddc::ChunkSpan Ff_host = _Ff_host.span_view();
	ddc::deepcopy(Ff_host, Ff);
#endif
	# if 0
	std::cout << "\n output:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DDimKx,DDimKy>(Ff_host),
        [=](ddc::DiscreteElement<DDimKx,DDimKy> const e) {
			double const kx = coordinate(ddc::select<DDimKx>(e));
			double const ky = coordinate(ddc::select<DDimKy>(e));
			std::cout << "(" << kx << ", " << ky << ") ->" << abs(Ff_host(e))*1/2/M_PI*(b-a)*(b-a)/Nx/Ny << " " << exp(-(kx*kx+ky*ky)/2) << ", ";
	});
	double error_squared = ddc::transform_reduce(
		ddc::get_domain<DDimK...>(Ff_host),
		0.,
		ddc::reducer::sum<double>(),
		[=](ddc::DiscreteElement<DDimK..> const e) {
			double const kx = coordinate(ddc::select<DDimKx>(e));
			double const ky = coordinate(ddc::select<DDimKy>(e));
			return pow((abs(Ff_host(e))*1/2/M_PI*(b-a)*(b-a)/Nx/Ny)-exp(-(kx*kx+ky*ky)/2),2)/Nx/Ny;
	
	});
	ASSERT_LE(sqrt(error_squared), 1e-2);
	# endif
}

