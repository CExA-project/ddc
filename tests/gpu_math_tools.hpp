// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipfft/hipfft.h>

template <typename X>
using DDim = ddc::UniformPointSampling<X>;

template <typename ...DDim>
using DElem = ddc::DiscreteElement<DDim...>;

template <typename ...DDim>
using DVect = ddc::DiscreteVector<DDim...>;

template <typename ...DDim>
using DDom = ddc::DiscreteDomain<DDim...>;

template <typename Dims>
struct K;

template<typename... X>
void FFT(ddc::ChunkSpan<std::complex<double>, DDom<DDim<K<X>>...>, std::experimental::layout_right, Kokkos::Cuda::memory_space> Ff, ddc::ChunkSpan<double, DDom<DDim<X>...>, std::experimental::layout_right, Kokkos::Cuda::memory_space> f)
{
	DDom<DDim<X>...> x_mesh = ddc::get_domain<DDim<X>...>(f);
   	hipfftHandle plan = -1;
	hipfftResult hipfft_rt = hipfftCreate(&plan);
	
	int n[x_mesh.rank()] = {(int)ddc::get<DDim<X>>(x_mesh.extents())...};
	hipfft_rt = hipfftPlanMany(&plan, // plan handle
						 x_mesh.rank(),
                         n, // Nx, Ny...
						 NULL,
						 1,
						 1,
						 NULL,
						 1,
						 1,
						 HIPFFT_D2Z,
						 1); 

	if(hipfft_rt != HIPFFT_SUCCESS)
        throw std::runtime_error("hipfftPlan1d failed");

	hipfft_rt = hipfftExecD2Z(plan, f.data(), (hipfftDoubleComplex*)Ff.data());
    if(hipfft_rt != HIPFFT_SUCCESS)
    	throw std::runtime_error("hipfftExecD2Z failed");
	hipfftDestroy(plan);
}

// TODO:
// - cuFFT+FFTW
// - FFT multidim but according to a subset of dimensions
template <typename... X>
static void TestGPUMathToolsFFT()
{
	const double a		= -2*M_PI;
	const double b		= 2*M_PI;
    const int Nx        = 32;

	DDom<DDim<X>...> const x_mesh = DDom<DDim<X>...>(
		ddc::init_discrete_space(DDim<X>::init(ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, X>(a+(b-a)/Nx/2), ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, X>(b-(b-a)/Nx/2), DVect<DDim<X>>(Nx)))...
	);
	ddc::Chunk _f = ddc::Chunk(x_mesh, ddc::DeviceAllocator<double>());
	ddc::ChunkSpan f = _f.span_view();
	ddc::for_each(
		ddc::policies::parallel_device,
		ddc::get_domain<DDim<X>...>(f),
		DDC_LAMBDA(DElem<DDim<X>...> const e) {
			// f(e) = (cos(4*coordinate(ddc::select<DDim<X>>(e)))*...);
			// f(e) = ((sin(coordinate(ddc::select<DDim<X>>(e))+1e-20)/(coordinate(ddc::select<DDim<X>>(e))+1e-20))*...);
			f(e) = exp(-(pow(coordinate(ddc::select<DDim<X>>(e)),2) + ...)/2);
		}
	);
	DDom<DDim<K<X>>...> const k_mesh = DDom<DDim<K<X>>...>(
		ddc::init_discrete_space(DDim<K<X>>::init(ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, K<X>>(0), ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, K<X>>((Nx-1)/(b-a)*M_PI), ddc::DiscreteVector<DDim<K<X>>>(Nx/2+1)))...
	);
	ddc::Chunk _Ff = ddc::Chunk(k_mesh, ddc::DeviceAllocator<std::complex<double>>());
	ddc::ChunkSpan Ff = _Ff.span_view();
	FFT<X...>(Ff, f);

	ddc::Chunk _f_host = ddc::Chunk(ddc::get_domain<DDim<X>...>(f), ddc::HostAllocator<double>());
    ddc::ChunkSpan f_host = _f_host.span_view();
	ddc::deepcopy(f_host, f);
	# if 0
	std::cout << "\n input:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DDim<X>...>(f_host),
        [=](DElem<DDim<X>...> const e) {
			(std::cout << ... << coordinate(ddc::select<DDim<X>>(e))) << "->" << f_host(e) << ", ";
	});
    # endif

	ddc::Chunk _Ff_host = ddc::Chunk(ddc::get_domain<DDim<K<X>>...>(Ff), ddc::HostAllocator<std::complex<double>>());
    ddc::ChunkSpan Ff_host = _Ff_host.span_view();
	ddc::deepcopy(Ff_host, Ff);
	# if 0
	std::cout << "\n output:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DDim<K<X>>...>(Ff_host),
        [=](DElem<DDim<K<X>>...> const e) {
			(std::cout << ... << coordinate(ddc::select<DDim<K<X>>>(e))) << "->" << abs(Ff_host(e))*pow((b-a)/Nx/sqrt(2*M_PI),sizeof...(X)) << " " << exp(-(pow(coordinate(ddc::select<DDim<K<X>>>(e)),2) + ...)/2) << ", ";
	});
	# endif
	double criterion = sqrt(ddc::transform_reduce(
		ddc::get_domain<DDim<K<X>>...>(Ff_host),
		0.,
		ddc::reducer::sum<double>(),
		[=](DElem<DDim<K<X>>...> const e) {
			return pow(real(Ff_host(e))*pow((b-a)/Nx/sqrt(2*M_PI),sizeof...(X))-exp(-(pow(coordinate(ddc::select<DDim<K<X>>>(e)),2) + ...)/2),2)/pow(Nx,sizeof...(X));
	
	}));
	std::cout << "\n Distance between analytical prediction and numerical result : " << criterion;
	ASSERT_LE(criterion, 0.5);
}

