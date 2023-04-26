// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

# if fftw_AVAIL 
#include <fftw3.h>
# endif

# if cufft_AVAIL 
#include <cufft.h>
# endif

# if hipfft_AVAIL 
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipfft/hipfft.h>
# endif

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

template <typename Kx>
using DFDim = ddc::FourierSampling<Kx>;

template<typename T, typename... X, typename ExecSpace, typename MemorySpace>
void FFT(ExecSpace execSpace, ddc::ChunkSpan<std::complex<T>, DDom<DFDim<K<X>>...>, std::experimental::layout_right, MemorySpace> Ff,
	     ddc::ChunkSpan<T, DDom<DDim<X>...>, std::experimental::layout_right, MemorySpace> f)
{
	static_assert(Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible,"MemorySpace has to be accessible for ExecutionSpace.");
	DDom<DDim<X>...> x_mesh = ddc::get_domain<DDim<X>...>(f);
	
	int n[x_mesh.rank()] = {(int)ddc::get<DDim<X>>(x_mesh.extents())...};
	int idist = 1;
	int odist = 1;
	for(int i=0;i<x_mesh.rank();i++) {	
		idist = idist*n[i];
		odist = odist*(n[i]/2+1); //Correct this
	}
	if constexpr(false) {} // Trick to get only else if
	# if fftw_omp_AVAIL 
	else if constexpr(std::is_same<ExecSpace, Kokkos::OpenMP>::value) {
		fftw_init_threads();
		fftw_plan_with_nthreads(omp_get_max_threads());
		fftw_plan plan = fftw_plan_many_dft_r2c(x_mesh.rank(), 
							n, 
							1,
							f.data(),
							NULL,
							1,
							idist,
							(fftw_complex*)Ff.data(),
							NULL,
							1,
							odist,
							0);
		fftw_execute(plan);
		fftw_destroy_plan(plan);
		std::cout << "performed with fftw_omp";
	}
	# endif
	# if fftw_AVAIL 
	else if constexpr(std::is_same<ExecSpace, Kokkos::Serial>::value) {
		fftw_plan plan = fftw_plan_many_dft_r2c(x_mesh.rank(), 
							n, 
							1,
							f.data(),
							NULL,
							1,
							idist,
							(fftw_complex*)Ff.data(),
							NULL,
							1,
							odist,
							0);
		fftw_execute(plan);
		fftw_destroy_plan(plan);
		std::cout << "performed with fftw";
	}
	# endif
	# if cufft_AVAIL 
	else if constexpr(std::is_same<ExecSpace, Kokkos::Cuda>::value) {
		cudaStream_t stream = execSpace.cuda_stream();

		cufftHandle plan = -1;
		cufftResult cufft_rt = cufftCreate(&plan);
		cufftSetStream(plan, stream);
		cufft_rt = cufftPlanMany(&plan, // plan handle
							 x_mesh.rank(),
							 n, // Nx, Ny...
							 NULL,
							 1,
							 idist,
							 NULL,
							 1,
							 odist,
							 CUFFT_D2Z,
							 1); 

		if(cufft_rt != CUFFT_SUCCESS)
			throw std::runtime_error("cufftPlan1d failed");

		cufft_rt = cufftExecD2Z(plan, f.data(), (cufftDoubleComplex*)Ff.data());
		if(cufft_rt != CUFFT_SUCCESS)
			throw std::runtime_error("cufftExecD2Z failed");
		cufftDestroy(plan);
		std::cout << "performed with cufft";
	}
	# endif
	# if hipfft_AVAIL 
	else if constexpr(std::is_same<ExecSpace, Kokkos::Cuda>::value) {
	// else if constexpr(std::is_same<ExecSpace, Kokkos::HIP::value) {
		hipStream_t stream = execSpace.cuda_stream();
		// hipStream_t stream = execSpace.hip_stream();

		hipfftHandle plan = -1;
		hipfftResult hipfft_rt = hipfftCreate(&plan);
		hipfftSetStream(plan, stream);
		hipfft_rt = hipfftPlanMany(&plan, // plan handle
							 x_mesh.rank(),
							 n, // Nx, Ny...
							 NULL,
							 1,
							 idist,
							 NULL,
							 1,
							 odist,
							 HIPFFT_D2Z,
							 1); 

		if(hipfft_rt != HIPFFT_SUCCESS)
			throw std::runtime_error("hipfftPlan1d failed");

		hipfft_rt = hipfftExecD2Z(plan, f.data(), (hipfftDoubleComplex*)Ff.data());
		if(hipfft_rt != HIPFFT_SUCCESS)
			throw std::runtime_error("hipfftExecD2Z failed");
		hipfftDestroy(plan);
		std::cout << "performed with hipfft";
	}
	# endif
}

