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

// Macro to orient to A or B exeution code according to type of T (float or not). Trick necessary because of the old convention of fftw (need to add an "f" in all functions or types names for the float version)
#define execIfFloat(T,A,B) if constexpr (std::is_same_v<typename base_type<T>::type,float>) { A; } else { B; }

template<typename T>
struct base_type { using type = T; };

template<typename T>
struct base_type<std::complex<T>> { using type = T; };

// is_complex : trait to determine if type is std::complex<something>
template<typename T>
struct is_complex : std::false_type {};

template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};

// transform_type : trait to determine the type of transformation (R2C, C2R, C2C...) <- no information about base type (float or double)
enum class TransformType { R2R, R2C, C2R };

template<typename T1, typename T2>
struct transform_type { static constexpr TransformType value = TransformType::R2R; };

template<typename T1, typename T2>
struct transform_type<T1,std::complex<T2>> { static constexpr TransformType value = TransformType::R2C; };

template<typename T1, typename T2>
struct transform_type<std::complex<T1>,T2> { static constexpr TransformType value = TransformType::C2R; };

#if fftw_AVAIL
// _fftw_type : compatible with both single and double precision
template<typename T>
struct _fftw_type {
  using type = T;
};

template<typename T>
struct _fftw_type<std::complex<T>> {
  using type = typename std::conditional<std::is_same_v<typename base_type<T>::type,float>,fftwf_complex,fftw_complex>::type;
};

// _fftw_plan : compatible with both single and double precision
template<typename T>
using _fftw_plan = typename std::conditional<std::is_same_v<typename base_type<T>::type,float>,fftwf_plan,fftw_plan>::type;

// _fftw_plan_many_dft : templated function working for all types of transformation
template<typename Tin, typename Tout, typename... ArgType>
_fftw_plan<Tin> _fftw_plan_many_dft(ArgType... args) {
  const TransformType transformType = transform_type<Tin,Tout>::value; 
  if constexpr (transformType==TransformType::R2C&&std::is_same_v<typename base_type<Tin>::type,float>)
	return fftwf_plan_many_dft_r2c(args...);
  else if constexpr (transformType==TransformType::R2C&&std::is_same_v<typename base_type<Tin>::type,double>)
	return fftw_plan_many_dft_r2c(args...);
  else if constexpr (transformType==TransformType::C2R&&std::is_same_v<typename base_type<Tout>::type,float>)
	return fftwf_plan_many_dft_c2r(args...);
  else if constexpr (transformType==TransformType::C2R&&std::is_same_v<typename base_type<Tout>::type,double>)
	return fftw_plan_many_dft_c2r(args...);
  // else constexpr
  //   static_assert(false, "Transform type not supported");
}
#endif
#if cufft_AVAIL
// _cufft_type : compatible with both single and double precision
template<typename T>
struct _cufft_type {
  using type = T;
};

template<typename T>
struct _cufft_type<std::complex<T>> {
  using type = typename std::conditional<std::is_same_v<typename base_type<T>::type,float>,cufftComplex,cufftDoubleComplex>::type;
};

// cufft_transform_type : argument passed in the cufftMakePlan function
template<typename Tin, typename Tout>
constexpr auto cufft_transform_type()  { 
  const TransformType transformType = transform_type<Tin,Tout>::value; 
  if constexpr (transformType==TransformType::R2C&&std::is_same_v<typename base_type<Tin>::type,float>)
	return CUFFT_R2C;
  else if constexpr (transformType==TransformType::R2C&&std::is_same_v<typename base_type<Tin>::type,double>)
	return CUFFT_D2Z;
  else if constexpr (transformType==TransformType::C2R&&std::is_same_v<typename base_type<Tout>::type,float>)
	return CUFFT_C2R;
  else if constexpr (transformType==TransformType::C2R&&std::is_same_v<typename base_type<Tout>::type,double>)
	return CUFFT_Z2D;
  // else constexpr
//	static_assert(false, "Transform type not supported");
}

// cufftExec : argument passed in the cufftMakePlan function
// _fftw_plan_many_dft : templated function working for all types of transformation
template<typename Tin, typename Tout, typename... ArgType>
cufftResult _cufftExec(ArgType... args) {
  const TransformType transformType = transform_type<Tin,Tout>::value; 
  if constexpr (transformType==TransformType::R2C&&std::is_same_v<typename base_type<Tin>::type,float>)
	return cufftExecR2C(args...);
  else if constexpr (transformType==TransformType::R2C&&std::is_same_v<typename base_type<Tin>::type,double>)
	return cufftExecD2Z(args...);
  else if constexpr (transformType==TransformType::C2R&&std::is_same_v<typename base_type<Tout>::type,float>)
	return cufftExecC2R(args...);
  else if constexpr (transformType==TransformType::C2R&&std::is_same_v<typename base_type<Tout>::type,double>)
	return cufftExeZ2D(args...);
  // else constexpr
  //   static_assert(false, "Transform type not supported");
}
#endif

template <typename Dims>
struct K;

template<typename Tin, typename Tout, typename... X, typename ExecSpace, typename MemorySpace>
void FFT(ExecSpace execSpace, ddc::ChunkSpan<Tout, ddc::DiscreteDomain<ddc::PeriodicSampling<K<X>>...>, std::experimental::layout_right, MemorySpace> Ff,
	     ddc::ChunkSpan<Tin, ddc::DiscreteDomain<ddc::UniformPointSampling<X>...>, std::experimental::layout_right, MemorySpace> f)
{
	static_assert(std::is_same_v<typename base_type<Tin>::type,float> || std::is_same_v<typename base_type<Tin>::type,double>,"Base type of Tin (and Tout) must be float or double.");
	static_assert(std::is_same_v<typename base_type<Tin>::type,typename base_type<Tout>::type>,"Types Tin and Tout must be based on same type (float or double)");
	static_assert(Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible,"MemorySpace has to be accessible for ExecutionSpace.");
	ddc::DiscreteDomain<ddc::UniformPointSampling<X>...> x_mesh = ddc::get_domain<ddc::UniformPointSampling<X>...>(f);
	
	int n[x_mesh.rank()] = {(int)ddc::get<ddc::UniformPointSampling<X>>(x_mesh.extents())...};
	int idist = 1;
	int odist = 1;
	for(int i=0;i<x_mesh.rank();i++) {	
		idist = transform_type<Tin,Tout>::value==TransformType::C2R&&i==0 ? idist*(n[i]/2+1) : idist*n[i];
		odist = transform_type<Tin,Tout>::value==TransformType::R2C&&i==0 ? odist*(n[i]/2+1) : odist*n[i];
	}

	if constexpr(false) {} // Trick to get only else if
	# if fftw_AVAIL 
	else if constexpr(std::is_same<ExecSpace, Kokkos::Serial>::value) {
		_fftw_plan<Tin> plan = _fftw_plan_many_dft<Tin,Tout>((int)x_mesh.rank(), 
							n, 
							1,
							f.data(),
							(int*)NULL,
							1,
							idist,
							reinterpret_cast<typename _fftw_type<Tout>::type*>(Ff.data()),
							(int*)NULL,
							1,
							odist,
							FFTW_ESTIMATE);
		execIfFloat(Tin,fftwf_execute(plan),fftw_execute(plan))
		execIfFloat(Tin,fftwf_destroy_plan(plan),fftw_destroy_plan(plan))
		std::cout << "performed with fftw";
	}
	# endif
	# if fftw_omp_AVAIL 
	else if constexpr(std::is_same<ExecSpace, Kokkos::OpenMP>::value) {
		execIfFloat(Tin,fftwf_init_threads(),fftw_init_threads())
		execIfFloat(Tin,fftwf_plan_with_nthreads(ExecSpace::concurrency()),fftw_plan_with_nthreads(ExecSpace::concurrency()))
		fftw_plan_with_nthreads(ExecSpace::concurrency());
		_fftw_plan<Tin> plan = _fftw_plan_many_dft<Tin,Tout>((int)x_mesh.rank(), 
							n, 
							1,
							f.data(),
							(int*)NULL,
							1,
							idist,
							reinterpret_cast<typename _fftw_type<Tout>::type*>(Ff.data()),
							(int*)NULL,
							1,
							odist,
							FFTW_ESTIMATE);
		execIfFloat(Tin,fftwf_execute(plan),fftw_execute(plan))
		execIfFloat(Tin,fftwf_destroy_plan(plan),fftw_destroy_plan(plan))
		std::cout << "performed with fftw_omp";
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
							 cufft_transform_type<Tin,Tout>(),
							 1); 

		if(cufft_rt != CUFFT_SUCCESS)
			throw std::runtime_error("cufftPlan1d failed");

		cufft_rt = _cufftExec<Tin,Tout>(plan, f.data(), reinterpret_cast<typename _cufft_type<Tout>::type*>(Ff.data()));
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

