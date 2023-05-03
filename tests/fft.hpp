// SPDX-License-Identifier: MIT

#include <experimental/mdspan>

#include <ddc/ddc.hpp>
#include <kernels/fft.hpp>

template <typename X>
using DDim = ddc::UniformPointSampling<X>;

template <typename ...DDim>
using DElem = ddc::DiscreteElement<DDim...>;

template <typename ...DDim>
using DVect = ddc::DiscreteVector<DDim...>;

template <typename ...DDim>
using DDom = ddc::DiscreteDomain<DDim...>;

template <typename Kx>
using DFDim = ddc::PeriodicSampling<Kx>;

// is_complex : trait to determine if type is std::complex<something>
template<typename T>
  struct is_complex : std::false_type {};

template<typename T>
  struct is_complex<std::complex<T>> : std::true_type {};

// LastSelector: returns a if Dim==Last, else b
template <typename T, typename Dim, typename Last>
constexpr T LastSelector(const T a, const T b) {
	return std::is_same<Dim,Last>::value ? a : b;
}

template <typename T, typename Dim, typename First, typename Second, typename... Tail>
constexpr T LastSelector(const T a, const T b) {
	return LastSelector<T,Dim,Second,Tail...>(a, b);
}

#if fftw_omp_AVAIL
template <typename MemorySpace, typename T>
using Allocator = typename std::conditional<std::is_same_v<MemorySpace,Kokkos::Serial::memory_space> || std::is_same_v<MemorySpace,Kokkos::OpenMP::memory_space>, ddc::HostAllocator<T>, ddc::DeviceAllocator<T>>::type;

template <typename ExecSpace>
constexpr auto policy = []{ if constexpr (std::is_same<ExecSpace,Kokkos::Serial>::value) { return ddc::policies::serial_host; } else if constexpr (std::is_same<ExecSpace,Kokkos::OpenMP>::value) { return ddc::policies::parallel_host; } else { return ddc::policies::parallel_device; } };
#else
template <typename MemorySpace, typename T>
using Allocator = typename std::conditional<std::is_same_v<MemorySpace,Kokkos::Serial::memory_space>, ddc::HostAllocator<T>, ddc::DeviceAllocator<T>>::type;

template <typename ExecSpace>
constexpr auto policy = []{ if constexpr (std::is_same<ExecSpace,Kokkos::Serial>::value) { return ddc::policies::serial_host; } else { return ddc::policies::parallel_device; } };
#endif

// TODO:
// - FFT multidim but according to a subset of dimensions
template <typename ExecSpace, typename MemorySpace, typename Tin, typename Tout, typename... X>
static void TestFFT()
{
	const double a		= -5;
	const double b		= 5;
    const std::size_t Nx = 16; // Optimal value is (b-a)^2/(2*pi)

	DDom<DDim<X>...> const x_mesh = DDom<DDim<X>...>(
		ddc::init_discrete_space(DDim<X>::init(ddc::detail::TaggedVector<ddc::CoordinateElement, X>(a+(b-a)/Nx/2), ddc::detail::TaggedVector<ddc::CoordinateElement, X>(b-(b-a)/Nx/2), DVect<DDim<X>>(Nx)))...
	);
	ddc::Chunk _f = ddc::Chunk(x_mesh, Allocator<MemorySpace, Tin>());
	ddc::ChunkSpan f = _f.span_view();
	ddc::for_each(
		policy<ExecSpace>(),
		ddc::get_domain<DDim<X>...>(f),
		DDC_LAMBDA(DElem<DDim<X>...> const e) {
			// f(e) = (cos(4*coordinate(ddc::select<DDim<X>>(e)))*...);
			// f(e) = ((sin(coordinate(ddc::select<DDim<X>>(e))+1e-20)/(coordinate(ddc::select<DDim<X>>(e))+1e-20))*...);
			f(e) = static_cast<Tin>(exp(-(pow(coordinate(ddc::select<DDim<X>>(e)),2) + ...)/2));
		}
	);

	DDom<DFDim<K<X>>...> const k_mesh = DDom<DFDim<K<X>>...>(
		ddc::init_discrete_space(DFDim<K<X>>::init(ddc::detail::TaggedVector<ddc::CoordinateElement, K<X>>(0), ddc::detail::TaggedVector<ddc::CoordinateElement, K<X>>(is_complex<Tin>::value&&is_complex<Tout>::value ? 2*(Nx-1)/(b-a)*M_PI : LastSelector<double,X,X...>(Nx/(b-a)*M_PI,2*(Nx-1)/(b-a)*M_PI)), ddc::DiscreteVector<DFDim<K<X>>>(is_complex<Tin>::value&&is_complex<Tout>::value ? Nx : LastSelector<double,X,X...>(Nx/2+1,Nx)), ddc::DiscreteVector<DFDim<K<X>>>(Nx)))...
	);
	ddc::Chunk _Ff = ddc::Chunk(k_mesh, Allocator<MemorySpace,Tout>());
	ddc::ChunkSpan Ff = _Ff.span_view();
	FFT(ExecSpace(), Ff, f);
	Kokkos::fence();

	// deepcopy of Ff because FFT C2R overwrites the input
	ddc::Chunk _Ff_bis = ddc::Chunk(ddc::get_domain<DFDim<K<X>>...>(Ff), Allocator<MemorySpace,Tout>());
    ddc::ChunkSpan Ff_bis = _Ff_bis.span_view();
	ddc::deepcopy(Ff_bis, Ff);

	ddc::Chunk _FFf = ddc::Chunk(x_mesh, Allocator<MemorySpace,Tin>());
	ddc::ChunkSpan FFf = _FFf.span_view();
	FFT(ExecSpace(), FFf, Ff_bis);

	ddc::Chunk _f_host = ddc::Chunk(ddc::get_domain<DDim<X>...>(f), ddc::HostAllocator<Tin>());
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

	ddc::Chunk _Ff_host = ddc::Chunk(ddc::get_domain<DFDim<K<X>>...>(Ff), ddc::HostAllocator<Tout>());
    ddc::ChunkSpan Ff_host = _Ff_host.span_view();
	ddc::deepcopy(Ff_host, Ff);
	# if 0
	std::cout << "\n output:\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DFDim<K<X>>...>(Ff_host),
        [=](DElem<DFDim<K<X>>...> const e) {
			(std::cout << ... << coordinate(ddc::select<DFDim<K<X>>>(e))) << "->" << abs(Ff_host(e))*pow((b-a)/Nx/sqrt(2*M_PI),sizeof...(X)) << " " << exp(-(pow(coordinate(ddc::select<DFDim<K<X>>>(e)),2) + ...)/2) << ", ";
	});
	# endif

	ddc::Chunk _FFf_host = ddc::Chunk(ddc::get_domain<DDim<X>...>(FFf), ddc::HostAllocator<Tin>());
    ddc::ChunkSpan FFf_host = _FFf_host.span_view();
	ddc::deepcopy(FFf_host, FFf);
	# if 0
	std::cout << "\n iFFT(FFT):\n";
	ddc::for_each(
        ddc::policies::serial_host,
        ddc::get_domain<DDim<X>...>(FFf_host),
        [=](DElem<DDim<X>...> const e) {
			(std::cout << ... << coordinate(ddc::select<DDim<X>>(e))) << "->" << abs(FFf_host(e))*pow(1./Nx,sizeof...(X)) << " " << abs(f_host(e)) << ", ";
	});
	# endif

	double criterion = sqrt(ddc::transform_reduce(
		ddc::get_domain<DFDim<K<X>>...>(Ff_host),
		0.,
		ddc::reducer::sum<double>(),
		[=](DElem<DFDim<K<X>>...> const e) {
			return pow(abs(Ff_host(e))*pow((b-a)/Nx/sqrt(2*M_PI),sizeof...(X))-exp(-(pow(coordinate(ddc::select<DFDim<K<X>>>(e)),2) + ...)/2),2)/(LastSelector<std::size_t,X,X...>(Nx/2,Nx)*...);
	}));

	double criterion2 = sqrt(ddc::transform_reduce(
		ddc::get_domain<DDim<X>...>(FFf_host),
		0.,
		ddc::reducer::sum<double>(),
		[=](DElem<DDim<X>...> const e) {
			return pow(abs(FFf_host(e))*pow(1./Nx,sizeof...(X))-abs(f_host(e)),2)/pow(Nx,sizeof...(X));
	}));

	std::cout << "\n Distance between analytical prediction and numerical result : " << criterion;
	std::cout << "\n Distance between input and iFFT(FFT(input)) : " << criterion2;
	ASSERT_LE(criterion, 2e-6);
	ASSERT_LE(criterion2, 5e-8);
}

