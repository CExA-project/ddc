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

template <typename Dims>
struct K;

// LastSelector: returns a if Dim==Last, else b
template <typename Dim, typename Last>
constexpr int LastSelector(const int a, const int b) {
	return std::is_same<Dim,Last>::value ? a : b;
}

template <typename Dim, typename First, typename Second, typename... Tail>
constexpr int LastSelector(const int a, const int b) {
	return LastSelector<Dim,Second,Tail...>(a, b);
}

template <typename MemorySpace, typename T>
using Allocator = typename std::conditional<std::is_same<MemorySpace,Kokkos::Serial::memory_space>::value, ddc::HostAllocator<T>, ddc::DeviceAllocator<T>>::type;

template <typename ExecSpace>
constexpr auto policy = []{ if constexpr (std::is_same<ExecSpace,Kokkos::Serial>::value) { return ddc::policies::serial_host; } else { return ddc::policies::parallel_device; } };

// TODO:
// - cuFFT+FFTW
// - FFT multidim but according to a subset of dimensions
template <typename ExecSpace, typename MemorySpace, typename T, typename... X>
static void TestFFT()
{
	const T a		= -5;
	const T b		= 5;
    const int Nx        = 16; // Optimal value is (b-a)^2/(2*pi)~25, we retain 32 which is the next power of 2

	DDom<DDim<X>...> const x_mesh = DDom<DDim<X>...>(
		ddc::init_discrete_space(DDim<X>::init(ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, X>(a+(b-a)/Nx/2), ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, X>(b-(b-a)/Nx/2), DVect<DDim<X>>(Nx)))...
	);
	ddc::Chunk _f = ddc::Chunk(x_mesh, Allocator<MemorySpace, T>());
	ddc::ChunkSpan f = _f.span_view();
	ddc::for_each(
		policy<ExecSpace>(),
		ddc::get_domain<DDim<X>...>(f),
		DDC_LAMBDA(DElem<DDim<X>...> const e) {
			// f(e) = (cos(4*coordinate(ddc::select<DDim<X>>(e)))*...);
			// f(e) = ((sin(coordinate(ddc::select<DDim<X>>(e))+1e-20)/(coordinate(ddc::select<DDim<X>>(e))+1e-20))*...);
			f(e) = exp(-(pow(coordinate(ddc::select<DDim<X>>(e)),2) + ...)/2);
		}
	);

	DDom<DDim<K<X>>...> const k_mesh = DDom<DDim<K<X>>...>(
		ddc::init_discrete_space(DDim<K<X>>::init(ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, K<X>>(LastSelector<X,X...>(0,-Nx/(b-a)*M_PI)), ddc::ddc_detail::TaggedVector<ddc::CoordinateElement, K<X>>(Nx/(b-a)*M_PI), ddc::DiscreteVector<DDim<K<X>>>(LastSelector<X,X...>(Nx/2+1,Nx))))...
	);
	ddc::Chunk _Ff = ddc::Chunk(k_mesh, Allocator<MemorySpace,std::complex<T>>());
	ddc::ChunkSpan Ff = _Ff.span_view();
	FFT<ExecSpace, MemorySpace, T, X...>(Ff, f);

	ddc::Chunk _f_host = ddc::Chunk(ddc::get_domain<DDim<X>...>(f), ddc::HostAllocator<T>());
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

	ddc::Chunk _Ff_host = ddc::Chunk(ddc::get_domain<DDim<K<X>>...>(Ff), ddc::HostAllocator<std::complex<T>>());
    ddc::ChunkSpan Ff_host = _Ff_host.span_view();
	ddc::deepcopy(Ff_host, Ff);
	# if 1
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
		ddc::reducer::sum<T>(),
		[=](DElem<DDim<K<X>>...> const e) {
			return pow(abs(Ff_host(e))*pow((b-a)/Nx/sqrt(2*M_PI),sizeof...(X))-exp(-(pow(coordinate(ddc::select<DDim<K<X>>>(e)),2) + ...)/2),2)/pow(Nx/2+1,sizeof...(X));
	
	}));
	std::cout << "\n Distance between analytical prediction and numerical result : " << criterion;
	ASSERT_LE(criterion, 1e-6);
}

