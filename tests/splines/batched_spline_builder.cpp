#include <algorithm>
#include <array>
#include <cmath>
#include <iosfwd>
#include <vector>

#include <experimental/mdspan>

#include <ddc/ddc.hpp>

#include <ddc/kernels/splines/bsplines_non_uniform.hpp>
#include <ddc/kernels/splines/bsplines_uniform.hpp>
#include <ddc/kernels/splines/greville_interpolation_points.hpp>
#include <ddc/kernels/splines/null_boundary_value.hpp>
#include <ddc/kernels/splines/spline_boundary_conditions.hpp>
#include <ddc/kernels/splines/spline_builder.hpp>
#include <ddc/kernels/splines/spline_builder_batched.hpp>
#include <ddc/kernels/splines/spline_evaluator.hpp>
#include <ddc/kernels/splines/view.hpp>

#include <gtest/gtest.h>
#include "ddc/coordinate.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/for_each.hpp"
#include "ddc/uniform_point_sampling.hpp"

#include "Kokkos_Core_fwd.hpp"
#include "cosine_evaluator.hpp"
#include "polynomial_evaluator.hpp"
#include "spline_error_bounds.hpp"

struct DimX
{
    static constexpr bool PERIODIC = true;
};

struct DimY {
    static constexpr bool PERIODIC = true; // TODO : remove
};

struct DimZ {
    static constexpr bool PERIODIC = true; // TODO : remove
};

static constexpr std::size_t s_degree_x = DEGREE_X;

template <typename BSpX>
using GrevillePoints
        = GrevilleInterpolationPoints<BSpX, BoundCond::PERIODIC, BoundCond::PERIODIC>;

#if defined(BSPLINES_TYPE_UNIFORM)
template <typename X>
using BSplines = UniformBSplines<X, s_degree_x>;

template <typename X, typename I>
// using IDim = std::conditional_t<std::is_same_v<X,DimX>, typename GrevillePoints<BSplines<DimX>>::interpolation_mesh_type, ddc::UniformPointSampling<X>>; // TODO : Remove explicit DimX
using IDim = std::conditional_t<std::is_same_v<X,I>, typename GrevillePoints<BSplines<X>>::interpolation_mesh_type, ddc::UniformPointSampling<X>>; // TODO : Remove explicit DimX

#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
using BSplines = NonUniformBSplines<X, s_degree_x>;

template <typename X, typename I>
using IDim = std::conditional_t<std::is_same_v<X,I>, typename GrevillePoints<BSplines<X>>::interpolation_mesh_type, ddc::NonUniformPointSampling<X>>;
# endif
template <typename IDimX>
using evaluator_type = CosineEvaluator::Evaluator<IDimX>;

template <typename... IDimX>
using Index = ddc::DiscreteElement<IDimX...>;
template <typename... IDimX>
using DVect = ddc::DiscreteVector<IDimX...>;
template <typename X>
using Coord = ddc::Coordinate<X>;

template <typename I, typename... X>
using BatchDims = ddc::type_seq_remove_t<ddc::detail::TypeSeq<X...>,ddc::detail::TypeSeq<I>>;

template<typename X>
static constexpr Coord<X> x0() {
  return Coord<X>(0.);
}

template<typename X>
static constexpr Coord<X> xN() {
  return Coord<X>(1.);
}

template<typename X>
static constexpr double dx(double ncells) {
  return (xN<X>()-x0<X>())/ncells;
}

template<typename X>
static constexpr std::vector<Coord<X>> breaks(double ncells) {
  std::vector<Coord<X>> out(ncells+1);
  for (int i(0); i < ncells+1; ++i) {
     out[i] = x0<X>() + i * dx<X>(ncells);
  }
  return out;
}

template <class IDimI, class T>
struct DimsInitializer;

template <class IDimI, class... IDimX> // TODO: rename X with IDimX
struct DimsInitializer<IDimI, ddc::detail::TypeSeq<IDimX...>>
{
  void operator()(std::size_t const ncells) {
  #if defined(BSPLINES_TYPE_UNIFORM)
        (ddc::init_discrete_space(IDimX::init(x0<typename IDimX::continuous_dimension_type>(), xN<typename IDimX::continuous_dimension_type>(), DVect<IDimX>(ncells))),...);
        ddc::init_discrete_space<BSplines<typename IDimI::continuous_dimension_type>>(x0<typename IDimI::continuous_dimension_type>(), xN<typename IDimI::continuous_dimension_type>(), ncells);
  #elif defined(BSPLINES_TYPE_NON_UNIFORM)
        (ddc::init_discrete_space<IDimX>(breaks<typename IDimX::continuous_dimension_type>(ncells)), ...);
        ddc::init_discrete_space<BSplines<typename IDimI::continuous_dimension_type>>(breaks<typename IDimI::continuous_dimension_type>(ncells));
  #endif
    ddc::init_discrete_space<IDimI>(GrevillePoints<BSplines<typename IDimI::continuous_dimension_type>>::get_sampling());
  }
};

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename I, typename... X>
static void BatchedSplineBuilderTest()
{
	Kokkos::DefaultHostExecutionSpace host_exec_space = Kokkos::DefaultHostExecutionSpace();
	ExecSpace exec_space = ExecSpace();
	std::size_t constexpr ncells = 10; // TODO : restore 10
    // 1. Create BSplines
	DimsInitializer<IDim<I,I>,BatchDims<IDim<I,I>,IDim<X,I>...>> dims_initializer;
	dims_initializer(ncells);
	// auto const dom_coef = ddc::detail::convert_type_seq_to_discrete_domain<ddc::type_seq_replace_t<ddc::detail::TypeSeq<IDim<X,I>...>,ddc::detail::TypeSeq<IDim<I,I>>,ddc::detail::TypeSeq<BSplines<I>>>>((std::is_same_v<X,I> ? ddc::discrete_space<BSplines<X>>().full_domain() : ddc::DiscreteDomain<IDim<X,I>>(Index<IDim<X,I>>(0), DVect<IDim<X,I>>(ncells)))...);

    // 2. Create a Spline represented by a chunk over BSplines
    // The chunk is filled with garbage data, we need to initialize it
	// Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> coef_kv("coef_kv", dom_bsplines_x.size(), nbatch);
	// ddc::ChunkSpan<double, ddc::DiscreteDomain<BSplines<X>,IDim<Y>>, std::experimental::layout_right, ddc::KokkosAllocator<double, MemorySpace>> coef(coef_kv, dom_coef);

    // 3. Create the interpolation domain
	ddc::DiscreteDomain<IDim<X,I>...> const dom_vals = ddc::DiscreteDomain<IDim<X,I>...>((std::is_same_v<X,I> ? GrevillePoints<BSplines<X>>::get_domain() : ddc::DiscreteDomain<IDim<X,I>>(Index<IDim<X,I>>(0), DVect<IDim<X,I>>(ncells)))...);


    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    SplineBuilderBatched<SplineBuilder<ExecSpace, BSplines<I>, IDim<I,I>, BoundCond::PERIODIC, BoundCond::PERIODIC>, MemorySpace, IDim<X,I>...> spline_builder(dom_vals);

    ddc::DiscreteDomain<IDim<I,I>> const interpolation_domain = spline_builder.interpolation_domain();
	auto const dom_y = spline_builder.batch_domain();
    ddc::DiscreteDomain<BSplines<I>> const dom_bsplines_x = spline_builder.bsplines_domain();
	auto const dom_coef = spline_builder.spline_domain();

    ddc::Chunk coef_alloc(dom_coef, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan coef = coef_alloc.span_view();

    // 5. Allocate and fill a chunk over the interpolation domain
    ddc::Chunk vals1_cpu_alloc(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan vals1_cpu = vals1_cpu_alloc.span_view();
    evaluator_type<IDim<I,I>> evaluator(interpolation_domain);
    evaluator(vals1_cpu);
    ddc::Chunk vals1_alloc(interpolation_domain, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan vals1 = vals1_alloc.span_view();
	ddc::deepcopy(vals1, vals1_cpu);

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
	ddc::ChunkSpan vals = vals_alloc.span_view();
	ddc::for_each(
			  ddc::policies::policy(exec_space),
              vals.domain(),
              DDC_LAMBDA (Index<IDim<X,I>...> const e) { 
				  vals(e) = vals1(ddc::select<IDim<I,I>>(e));
              });

    auto vals_ptr = vals.data_handle();
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,1),KOKKOS_LAMBDA (int j) { printf("%f ", vals_ptr[1]); });

	// 5.5 Permute Layout TODO : encapsulate
	Kokkos::View<ddc::detail::mdspan_to_kokkos_element_t<double, sizeof...(X)>, Kokkos::LayoutRight, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vals_kv(vals.data_handle(), ddc::select<IDim<X,I>>(dom_vals).extents()...);
	Kokkos::View<ddc::detail::mdspan_to_kokkos_element_t<double, sizeof...(X)>, Kokkos::LayoutRight, ExecSpace> vals_tr_kv("vals_tr_kv", ddc::select<IDim<X,I>>(dom_vals.extents())...);
	Kokkos::deep_copy(vals_tr_kv, vals_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDim<X,I>...>, std::experimental::layout_right, MemorySpace> vals_tr(vals_tr_kv, vals.domain());

    // 6. Finally build the spline by filling `coef`
    spline_builder(coef, vals); // TODO : clarify the suffixes _tr

	// Temporary deep_copy TODO : remove & eval on GPU
    ddc::Chunk coef_cpu_alloc(dom_coef, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan coef_cpu = coef_cpu_alloc.span_view();
	ddc::deepcopy(coef_cpu, coef);
	// auto coef_allockv = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), coef_kv);
	// ddc::ChunkSpan<double, ddc::DiscreteDomain<BSplines<X>,IDim<Y>>, std::experimental::layout_right, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>> coef(coef_allockv, dom_coef);
	
	auto vals_cpu_kv = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), vals_tr_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDim<X,I>...>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space> vals_cpu(vals_cpu_kv, dom_vals);

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    SplineEvaluator<BSplines<I>> spline_evaluator(g_null_boundary<BSplines<I>>, g_null_boundary<BSplines<I>>);

	ddc::Chunk coords_eval_alloc(interpolation_domain, ddc::KokkosAllocator<Coord<I>, Kokkos::HostSpace>());
	ddc::ChunkSpan coords_eval = coords_eval_alloc.span_view();
	ddc::for_each(
			ddc::policies::policy(host_exec_space),
            coords_eval.domain(),
            DDC_LAMBDA(Index<IDim<I,I>> const e) {
        coords_eval(e) = ddc::coordinate(e);
    });

    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan spline_eval = spline_eval_alloc.span_view();
	ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan spline_eval_deriv = spline_eval_deriv_alloc.span_view();


	# if 0 
	// TODO: encapsulate in ddc function
	Kokkos::View<double**, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> spline_eval_kv(spline_eval.data_handle(), ddc::select<IDim<X,I>>(dom_vals).size()...);
	Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> spline_eval_tr_kv("spline_eval_tr_kv", ddc::select<IDim<X,I>>(dom_vals).size()...);
	Kokkos::deep_copy(spline_eval_tr_kv, spline_eval_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDim<X,I>,IDim<Y>>, std::experimental::layout_right, ExecSpace> spline_eval_tr(spline_eval_tr_kv, spline_eval.domain());
	# endif
	// TODO: encapsulate in ddc function
	# if 1
	// Kokkos::View<double**, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> coef_allockv(coef.data_handle(), dom_bsplines_x.extent<BSplines<X>>(), nbatch);
	// Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> coef2_kv("coef2_kv", dom_bsplines_x.size(), nbatch);
	// Kokkos::deep_copy(coef2_kv, coef_kv);
	// ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplines<X>,IDim<Y>>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space> coef2(coef2_kv, coef.domain());
	# endif
	ddc::for_each(
			ddc::policies::policy(host_exec_space),
            dom_y,
            DDC_LAMBDA (typename decltype(dom_y)::discrete_element_type const iy) {
	    spline_evaluator(
			spline_eval[iy],
			ddc::ChunkSpan<const Coord<I>, ddc::DiscreteDomain<IDim<I,I>>, std::experimental::layout_right, Kokkos::HostSpace>(coords_eval),
			coef_cpu[iy].span_cview()
	);
		spline_evaluator
            .deriv(
			spline_eval_deriv[iy],
			ddc::ChunkSpan<const Coord<I>, ddc::DiscreteDomain<IDim<I,I>>, std::experimental::layout_right, Kokkos::HostSpace>(coords_eval),
			coef_cpu[iy].span_cview()
	);
});
   # if 0
   for (int i=0; i<ncells; i++) {
      for (int j=0; j<ncells; j++) {
      	std::cout << spline_eval(ddc::DiscreteElement<IDim<DimX,I>>(i),ddc::DiscreteElement<IDim<DimY,I>>(j)) << " - ";
		}
      std::cout << "\n";
	}
	# endif
	// 8. Checking errors
    double max_norm_error = ddc::transform_reduce(
			ddc::policies::policy(host_exec_space),
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
            DDC_LAMBDA (Index<IDim<X,I>...> const e) {
        return Kokkos::abs(spline_eval(e) - vals_cpu(e));
	});

	double max_norm_error_diff = ddc::transform_reduce(
			ddc::policies::policy(host_exec_space),
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
           	DDC_LAMBDA (Index<IDim<X,I>...> const e) {
        	Coord<I> const x = ddc::coordinate(ddc::select<IDim<I,I>>(e));
        return Kokkos::abs(spline_eval_deriv(e) - evaluator.deriv(x,1));
	});
	#if 0
    double const max_norm_error_integ = std::fabs(
            spline_evaluator.integrate(coef_cpu.span_cview()) - evaluator.deriv(xN, -1)
            + evaluator.deriv(x0, -1));
	# endif

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    // double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type<IDim<I,I>>> error_bounds(evaluator);
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(dx<I>(ncells), s_degree_x), 1.0e-14 * max_norm));
    EXPECT_LE(
            max_norm_error_diff,
            std::max(error_bounds.error_bound_on_deriv(dx<I>(ncells), s_degree_x), 1e-12 * max_norm_diff));
	#if 0
    EXPECT_LE(
            max_norm_error_integ,
            std::max(error_bounds.error_bound_on_int(h, s_degree_x), 1.0e-14 * max_norm_int));
	#endif
}

TEST(BatchedSplineBuilderHost, 2DX)
{
	BatchedSplineBuilderTest<Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultHostExecutionSpace::memory_space,DimX,DimX,DimY>();
}

TEST(BatchedSplineBuilderHost, 2DY)
{
	BatchedSplineBuilderTest<Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultHostExecutionSpace::memory_space,DimY,DimX,DimY>();
}

TEST(BatchedSplineBuilderDevice, 2DX)
{
	BatchedSplineBuilderTest<Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space,DimX,DimX,DimY>();
}

TEST(BatchedSplineBuilderDevice, 2DY)
{
	BatchedSplineBuilderTest<Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space,DimY,DimX,DimY>();
}

TEST(BatchedSplineBuilderHost, 3DX)
{
	BatchedSplineBuilderTest<Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultHostExecutionSpace::memory_space,DimX,DimX,DimY,DimZ>();
}

TEST(BatchedSplineBuilderHost, 3DY)
{
	BatchedSplineBuilderTest<Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultHostExecutionSpace::memory_space,DimY,DimX,DimY,DimZ>();
}

TEST(BatchedSplineBuilderHost, 3DZ)
{
	BatchedSplineBuilderTest<Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultHostExecutionSpace::memory_space,DimZ,DimX,DimY,DimZ>();
}

TEST(BatchedSplineBuilderDevice, 3DX)
{
	BatchedSplineBuilderTest<Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space,DimX,DimX,DimY,DimZ>();
}

TEST(BatchedSplineBuilderDevice, 3DY)
{
	BatchedSplineBuilderTest<Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space,DimY,DimX,DimY,DimZ>();
}

TEST(BatchedSplineBuilderDevice, 3DZ)
{
	BatchedSplineBuilderTest<Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space,DimZ,DimX,DimY,DimZ>();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::ddc::ScopeGuard scope(argc, argv);
    return RUN_ALL_TESTS();
}
