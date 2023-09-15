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

static constexpr std::size_t s_degree_x = DEGREE_X;

# if 0
#if defined(BSPLINES_TYPE_UNIFORM)
using BSplinesX = UniformBSplines<DimX, s_degree_x>;
using IDimY = ddc::UniformPointSampling<DimY>;
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
using BSplinesX = NonUniformBSplines<DimX, s_degree_x>;
using IDimY = ddc::NonUniformPointSampling<DimY>;
#endif
# else
template <typename BSpX>
using GrevillePoints
        = GrevilleInterpolationPoints<BSpX, BoundCond::PERIODIC, BoundCond::PERIODIC>;

#if defined(BSPLINES_TYPE_UNIFORM)
template <typename X>
using BSplines = UniformBSplines<X, s_degree_x>;

template <typename X>
using IDim = std::conditional_t<std::is_same_v<X,DimX>, typename GrevillePoints<BSplines<X>>::interpolation_mesh_type, ddc::UniformPointSampling<X>>;

#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
using BSplines = NonUniformBSplines<X, s_degree_x>;

template <typename X>
using IDim = std::conditional_t<std::is_same_v<X,DimX>, typename GrevillePoints<BSplines<X>>::interpolation_mesh_type, ddc::NonUniformPointSampling<X>>;
#endif
# endif

template <typename X>
using evaluator_type = CosineEvaluator::Evaluator<IDim<X>>;

template <typename... IDimX>
using Index = ddc::DiscreteElement<IDimX...>;
template <typename IDimX>
using DVect = ddc::DiscreteVector<IDimX>;
template <typename X>
using Coord = ddc::Coordinate<X>;

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename X, typename Y>
static void BatchedSplineBuilderTest()
{
    Coord<X> constexpr x0(0.);
    Coord<X> constexpr xN(1.);

    Coord<Y> constexpr y0(0.);
    Coord<Y> constexpr yN(1.);
    
	std::size_t constexpr ncells = 10; // TODO : restore 10
	std::size_t constexpr nbatch = 10; // TODO : restore 10
  	// std::size_t constexpr nbatch = 65535; // TODO : handle bigger matrices but create chunks of it
    // 1. Create BSplines
#if defined(BSPLINES_TYPE_UNIFORM)
        ddc::init_discrete_space<BSplines<X>>(x0, xN, ncells);
        ddc::init_discrete_space(IDim<Y>::init(y0, yN, DVect<IDim<Y>>(nbatch)));
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
        std::vector<Coord<X>> breaks_x(ncells+1);
        std::vector<Coord<Y>> breaks_y(nbatch+1);
        double dx = (xN - x0) / ncells;
        double dy = (yN - y0) / nbatch;
        for (int i(0); i < ncells+1; ++i) {
            breaks_x[i] = Coord<X>(x0 + i * dx);
        }
        for (int i(0); i < nbatch+1; ++i) {
            breaks_y[i] = Coord<Y>(y0 + i * dy);
        }
        ddc::init_discrete_space<BSplines<X>>(breaks_x);
        ddc::init_discrete_space<IDim<Y>>(breaks_y);
#endif
    ddc::DiscreteDomain<BSplines<X>> const& dom_bsplines_x(
            ddc::discrete_space<BSplines<X>>().full_domain());
    ddc::DiscreteDomain<IDim<Y>> const dom_y(Index<IDim<Y>>(0), DVect<IDim<Y>>(nbatch));
	ddc::DiscreteDomain<BSplines<X>, IDim<Y>> const dom_coef(dom_bsplines_x, dom_y);

    // 2. Create a Spline represented by a chunk over BSplines
    // The chunk is filled with garbage data, we need to initialize it
    ddc::Chunk coef_alloc(dom_coef, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan coef = coef_alloc.span_view();
	// Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> coef_kv("coef_kv", dom_bsplines_x.size(), nbatch);
	// ddc::ChunkSpan<double, ddc::DiscreteDomain<BSplines<X>,IDim<Y>>, std::experimental::layout_right, ddc::KokkosAllocator<double, MemorySpace>> coef(coef_kv, dom_coef);

    // 3. Create the interpolation domain
    ddc::init_discrete_space<IDim<X>>(GrevillePoints<BSplines<X>>::get_sampling());
    ddc::DiscreteDomain<IDim<X>> interpolation_domain(GrevillePoints<BSplines<X>>::get_domain());
	ddc::DiscreteDomain<IDim<X>, IDim<Y>> const dom_vals(interpolation_domain, dom_y);

    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    SplineBuilderBatched<SplineBuilder<ExecSpace, BSplines<X>, IDim<X>, BoundCond::PERIODIC, BoundCond::PERIODIC>, MemorySpace, IDim<Y>> spline_builder(
           interpolation_domain, dom_y);

    // 5. Allocate and fill a chunk over the interpolation domain
    ddc::Chunk vals1_cpu_alloc(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan vals1_cpu = vals1_cpu_alloc.span_view();
    evaluator_type<X> evaluator(interpolation_domain);
    evaluator(vals1_cpu);
    ddc::Chunk vals1_alloc(interpolation_domain, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan vals1 = vals1_alloc.span_view();
	ddc::deepcopy(vals1, vals1_cpu);

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
	ddc::ChunkSpan vals = vals_alloc.span_view();
	ddc::for_each(
			  ddc::policies::policy<ExecSpace>(),
              vals.domain(),
              DDC_LAMBDA (Index<IDim<X>,IDim<Y>> const e) { 
				  vals(e) = vals1(ddc::select<IDim<X>>(e));
              });

	// 5.5 Permute Layout TODO : encapsulate
	Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vals_kv(vals.data_handle(), ncells, nbatch);
	Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> vals_tr_kv("vals_tr_kv", ncells, nbatch);
	Kokkos::deep_copy(vals_tr_kv, vals_kv);
ddc::ChunkSpan<double, ddc::DiscreteDomain<IDim<X>,IDim<Y>>, std::experimental::layout_right, MemorySpace> vals_tr(vals_tr_kv, vals.domain());

    // 6. Finally build the spline by filling `coef`
    spline_builder(coef, vals); // TODO : clarify the suffixes _tr

	// Temporary deep_copy TODO : remove & eval on GPU
    ddc::Chunk coef_cpu_alloc(dom_coef, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan coef_cpu = coef_cpu_alloc.span_view();
	ddc::deepcopy(coef_cpu, coef);
	// auto coef_allockv = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), coef_kv);
	// ddc::ChunkSpan<double, ddc::DiscreteDomain<BSplines<X>,IDim<Y>>, std::experimental::layout_right, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>> coef(coef_allockv, dom_coef);
	
	auto vals_cpu_kv = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), vals_tr_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDim<X>,IDim<Y>>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space> vals_cpu(vals_cpu_kv, dom_vals);

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    SplineEvaluator<BSplines<X>> spline_evaluator(g_null_boundary<BSplines<X>>, g_null_boundary<BSplines<X>>);

	ddc::Chunk coords_eval_alloc(interpolation_domain, ddc::KokkosAllocator<Coord<X>, Kokkos::HostSpace>());
	ddc::ChunkSpan coords_eval = coords_eval_alloc.span_view();
	ddc::for_each(
			ddc::policies::policy<Kokkos::DefaultHostExecutionSpace>(),
            coords_eval.domain(),
            DDC_LAMBDA(Index<IDim<X>> const e) {
        coords_eval(e) = ddc::coordinate(e);
    });

    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan spline_eval = spline_eval_alloc.span_view();
	ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan spline_eval_deriv = spline_eval_deriv_alloc.span_view();


	# if 0 
	// TODO: encapsulate in ddc function
	Kokkos::View<double**, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> spline_eval_kv(spline_eval.data_handle(), ncells, nbatch);
	Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> spline_eval_tr_kv("spline_eval_tr_kv", ncells, nbatch);
	Kokkos::deep_copy(spline_eval_tr_kv, spline_eval_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDim<X>,IDim<Y>>, std::experimental::layout_right, ExecSpace> spline_eval_tr(spline_eval_tr_kv, spline_eval.domain());
	# endif
	// TODO: encapsulate in ddc function
	# if 1
	// Kokkos::View<double**, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> coef_allockv(coef.data_handle(), dom_bsplines_x.extent<BSplines<X>>(), nbatch);
	// Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> coef2_kv("coef2_kv", dom_bsplines_x.size(), nbatch);
	// Kokkos::deep_copy(coef2_kv, coef_kv);
	// ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplines<X>,IDim<Y>>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space> coef2(coef2_kv, coef.domain());
	# endif
	ddc::for_each(
			ddc::policies::policy<Kokkos::DefaultHostExecutionSpace>(),
            dom_y,
            DDC_LAMBDA (ddc::DiscreteElement<IDim<Y>> const iy) {
	    spline_evaluator(
			spline_eval[iy],
			ddc::ChunkSpan<const Coord<X>, ddc::DiscreteDomain<IDim<X>>, std::experimental::layout_right, Kokkos::HostSpace>(coords_eval),
			coef_cpu[iy].span_cview()
	);
		spline_evaluator
            .deriv(
			spline_eval_deriv[iy],
			ddc::ChunkSpan<const Coord<X>, ddc::DiscreteDomain<IDim<X>>, std::experimental::layout_right, Kokkos::HostSpace>(coords_eval),
			coef_cpu[iy].span_cview()
	);
});
   # if 0
   for (int i=0; i<ncells; i++) {
      for (int j=0; j<nbatch; j++) {
      	std::cout << spline_eval(ddc::DiscreteElement<IDim<X>>(i),ddc::DiscreteElement<IDim<Y>>(j)) << " - ";
		}
      std::cout << "\n";
	}
	# endif
	// 8. Checking errors
    double max_norm_error = ddc::transform_reduce(
			ddc::policies::policy<Kokkos::DefaultHostExecutionSpace>(),
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
            DDC_LAMBDA (Index<IDim<X>,IDim<Y>> const e) {
        return Kokkos::abs(spline_eval(e) - vals_cpu(e));
	});

	double max_norm_error_diff = ddc::transform_reduce(
			ddc::policies::policy<Kokkos::DefaultHostExecutionSpace>(),
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
           	DDC_LAMBDA (Index<IDim<X>,IDim<Y>> const e) {
        	Coord<X> const x = ddc::coordinate(ddc::select<IDim<X>>(e));
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

    SplineErrorBounds<evaluator_type<X>> error_bounds(evaluator);
    const double h = (xN - x0) / ncells;
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(h, s_degree_x), 1.0e-14 * max_norm));
    EXPECT_LE(
            max_norm_error_diff,
            std::max(error_bounds.error_bound_on_deriv(h, s_degree_x), 1e-12 * max_norm_diff));
	#if 0
    EXPECT_LE(
            max_norm_error_integ,
            std::max(error_bounds.error_bound_on_int(h, s_degree_x), 1.0e-14 * max_norm_int));
	# endif
}

TEST(BatchedSplineBuilderHost, Identity)
{
	BatchedSplineBuilderTest<Kokkos::DefaultHostExecutionSpace,Kokkos::DefaultHostExecutionSpace::memory_space,DimX,DimY>();
}

TEST(BatchedSplineBuilderDevice, Identity)
{
	BatchedSplineBuilderTest<Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space,DimX,DimY>();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::ddc::ScopeGuard scope(argc, argv);
    return RUN_ALL_TESTS();
}
