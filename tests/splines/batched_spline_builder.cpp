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

struct DimY { };

static constexpr std::size_t s_degree_x = DEGREE_X;

#if defined(BSPLINES_TYPE_UNIFORM)
using BSplinesX = UniformBSplines<DimX, s_degree_x>;
using IDimY = ddc::UniformPointSampling<DimY>;
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
using BSplinesX = NonUniformBSplines<DimX, s_degree_x>;
using IDimY = ddc::NonUniformPointSampling<DimY>;
#endif

using GrevillePoints
        = GrevilleInterpolationPoints<BSplinesX, BoundCond::PERIODIC, BoundCond::PERIODIC>;

using IDimX = GrevillePoints::interpolation_mesh_type;

using evaluator_type = CosineEvaluator::Evaluator<IDimX>;

using IndexX = ddc::DiscreteElement<IDimX>;
using DVectX = ddc::DiscreteVector<IDimX>;
using BsplIndexX = ddc::DiscreteElement<BSplinesX>;
using CoordX = ddc::Coordinate<DimX>;


using IndexY = ddc::DiscreteElement<IDimY>;
using DVectY = ddc::DiscreteVector<IDimY>;
using CoordY = ddc::Coordinate<DimY>;

using IndexXY = ddc::DiscreteElement<IDimX,IDimY>;

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
__host__ void BatchedSplineBuilderTest()
{
    CoordX constexpr x0(0.);
    CoordX constexpr xN(1.);

    CoordY constexpr y0(0.);
    CoordY constexpr yN(1.);
    
	std::size_t constexpr ncells = 10; // TODO : restore 10
    // 1. Create BSplines
#if defined(BSPLINES_TYPE_UNIFORM)
        ddc::init_discrete_space<BSplinesX>(x0, xN, ncells);
        ddc::init_discrete_space(IDimY::init(y0, yN, DVectY(ncells)));
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
        DVectX constexpr npoints(ncells + 1);
        std::vector<CoordX> breaks_x(npoints);
        std::vector<CoordY> breaks_y(npoints);
        double dx = (xN - x0) / ncells;
        double dy = (xN - x0) / ncells;
        for (int i(0); i < npoints; ++i) {
            breaks_x[i] = CoordX(x0 + i * dx);
            breaks_y[i] = CoordY(y0 + i * dy);
        }
        ddc::init_discrete_space<BSplinesX>(breaks_x);
        ddc::init_discrete_space<IDimY>(breaks_y);
#endif
    ddc::DiscreteDomain<BSplinesX> const& dom_bsplines_x(
            ddc::discrete_space<BSplinesX>().full_domain());
    ddc::DiscreteDomain<IDimY> const dom_y(IndexY(0), DVectY(ncells));
	ddc::DiscreteDomain<BSplinesX, IDimY> const dom_coef(dom_bsplines_x, dom_y);

    // 2. Create a Spline represented by a chunk over BSplines
    // The chunk is filled with garbage data, we need to initialize it
    // ddc::Chunk coef_gpu_(dom_coef, ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>());
    // ddc::ChunkSpan coef_gpu = coef_gpu_.span_view();
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> coef_tr_kv("coef_tr_kv", dom_coef.extent<BSplinesX>(), ncells);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<BSplinesX,IDimY>, std::experimental::layout_right, ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>> coef_tr(coef_tr_kv, dom_coef);

    // 3. Create the interpolation domain
    ddc::init_discrete_space<IDimX>(GrevillePoints::get_sampling());
    ddc::DiscreteDomain<IDimX> interpolation_domain(GrevillePoints::get_domain());
	ddc::DiscreteDomain<IDimX, IDimY> const dom_vals(interpolation_domain, dom_y);

    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    SplineBuilderBatched<SplineBuilder<BSplinesX, IDimX, BoundCond::PERIODIC, BoundCond::PERIODIC>, Kokkos::DefaultExecutionSpace::memory_space, IDimY> spline_builder(
           interpolation_domain, dom_y);

    // 5. Allocate and fill a chunk over the interpolation domain
    ddc::Chunk vals1_(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan vals1 = vals1_.span_view();
    evaluator_type evaluator(interpolation_domain);
    evaluator(vals1);
    ddc::Chunk vals1_gpu_(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>());
    ddc::ChunkSpan vals1_gpu = vals1_gpu_.span_view();
	ddc::deepcopy(vals1_gpu, vals1);

    ddc::Chunk vals_(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>());
	ddc::ChunkSpan vals = vals_.span_view();
	ddc::for_each(
			  ddc::policies::parallel_device,
              vals.domain(),
              DDC_LAMBDA (IndexXY const e) { 
				  vals(e) = vals1_gpu(ddc::select<IDimX>(e));
              });

	// 5.5 Permute Layout TODO : encapsulate
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vals_kv(vals.data_handle(), ncells, ncells);
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> vals_tr_kv("vals_tr_kv", ncells, ncells);
	Kokkos::deep_copy(vals_tr_kv, vals_kv);
ddc::ChunkSpan<double, ddc::DiscreteDomain<IDimX,IDimY>, std::experimental::layout_right, ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>> vals_tr(vals_tr_kv, vals.domain());

    // 6. Finally build the spline by filling `coef`
    spline_builder(coef_tr, vals_tr); // TODO : clarify the suffixes _tr

	// Temporary deep_copy TODO : remove & eval on GPU
	auto coef_kv = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), coef_tr_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<BSplinesX,IDimY>, std::experimental::layout_right, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>> coef(coef_kv, dom_coef);
	
	auto vals_cpu_kv = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), vals_tr_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDimX,IDimY>, std::experimental::layout_right, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>> vals_cpu(vals_cpu_kv, dom_vals);

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    SplineEvaluator<BSplinesX> spline_evaluator(g_null_boundary<BSplinesX>, g_null_boundary<BSplinesX>);

	ddc::Chunk coords_eval_(interpolation_domain, ddc::KokkosAllocator<CoordX, Kokkos::DefaultHostExecutionSpace::memory_space>());
	ddc::ChunkSpan coords_eval = coords_eval_.span_view();
	ddc::for_each(
			ddc::policies::parallel_host,
            coords_eval.domain(),
            DDC_LAMBDA(IndexX const e) {
        coords_eval(e) = ddc::coordinate(e);
    });

    ddc::Chunk spline_eval_(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan spline_eval = spline_eval_.span_view();
	ddc::Chunk spline_eval_deriv_(dom_vals, ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan spline_eval_deriv = spline_eval_deriv_.span_view();


	# if 0 
	// TODO: encapsulate in ddc function
	Kokkos::View<double**, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> spline_eval_kv(spline_eval.data_handle(), ncells, ncells);
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace> spline_eval_tr_kv("spline_eval_tr_kv", ncells, ncells);
	Kokkos::deep_copy(spline_eval_tr_kv, spline_eval_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDimX,IDimY>, std::experimental::layout_right, Kokkos::DefaultExecutionSpace> spline_eval_tr(spline_eval_tr_kv, spline_eval.domain());
	# endif
	// TODO: encapsulate in ddc function
	# if 1
	// Kokkos::View<double**, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> coef_kv(coef.data_handle(), dom_bsplines_x.extent<BSplinesX>(), ncells);
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> coef_tr2_kv("coef_tr2_kv", dom_bsplines_x.extent<BSplinesX>(), ncells);
	Kokkos::deep_copy(coef_tr2_kv, coef_tr_kv);
	ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplinesX,IDimY>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space> coef_tr2(coef_tr2_kv, coef.domain());
	# endif
	ddc::for_each(
			ddc::policies::parallel_host,
            dom_y,
            DDC_LAMBDA (ddc::DiscreteElement<IDimY> const iy) {
	    spline_evaluator(
			spline_eval[iy],
			ddc::ChunkSpan<const CoordX, ddc::DiscreteDomain<IDimX>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space>(coords_eval.data_handle(),coords_eval.domain()),
			coef_tr2[iy]
	);
		spline_evaluator
            .deriv(
			spline_eval_deriv[iy],
			ddc::ChunkSpan<const CoordX, ddc::DiscreteDomain<IDimX>, std::experimental::layout_right, Kokkos::DefaultHostExecutionSpace::memory_space>(coords_eval.data_handle(),coords_eval.domain()),
			coef_tr2[iy]
	);
});
   # if 0
   for (int i=0; i<10; i++) {
      for (int j=0; j<10; j++) {
      	std::cout << spline_eval(ddc::DiscreteElement<IDimX>(i),ddc::DiscreteElement<IDimY>(j)) << " - ";
		}
      std::cout << "\n";
	}
	# endif
	// 8. Checking errors
    double max_norm_error = ddc::transform_reduce(
			ddc::policies::parallel_host,
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
            DDC_LAMBDA (IndexXY const e) {
        return Kokkos::abs(spline_eval(e) - vals_cpu(e));
	});

	double max_norm_error_diff = ddc::transform_reduce(
			ddc::policies::parallel_host,
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
           	DDC_LAMBDA (IndexXY const e) {
        	CoordX const x = ddc::coordinate(ddc::select<IDimX>(e));
        return Kokkos::abs(spline_eval_deriv(e) - evaluator.deriv(x,1));
	});
	#if 0
    double const max_norm_error_integ = std::fabs(
            spline_evaluator.integrate(coef.span_cview()) - evaluator.deriv(xN, -1)
            + evaluator.deriv(x0, -1));
	# endif

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    // double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type> error_bounds(evaluator);
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

TEST(BatchedSplineBuilderTest, Identity)
{
	BatchedSplineBuilderTest();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::ddc::ScopeGuard scope(argc, argv);
    return RUN_ALL_TESTS();
}
