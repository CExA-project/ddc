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
#include "ddc/uniform_point_sampling.hpp"

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
using SplineX = ddc::Chunk<double, ddc::DiscreteDomain<BSplinesX>>;
using FieldX = ddc::Chunk<double, ddc::DiscreteDomain<IDimX>>;
using CoordX = ddc::Coordinate<DimX>;


using IndexY = ddc::DiscreteElement<IDimY>;
using DVectY = ddc::DiscreteVector<IDimY>;
using FieldY = ddc::Chunk<double, ddc::DiscreteDomain<IDimY>>;
using CoordY = ddc::Coordinate<DimY>;

using IndexXY = ddc::DiscreteElement<IDimX,IDimY>;
using FieldXY = ddc::Chunk<double, ddc::DiscreteDomain<IDimX,IDimY>>;
using SplineXFieldY = ddc::Chunk<double, ddc::DiscreteDomain<BSplinesX,IDimY>>;
// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
TEST(BatchedSplineBuilderTest, Identity)
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
    SplineXFieldY coef_(dom_coef);
    ddc::ChunkSpan coef = coef_.span_view();

    // 3. Create the interpolation domain
    ddc::init_discrete_space<IDimX>(GrevillePoints::get_sampling());
    ddc::DiscreteDomain<IDimX> interpolation_domain(GrevillePoints::get_domain());
	ddc::DiscreteDomain<IDimX, IDimY> const dom_vals(interpolation_domain, dom_y);

    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    SplineBuilderBatched<SplineBuilder<BSplinesX, IDimX, BoundCond::PERIODIC, BoundCond::PERIODIC>, IDimY> spline_builder(
           interpolation_domain, dom_y);

    // 5. Allocate and fill a chunk over the interpolation domain
    FieldX vals1(interpolation_domain);
    evaluator_type evaluator(interpolation_domain);
    evaluator(vals1);
    FieldXY vals(dom_vals);
	ddc::for_each(
              vals.domain(),
              [&](IndexXY const e) {
				  vals(e) = vals1(ddc::select<IDimX>(e));
              });
    // 6. Finally build the spline by filling `coef`
    spline_builder(coef, vals);
	ddc::for_each(
              coef.domain(),
              [&](ddc::DiscreteElement<BSplinesX, IDimY> const e) {
				  std::cout << coef(e) << " ";
              });
    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    SplineEvaluator<BSplinesX> spline_evaluator(g_null_boundary<BSplinesX>, g_null_boundary<BSplinesX>);

	ddc::Chunk<CoordX, ddc::DiscreteDomain<IDimX>> coords_eval_(interpolation_domain);
	ddc::ChunkSpan coords_eval = coords_eval_.span_view();
	ddc::for_each(
            ddc::policies::serial_host,
            coords_eval.domain(),
            DDC_LAMBDA(IndexX const e) {
        coords_eval(e) = ddc::coordinate(e);
    });

    FieldXY spline_eval_(dom_vals);
    ddc::ChunkSpan spline_eval = spline_eval_.span_view();
	FieldXY spline_eval_deriv_(dom_vals);
    ddc::ChunkSpan spline_eval_deriv = spline_eval_deriv_.span_view();


	# if 1 
	// TODO: encapsulate in ddc function
	Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> spline_eval_kv(spline_eval.data_handle(), ncells, ncells);
	Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::DefaultHostExecutionSpace> spline_eval_tr_kv("spline_eval_tr_kv", ncells, ncells);
	Kokkos::deep_copy(spline_eval_tr_kv, spline_eval_kv);
	ddc::ChunkSpan<double, ddc::DiscreteDomain<IDimX,IDimY>, std::experimental::layout_left, ddc::HostAllocator<double>> spline_eval_tr(spline_eval_tr_kv, spline_eval.domain());
	std::cout << "------------" << "\n";
	for (int i=0; i<ncells*ncells; i++) {
		// std::cout << spline_eval.data_handle()[i] << " " << spline_eval_tr.data_handle()[i] << "\n";
	}
	// TODO: encapsulate in ddc function
	Kokkos::View<double**, Kokkos::DefaultHostExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> coef_kv(coef.data_handle(), dom_bsplines_x.extent<BSplinesX>(), ncells);
	Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::DefaultHostExecutionSpace> coef_tr_kv("coef_tr_kv", dom_bsplines_x.extent<BSplinesX>(), ncells);
	Kokkos::deep_copy(coef_tr_kv, coef_kv);
	ddc::ChunkSpan<const double, ddc::DiscreteDomain<BSplinesX,IDimY>, std::experimental::layout_right, ddc::HostAllocator<double>> coef_tr(coef_tr_kv, coef.domain());
	# endif

	ddc::for_each(
            ddc::policies::serial_host,
            dom_y,
            DDC_LAMBDA(ddc::DiscreteElement<IDimY> const iy) {
		# if 0
		for (auto ix : dom_bsplines_x) {
			std::cout << coef(ix,iy) << " " << coef[iy](ix) << " " << coef_tr[iy](ix)<< "\n";
		}
		# endif
	    spline_evaluator(
			spline_eval[iy],
			ddc::ChunkSpan<const CoordX, ddc::DiscreteDomain<IDimX>, std::experimental::layout_right, Kokkos::HostSpace>(coords_eval.data_handle(),coords_eval.domain()),
			coef_tr[iy]
	);
		spline_evaluator
            .deriv(
			spline_eval_deriv[iy],
			ddc::ChunkSpan<const CoordX, ddc::DiscreteDomain<IDimX>, std::experimental::layout_right, Kokkos::HostSpace>(coords_eval.data_handle(),coords_eval.domain()),
			coef_tr[iy]
	);
});

	std::cout << "---------- TEST ----------\n";
    // 8. Checking errors
    double max_norm_error = ddc::transform_reduce(
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
            [&](IndexXY const e) {
		std::cout << spline_eval(e) << " " << vals(e) << "\n";
        return Kokkos::abs(spline_eval(e) - vals(e));
	});

	double max_norm_error_diff = ddc::transform_reduce(
            spline_eval.domain(),
			0.,
			ddc::reducer::max<double>(),
           	[&](IndexXY const e) {
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::ddc::ScopeGuard scope(argc, argv);
    return RUN_ALL_TESTS();
}
