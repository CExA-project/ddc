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
    SplineXFieldY coef(dom_coef);

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
              [&](ddc::DiscreteElement<IDimX,IDimY> const e) {
				  vals(e) = vals1(ddc::select<IDimX>(e));
              });
    // 6. Finally build the spline by filling `coef`
    spline_builder(coef, vals);

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    SplineEvaluator<BSplinesX> spline_evaluator(g_null_boundary<BSplinesX>, g_null_boundary<BSplinesX>);

    ddc::Chunk<CoordX, ddc::DiscreteDomain<IDimX,IDimY>> coords_eval(dom_vals);
	ddc::for_each(
            ddc::policies::serial_host,
            coords_eval.domain(),
            DDC_LAMBDA(ddc::DiscreteElement<IDimX,IDimY> const e) {
		IndexX ix = ddc::select<IDimX>(e);
        coords_eval(e) = ddc::coordinate(ix);
    });
	#if 0
	ddc::Chunk<ddc::Coordinate<DimX>, ddc::DiscreteDomain<IDimX>> coords_eval(interpolation_domain);
    for (ddc::DiscreteElement<IDimX> const ix : interpolation_domain) {
        coords_eval(ix) = ddc::coordinate(ix);
    }
    FieldXY spline_eval(dom_vals);
	ddc::for_each(
            ddc::policies::serial_host,
            dom_y,
            DDC_LAMBDA(ddc::DiscreteElement<IDimY> const iy) {
	    spline_evaluator(spline_eval[iy].span_view(), coords_eval[iy].span_cview(), coef[iy].span_cview());
	});

    FieldX spline_eval_deriv(interpolation_domain);
    spline_evaluator
            .deriv(spline_eval_deriv.span_view(), coords_eval.span_cview(), coef.span_cview());

    // 8. Checking errors
    double max_norm_error = 0.;
    double max_norm_error_diff = 0.;
    for (IndexX const ix : interpolation_domain) {
        CoordX const x = ddc::coordinate(ix);

        // Compute error
        double const error = spline_eval(ix) - yvals(ix);
        max_norm_error = std::fmax(max_norm_error, std::fabs(error));

        // Compute error
        double const error_deriv = spline_eval_deriv(ix) - evaluator.deriv(x, 1);
        max_norm_error_diff = std::fmax(max_norm_error_diff, std::fabs(error_deriv));
    }
    double const max_norm_error_integ = std::fabs(
            spline_evaluator.integrate(coef.span_cview()) - evaluator.deriv(xN, -1)
            + evaluator.deriv(x0, -1));

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type> error_bounds(evaluator);
    const double h = (xN - x0) / ncells;
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(h, s_degree_x), 1.0e-14 * max_norm));
    EXPECT_LE(
            max_norm_error_diff,
            std::max(error_bounds.error_bound_on_deriv(h, s_degree_x), 1e-12 * max_norm_diff));
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
