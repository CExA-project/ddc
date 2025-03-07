// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#if defined(BSPLINES_TYPE_NON_UNIFORM)
#include <vector>
#endif

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "spline_error_bounds.hpp"

struct DimX
{
    static constexpr bool PERIODIC = true;
};

static constexpr std::size_t s_degree_x = DEGREE_X;

#if defined(BSPLINES_TYPE_UNIFORM)
struct BSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
struct BSplinesX : ddc::NonUniformBSplines<DimX, s_degree_x>
{
};
#endif

using GrevillePoints = ddc::
        GrevilleInterpolationPoints<BSplinesX, ddc::BoundCond::PERIODIC, ddc::BoundCond::PERIODIC>;

struct DDimX : GrevillePoints::interpolation_discrete_dimension_type
{
};

using evaluator_type = CosineEvaluator::Evaluator<DDimX>;

using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using CoordX = ddc::Coordinate<DimX>;

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
void TestPeriodicSplineBuilderTestIdentity()
{
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;

    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 10;

    // 1. Create BSplines
    {
#if defined(BSPLINES_TYPE_UNIFORM)
        ddc::init_discrete_space<BSplinesX>(x0, xN, ncells);
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
        DVectX const npoints(ncells + 1);
        std::vector<CoordX> breaks(npoints);
        double const dx = (xN - x0) / ncells;
        for (int i(0); i < npoints; ++i) {
            breaks[i] = CoordX(x0 + i * dx);
        }
        ddc::init_discrete_space<BSplinesX>(breaks);
#endif
    }
    ddc::DiscreteDomain<BSplinesX> const dom_bsplines_x(
            ddc::discrete_space<BSplinesX>().full_domain());

    // 2. Create a Spline represented by a chunk over BSplines
    // The chunk is filled with garbage data, we need to initialize it
    ddc::Chunk coef(dom_bsplines_x, ddc::KokkosAllocator<double, memory_space>());

    // 3. Create the interpolation domain
    ddc::init_discrete_space<DDimX>(GrevillePoints::get_sampling<DDimX>());
    ddc::DiscreteDomain<DDimX> const interpolation_domain(GrevillePoints::get_domain<DDimX>());

    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    ddc::SplineBuilder<
            execution_space,
            memory_space,
            BSplinesX,
            DDimX,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC,
            ddc::SplineSolver::GINKGO> const spline_builder(interpolation_domain);

    // 5. Allocate and fill a chunk over the interpolation domain
    ddc::Chunk yvals_alloc(interpolation_domain, ddc::KokkosAllocator<double, memory_space>());
    ddc::ChunkSpan const yvals(yvals_alloc.span_view());
    evaluator_type const evaluator(interpolation_domain);
    ddc::parallel_for_each(
            execution_space(),
            yvals.domain(),
            KOKKOS_LAMBDA(DElemX const ix) { yvals(ix) = evaluator(ddc::coordinate(ix)); });

    // 6. Finally build the spline by filling `coef`
    spline_builder(coef.span_view(), yvals.span_cview());

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    ddc::PeriodicExtrapolationRule<DimX> const periodic_extrapolation;
    ddc::SplineEvaluator<
            execution_space,
            memory_space,
            BSplinesX,
            DDimX,
            ddc::PeriodicExtrapolationRule<DimX>,
            ddc::PeriodicExtrapolationRule<DimX>> const
            spline_evaluator(periodic_extrapolation, periodic_extrapolation);

    ddc::Chunk
            coords_eval_alloc(interpolation_domain, ddc::KokkosAllocator<CoordX, memory_space>());
    ddc::ChunkSpan const coords_eval(coords_eval_alloc.span_view());
    ddc::parallel_for_each(
            execution_space(),
            interpolation_domain,
            KOKKOS_LAMBDA(DElemX const ix) { coords_eval(ix) = ddc::coordinate(ix); });

    ddc::Chunk
            spline_eval_alloc(interpolation_domain, ddc::KokkosAllocator<double, memory_space>());
    ddc::ChunkSpan const spline_eval(spline_eval_alloc.span_view());
    spline_evaluator(spline_eval.span_view(), coords_eval.span_cview(), coef.span_cview());

    ddc::Chunk spline_eval_deriv_alloc(
            interpolation_domain,
            ddc::KokkosAllocator<double, memory_space>());
    ddc::ChunkSpan const spline_eval_deriv(spline_eval_deriv_alloc.span_view());
    spline_evaluator
            .deriv(spline_eval_deriv.span_view(), coords_eval.span_cview(), coef.span_cview());

    ddc::Chunk integral(
            spline_builder.batch_domain(interpolation_domain),
            ddc::KokkosAllocator<double, memory_space>());
    spline_evaluator.integrate(integral.span_view(), coef.span_cview());

    ddc::Chunk<double, ddc::DiscreteDomain<DDimX>, ddc::KokkosAllocator<double, memory_space>>
            quadrature_coefficients_alloc;
    std::tie(std::ignore, quadrature_coefficients_alloc, std::ignore)
            = spline_builder.quadrature_coefficients();
    ddc::ChunkSpan const quadrature_coefficients = quadrature_coefficients_alloc.span_cview();
    double const quadrature_integral = ddc::parallel_transform_reduce(
            execution_space(),
            quadrature_coefficients.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(DElemX const ix) { return quadrature_coefficients(ix) * yvals(ix); });

    // 8. Checking errors
    double const max_norm_error = ddc::parallel_transform_reduce(
            execution_space(),
            interpolation_domain,
            0.0,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElemX const ix) {
                double const error = spline_eval(ix) - yvals(ix);
                return Kokkos::fabs(error);
            });
    double const max_norm_error_diff = ddc::parallel_transform_reduce(
            execution_space(),
            interpolation_domain,
            0.0,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElemX const ix) {
                CoordX const x = ddc::coordinate(ix);
                double const error_deriv = spline_eval_deriv(ix) - evaluator.deriv(x, 1);
                return Kokkos::fabs(error_deriv);
            });

    auto integral_host = ddc::create_mirror_view_and_copy(integral.span_view());
    double const max_norm_error_integ = std::fabs(
            integral_host(ddc::DiscreteElement<>()) - evaluator.deriv(xN, -1)
            + evaluator.deriv(x0, -1));
    double const max_norm_error_quadrature_integ
            = std::fabs(quadrature_integral - evaluator.deriv(xN, -1) + evaluator.deriv(x0, -1));

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type> const error_bounds(evaluator);
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
    EXPECT_LE(
            max_norm_error_quadrature_integ,
            std::max(error_bounds.error_bound_on_int(h, s_degree_x), 1.0e-14 * max_norm_int));
}

TEST(PeriodicSplineBuilderTest, Identity)
{
    TestPeriodicSplineBuilderTestIdentity();
}
