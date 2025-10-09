// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#if defined(BSPLINES_TYPE_NON_UNIFORM)
#    include <vector>
#endif

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#if defined(EVALUATOR_COSINE)
#    include "cosine_evaluator.hpp"
#endif
#include "polynomial_evaluator.hpp"
#include "spline_error_bounds.hpp"

struct DimX
{
    static constexpr bool PERIODIC = false;
};

static constexpr std::size_t s_degree_x = DEGREE_X;

#if defined(BCL_GREVILLE)
static constexpr ddc::BoundCond s_bcl = ddc::BoundCond::GREVILLE;
#elif defined(BCL_HERMITE)
static constexpr ddc::BoundCond s_bcl = ddc::BoundCond::HERMITE;
#endif

#if defined(BCR_GREVILLE)
static constexpr ddc::BoundCond s_bcr = ddc::BoundCond::GREVILLE;
#elif defined(BCR_HERMITE)
static constexpr ddc::BoundCond s_bcr = ddc::BoundCond::HERMITE;
#endif

#if defined(BSPLINES_TYPE_UNIFORM)
struct BSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
struct BSplinesX : ddc::NonUniformBSplines<DimX, s_degree_x>
{
};
#endif

using GrevillePoints = ddc::GrevilleInterpolationPoints<BSplinesX, s_bcl, s_bcr>;

struct DDimX : GrevillePoints::interpolation_discrete_dimension_type
{
};

#if defined(EVALUATOR_COSINE)
using evaluator_type = CosineEvaluator::Evaluator<DDimX>;
#elif defined(EVALUATOR_POLYNOMIAL)
using evaluator_type = PolynomialEvaluator::Evaluator<DDimX, s_degree_x>;
#endif

using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using SplineX = ddc::Chunk<double, ddc::DiscreteDomain<BSplinesX>>;
using FieldX = ddc::Chunk<double, ddc::DiscreteDomain<DDimX>>;
using CoordX = ddc::Coordinate<DimX>;

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
void TestNonPeriodicSplineBuilderTestIdentity()
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
    ddc::DiscreteDomain<ddc::Deriv<DimX>> const derivs_domain(
            ddc::DiscreteElement<ddc::Deriv<DimX>>(1),
            ddc::DiscreteVector<ddc::Deriv<DimX>>(s_degree_x / 2));

    // 2. Create a Spline represented by a chunk over BSplines
    // The chunk is filled with garbage data, we need to initialize it
    ddc::Chunk coef(dom_bsplines_x, ddc::KokkosAllocator<double, memory_space>());

    // 3. Create the interpolation and deriv domains
    ddc::init_discrete_space<DDimX>(GrevillePoints::get_sampling<DDimX>());
    ddc::DiscreteDomain<DDimX> const interpolation_domain(GrevillePoints::get_domain<DDimX>());

    auto const whole_derivs_domain = ddc::detail::get_whole_derivs_domain<
            ddc::Deriv<DimX>>(interpolation_domain, s_degree_x);
    // ddc::StridedDiscreteDomain<DDimX, ddc::Deriv<DimX>> const whole_derivs_domain(
    //         ddc::DiscreteElement<DDimX, ddc::Deriv<DimX>>(0, 1),
    //         ddc::DiscreteVector<DDimX, ddc::Deriv<DimX>>(2, s_degree_x / 2),
    //         ddc::DiscreteVector<
    //                 DDimX,
    //                 ddc::Deriv<DimX>>(interpolation_domain.extent<DDimX>().value() - 2, 1));

    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    ddc::SplineBuilder<
            execution_space,
            memory_space,
            BSplinesX,
            DDimX,
            s_bcl,
            s_bcr,
            ddc::SplineSolver::GINKGO> const spline_builder(interpolation_domain);

    // 5. Allocate and fill a chunk over the interpolation domain
    ddc::Chunk yvals_alloc(interpolation_domain, ddc::KokkosAllocator<double, memory_space>());
    ddc::ChunkSpan const yvals(yvals_alloc.span_view());
    evaluator_type const evaluator(interpolation_domain);
    ddc::parallel_for_each(
            execution_space(),
            yvals.domain(),
            KOKKOS_LAMBDA(DElemX const ix) { yvals(ix) = evaluator(ddc::coordinate(ix)); });

    int const shift = s_degree_x % 2; // shift = 0 for even order, 1 for odd order

    ddc::Chunk derivs_alloc(whole_derivs_domain, ddc::KokkosAllocator<double, memory_space>());
    ddc::ChunkSpan const derivs = derivs_alloc.span_view();

    ddc::ChunkSpan const derivs_lhs = derivs[interpolation_domain.front()];
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::parallel_for_each(
                execution_space(),
                derivs_domain,
                KOKKOS_LAMBDA(ddc::DiscreteElement<ddc::Deriv<DimX>> const ii) {
                    derivs_lhs(ii) = evaluator.deriv(x0, ii - derivs_domain.front() + shift);
                });
    }

    ddc::ChunkSpan const derivs_rhs = derivs[interpolation_domain.back()];
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::parallel_for_each(
                execution_space(),
                derivs_domain,
                KOKKOS_LAMBDA(ddc::DiscreteElement<ddc::Deriv<DimX>> const ii) {
                    derivs_rhs(ii) = evaluator.deriv(xN, ii - derivs_domain.front() + shift);
                });
    }

    // 6. Finally build the spline by filling `coef`
    spline_builder(coef.span_view(), yvals.span_cview(), derivs.span_cview());

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    ddc::NullExtrapolationRule const extrapolation_rule;
    ddc::SplineEvaluator<
            execution_space,
            memory_space,
            BSplinesX,
            DDimX,
            ddc::NullExtrapolationRule,
            ddc::NullExtrapolationRule> const
            spline_evaluator(extrapolation_rule, extrapolation_rule);

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

    ddc::Chunk<
            double,
            ddc::DiscreteDomain<ddc::Deriv<typename DDimX::continuous_dimension_type>>,
            ddc::KokkosAllocator<double, memory_space>>
            quadrature_coefficients_derivs_xmin_alloc;
    ddc::Chunk<double, ddc::DiscreteDomain<DDimX>, ddc::KokkosAllocator<double, memory_space>>
            quadrature_coefficients_alloc;
    ddc::Chunk<
            double,
            ddc::DiscreteDomain<ddc::Deriv<typename DDimX::continuous_dimension_type>>,
            ddc::KokkosAllocator<double, memory_space>>
            quadrature_coefficients_derivs_xmax_alloc;
    std::
            tie(quadrature_coefficients_derivs_xmin_alloc,
                quadrature_coefficients_alloc,
                quadrature_coefficients_derivs_xmax_alloc)
            = spline_builder.quadrature_coefficients();
    ddc::ChunkSpan const quadrature_coefficients(quadrature_coefficients_alloc.span_view());
#if defined(BCL_HERMITE)
    ddc::ChunkSpan const quadrature_coefficients_derivs_xmin(
            quadrature_coefficients_derivs_xmin_alloc.span_view());
    double const quadrature_integral_derivs_xmin = ddc::parallel_transform_reduce(
            execution_space(),
            quadrature_coefficients_derivs_xmin.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(
                    ddc::DiscreteElement<
                            ddc::Deriv<typename DDimX::continuous_dimension_type>> const ix) {
                return quadrature_coefficients_derivs_xmin(ix) * derivs_lhs(ix);
            });
#else
    double const quadrature_integral_derivs_xmin = 0.;
#endif
    double quadrature_integral = ddc::parallel_transform_reduce(
            execution_space(),
            quadrature_coefficients.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX> const ix) {
                return quadrature_coefficients(ix) * yvals(ix);
            });
#if defined(BCR_HERMITE)
    ddc::ChunkSpan const quadrature_coefficients_derivs_xmax(
            quadrature_coefficients_derivs_xmax_alloc.span_view());
    double const quadrature_integral_derivs_xmax = ddc::parallel_transform_reduce(
            execution_space(),
            quadrature_coefficients_derivs_xmax.domain(),
            0.0,
            ddc::reducer::sum<double>(),
            KOKKOS_LAMBDA(
                    ddc::DiscreteElement<
                            ddc::Deriv<typename DDimX::continuous_dimension_type>> const ix) {
                return quadrature_coefficients_derivs_xmax(ix) * derivs_rhs(ix);
            });
#else
    double const quadrature_integral_derivs_xmax = 0.;
#endif
    quadrature_integral += quadrature_integral_derivs_xmin + quadrature_integral_derivs_xmax;

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
    if constexpr (std::is_same_v<
                          evaluator_type,
                          PolynomialEvaluator::Evaluator<DDimX, s_degree_x>>) {
        EXPECT_LE(max_norm_error / max_norm, 1.0e-14);
        EXPECT_LE(max_norm_error_diff / max_norm_diff, 1.0e-12);
        EXPECT_LE(max_norm_error_integ / max_norm_int, 1.0e-14);
        EXPECT_LE(max_norm_error_quadrature_integ / max_norm_int, 1.0e-14);
    } else {
        SplineErrorBounds<evaluator_type> const error_bounds(evaluator);
        double const h = (xN - x0) / ncells;
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
}

TEST(NonPeriodicSplineBuilderTest, Identity)
{
    TestNonPeriodicSplineBuilderTestIdentity();
}
