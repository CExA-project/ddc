// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "spline_error_bounds.hpp"

inline namespace anonymous_namespace_workaround_periodicity_spline_builder_cpp {

struct DimX
{
    static constexpr bool PERIODIC = true;
};

constexpr std::size_t s_degree_x = DEGREE_X;

template <typename BSpX>
using GrevillePoints = ddc::
        GrevilleInterpolationPoints<BSpX, ddc::BoundCond::PERIODIC, ddc::BoundCond::PERIODIC>;

#if defined(BSPLINES_TYPE_UNIFORM)
template <typename X>
struct BSplines : ddc::UniformBSplines<X, s_degree_x>
{
};

// In the dimension of interest, the discrete dimension is deduced from the Greville points type.
template <typename X>
struct DDim : GrevillePoints<BSplines<X>>::interpolation_discrete_dimension_type
{
};

#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
struct BSplines : ddc::NonUniformBSplines<X, s_degree_x>
{
};

template <typename X>
struct DDim : GrevillePoints<BSplines<X>>::interpolation_discrete_dimension_type
{
};

#endif
template <typename DDimX>
using evaluator_type = CosineEvaluator::Evaluator<DDimX>;

template <typename... DDimX>
using DElem = ddc::DiscreteElement<DDimX...>;
template <typename... DDimX>
using DVect = ddc::DiscreteVector<DDimX...>;
template <typename... X>
using Coord = ddc::Coordinate<X...>;

// Templated function giving first coordinate of the mesh in given dimension.
template <typename X>
Coord<X> x0()
{
    return Coord<X>(0.);
}

// Templated function giving last coordinate of the mesh in given dimension.
template <typename X>
Coord<X> xN()
{
    return Coord<X>(1.);
}

// Templated function giving step of the mesh in given dimension.
template <typename X>
double dx(std::size_t ncells)
{
    return (xN<X>() - x0<X>()) / ncells;
}

// Templated function giving break points of mesh in given dimension for non-uniform case.
template <typename X>
std::vector<Coord<X>> breaks(std::size_t ncells)
{
    std::vector<Coord<X>> out(ncells + 1);
    for (std::size_t i(0); i < ncells + 1; ++i) {
        out[i] = x0<X>() + i * dx<X>(ncells);
    }
    return out;
}

// Helper to initialize space
template <class DDim>
void InterestDimInitializer(std::size_t const ncells)
{
    using CDim = typename DDim::continuous_dimension_type;
#if defined(BSPLINES_TYPE_UNIFORM)
    ddc::init_discrete_space<BSplines<CDim>>(x0<CDim>(), xN<CDim>(), ncells);
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
    ddc::init_discrete_space<BSplines<CDim>>(breaks<CDim>(ncells));
#endif
    ddc::init_discrete_space<DDim>(GrevillePoints<BSplines<CDim>>::template get_sampling<DDim>());
}

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename X>
void PeriodicitySplineBuilderTest()
{
    // Instantiate execution spaces and initialize spaces
    ExecSpace const exec_space;

    std::size_t const ncells = 10;
    InterestDimInitializer<DDim<X>>(ncells);

    // Create the values domain (mesh)
    ddc::DiscreteDomain<DDim<X>> const dom_vals
            = GrevillePoints<BSplines<X>>::template get_domain<DDim<X>>();

    // Create a SplineBuilder over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilder<
            ExecSpace,
            MemorySpace,
            BSplines<X>,
            DDim<X>,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC,
            ddc::SplineSolver::GINKGO> const spline_builder(dom_vals);

    // Compute useful domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<BSplines<X>> const dom_bsplines = spline_builder.spline_domain();

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplicated along batch dimensions
    ddc::Chunk vals_host_alloc(dom_vals, ddc::HostAllocator<double>());
    ddc::ChunkSpan const vals_host = vals_host_alloc.span_view();
    evaluator_type<DDim<X>> const evaluator(dom_vals);
    evaluator(vals_host);
    auto vals_alloc = ddc::create_mirror_view_and_copy(exec_space, vals_host);
    ddc::ChunkSpan const vals = vals_alloc.span_view();

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_bsplines, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();

    // Instantiate empty chunk of derivatives
    ddc::ChunkSpan<double const, ddc::StridedDiscreteDomain<DDim<X>, ddc::Deriv<X>>> const
            derivs {};

    // Finally compute the spline by filling `coef`
    spline_builder(coef, vals.span_cview(), derivs.span_cview());

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
    ddc::PeriodicExtrapolationRule<X> const extrapolation_rule;
    ddc::SplineEvaluator<
            ExecSpace,
            MemorySpace,
            BSplines<X>,
            DDim<X>,
            ddc::PeriodicExtrapolationRule<X>,
            ddc::PeriodicExtrapolationRule<X>> const
            spline_evaluator(extrapolation_rule, extrapolation_rule);

    // Instantiate chunk of coordinates of dom_interpolation
    ddc::Chunk coords_eval_alloc(dom_vals, ddc::KokkosAllocator<Coord<X>, MemorySpace>());
    ddc::ChunkSpan const coords_eval = coords_eval_alloc.span_view();
    // Translate function 1.5x domain width to the right.
    Coord<X> const displ(1.5);
    ddc::parallel_for_each(
            exec_space,
            coords_eval.domain(),
            KOKKOS_LAMBDA(DElem<DDim<X>> const e) { coords_eval(e) = ddc::coordinate(e) + displ; });


    // Instantiate chunks to receive outputs of spline_evaluator
    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval = spline_eval_alloc.span_view();

    // Call spline_evaluator on the same mesh we started with
    spline_evaluator(spline_eval, coords_eval.span_cview(), coef.span_cview());

    // Checking errors (we recover the initial values)
    double const max_norm_error = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDim<X>> const e) {
                return Kokkos::abs(
                        spline_eval(e)
                        - (-vals(e))); // Because function is even, we get f_eval = -f
            });

    double const max_norm = evaluator.max_norm();

    SplineErrorBounds<evaluator_type<DDim<X>>> const error_bounds(evaluator);
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(dx<X>(ncells), s_degree_x), 1.0e-14 * max_norm));
}

} // namespace anonymous_namespace_workaround_periodicity_spline_builder_cpp

TEST(PeriodicitySplineBuilderHost, 1D)
{
    PeriodicitySplineBuilderTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX>();
}

TEST(PeriodicitySplineBuilderDevice, 1D)
{
    PeriodicitySplineBuilderTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX>();
}
