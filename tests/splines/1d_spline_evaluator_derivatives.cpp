// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "spline_error_bounds.hpp"

inline namespace anonymous_namespace_workaround_1d_spline_evaluator_derivatives_cpp {

struct DimX
{
    static constexpr bool PERIODIC = true;
};

struct DDimBatch1
{
};

struct DDimBatch2
{
};

constexpr std::size_t s_degree_x = DEGREE_X;

constexpr ddc::BoundCond s_bcl = ddc::BoundCond::PERIODIC;
constexpr ddc::BoundCond s_bcr = ddc::BoundCond::PERIODIC;

template <typename BSpX>
using GrevillePoints = ddc::GrevilleInterpolationPoints<BSpX, s_bcl, s_bcr>;

#if defined(BSPLINES_TYPE_UNIFORM)
template <typename X>
struct BSplines : ddc::UniformBSplines<X, s_degree_x>
{
};
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
struct BSplines : ddc::NonUniformBSplines<X, s_degree_x>
{
};
#endif

// In the dimension of interest, the discrete dimension is deduced from the Greville points type.
template <typename X>
struct DDimGPS : GrevillePoints<BSplines<X>>::interpolation_discrete_dimension_type
{
};

template <typename DDimX>
using evaluator_type = CosineEvaluator::Evaluator<DDimX>;

template <typename... DDimX>
using DElem = ddc::DiscreteElement<DDimX...>;
template <typename... DDimX>
using DVect = ddc::DiscreteVector<DDimX...>;
template <typename... X>
using Coord = ddc::Coordinate<X...>;

constexpr ddc::SplineSolver s_spline_solver = ddc::SplineSolver::GINKGO;

// Templated function giving first coordinate of the mesh in given dimension.
template <typename X>
KOKKOS_FUNCTION Coord<X> x0()
{
    return Coord<X>(0.);
}

// Templated function giving last coordinate of the mesh in given dimension.
template <typename X>
KOKKOS_FUNCTION Coord<X> xn()
{
    return Coord<X>(1.);
}

// Templated function giving step of the mesh in given dimension.
template <typename X>
double dx(std::size_t ncells)
{
    return (xn<X>() - x0<X>()) / ncells;
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

template <class DDim>
void interest_dim_initializer(std::size_t const ncells)
{
    using CDim = typename DDim::continuous_dimension_type;
#if defined(BSPLINES_TYPE_UNIFORM)
    ddc::init_discrete_space<BSplines<CDim>>(x0<CDim>(), xn<CDim>(), ncells);
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
    ddc::init_discrete_space<BSplines<CDim>>(breaks<CDim>(ncells));
#endif
    ddc::init_discrete_space<DDim>(GrevillePoints<BSplines<CDim>>::template get_sampling<DDim>());
}

template <
        class DDimI,
        class ExecSpace,
        class SplineEvaluator,
        class CoordsSpan,
        class CoefSpan,
        class SplineDerivSpan,
        class DElem>
void test_deriv(
        ExecSpace const& exec_space,
        SplineEvaluator const& spline_evaluator,
        CoordsSpan const& coords_eval,
        CoefSpan const& coef,
        SplineDerivSpan const& spline_eval_deriv,
        evaluator_type<DDimI> const& evaluator,
        DElem const& deriv_order,
        std::size_t const ncells)
{
    using domain = decltype(spline_eval_deriv.domain());
    using I = typename DDimI::continuous_dimension_type;

    ddc::parallel_fill(exec_space, spline_eval_deriv, 0.0);
    exec_space.fence();

    spline_evaluator.deriv(deriv_order, spline_eval_deriv, coords_eval, coef);

    auto const order = ddc::select_or(deriv_order, ddc::DiscreteElement<ddc::Deriv<I>>(0)).uid();

    double const max_norm_error_diff = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(typename domain::discrete_element_type const e) {
                Coord<I> const x = ddc::coordinate(ddc::DiscreteElement<DDimI>(e));
                return Kokkos::abs(spline_eval_deriv(e) - evaluator.deriv(x, order));
            });

    double const max_norm_diff = evaluator.max_norm(order);

    SplineErrorBounds<evaluator_type<DDimI>> const error_bounds(evaluator);

    EXPECT_LE(
            max_norm_error_diff,
            std::
                    max(error_bounds.error_bound(
                                std::array<ddc::DiscreteElementType, 1> {order},
                                {dx<I>(ncells)},
                                {s_degree_x}),
                        1e-11 * max_norm_diff));
}

template <
        class DDimI,
        class ExecSpace,
        class SplineEvaluator,
        class CoordsSpan,
        class CoefSpan,
        class SplineDerivSpan>
void launch_deriv_tests(
        ExecSpace const& exec_space,
        SplineEvaluator const& spline_evaluator,
        CoordsSpan const& coords_eval,
        CoefSpan const& coef,
        SplineDerivSpan const& spline_eval_deriv,
        evaluator_type<DDimI> const& evaluator,
        std::size_t const ncells)
{
    using I = typename DDimI::continuous_dimension_type;

    auto const local_test_deriv = [&](auto deriv_order) {
        test_deriv(
                exec_space,
                spline_evaluator,
                coords_eval,
                coef,
                spline_eval_deriv,
                evaluator,
                deriv_order,
                ncells);
    };

    ddc::DiscreteDomain<ddc::Deriv<I>> const
            deriv(ddc::DiscreteElement<ddc::Deriv<I>>(1),
                  ddc::DiscreteVector<ddc::Deriv<I>>(BSplines<I>::degree()));

    local_test_deriv(ddc::DiscreteElement<>());
    for (ddc::DiscreteElement<ddc::Deriv<I>> const order : deriv) {
        local_test_deriv(order);
    }
}

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename DDimI, typename... DDims>
void TestSplineEvaluator1dDerivatives()
{
    using I = typename DDimI::continuous_dimension_type;

    // Instantiate execution spaces and initialize spaces
    ExecSpace const exec_space;

    std::size_t const ncells = 10;
    interest_dim_initializer<DDimI>(ncells);

    // Create the values domain (mesh)
    ddc::DiscreteDomain<DDimI> const interpolation_domain
            = GrevillePoints<BSplines<I>>::template get_domain<DDimI>();
    // The following line creates a discrete domain over all dimensions (DDims...) except DDimI.
    auto const dom_vals_tmp = ddc::remove_dims_of_t<ddc::DiscreteDomain<DDims...>, DDimI>(
            ddc::DiscreteDomain<DDims>(DElem<DDims>(0), DVect<DDims>(ncells))...);
    ddc::DiscreteDomain<DDims...> const dom_vals(dom_vals_tmp, interpolation_domain);

    // Create a SplineBuilder over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilder<
            ExecSpace,
            MemorySpace,
            BSplines<I>,
            DDimI,
            s_bcl,
            s_bcr,
            s_spline_solver> const spline_builder(interpolation_domain);

    // Compute useful domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<DDimI> const dom_interpolation = spline_builder.interpolation_domain();
    auto const dom_spline = spline_builder.batched_spline_domain(dom_vals);

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplicated along batch dimensions
    ddc::Chunk vals_1d_host_alloc(dom_interpolation, ddc::HostAllocator<double>());
    ddc::ChunkSpan const vals_1d_host = vals_1d_host_alloc.span_view();
    evaluator_type<DDimI> const evaluator(dom_interpolation);
    evaluator(vals_1d_host);
    auto vals_1d_alloc = ddc::create_mirror_view_and_copy(exec_space, vals_1d_host);
    ddc::ChunkSpan const vals_1d = vals_1d_alloc.span_view();

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const vals = vals_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            vals.domain(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) { vals(e) = vals_1d(DElem<DDimI>(e)); });

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
    spline_builder(coef, vals.span_cview());

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
    using extrapolation_rule_type = ddc::PeriodicExtrapolationRule<I>;
    extrapolation_rule_type const extrapolation_rule;

    ddc::SplineEvaluator<
            ExecSpace,
            MemorySpace,
            BSplines<I>,
            DDimI,
            extrapolation_rule_type,
            extrapolation_rule_type> const
            spline_evaluator_batched(extrapolation_rule, extrapolation_rule);

    // Instantiate chunk of coordinates of dom_interpolation
    ddc::Chunk coords_eval_alloc(dom_vals, ddc::KokkosAllocator<Coord<I>, MemorySpace>());
    ddc::ChunkSpan const coords_eval = coords_eval_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            coords_eval.domain(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                coords_eval(e) = ddc::coordinate(DElem<DDimI>(e));
            });


    // Instantiate chunks to receive outputs of spline_evaluator
    ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv = spline_eval_deriv_alloc.span_view();

    launch_deriv_tests(
            exec_space,
            spline_evaluator_batched,
            coords_eval.span_cview(),
            coef.span_cview(),
            spline_eval_deriv,
            evaluator,
            ncells);
}

} // namespace anonymous_namespace_workaround_1d_spline_evaluator_derivatives_cpp

#if defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX_DEGREE(name, degree) name##Periodic##Uniform##Degree##degree
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX_DEGREE(name, degree) name##Periodic##NonUniform##Degree##degree
#endif
#define SUFFIX_DEGREE_MACRO_EXP(name, degree) SUFFIX_DEGREE(name, degree)
#define SUFFIX(name) SUFFIX_DEGREE_MACRO_EXP(name, DEGREE_X)

TEST(SUFFIX(SplineEvaluator1dDerivativesHost), 1DX)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesDevice), 1DX)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesHost), 2DXB1)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesHost), 2DB1X)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesDevice), 2DXB1)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesDevice), 2DB1X)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesHost), 3DXB1B2)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesHost), 3DB1XB2)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>,
            DDimBatch2>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesHost), 3DB1B2X)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesDevice), 3DXB1B2)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesDevice), 3DB1XB2)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>,
            DDimBatch2>();
}

TEST(SUFFIX(SplineEvaluator1dDerivativesDevice), 3DB1B2X)
{
    TestSplineEvaluator1dDerivatives<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2,
            DDimGPS<DimX>>();
}
