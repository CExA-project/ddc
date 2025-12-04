// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#if defined(BSPLINES_TYPE_UNIFORM)
#    include <type_traits>
#endif
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "evaluator_3d.hpp"
#include "spline_error_bounds.hpp"

inline namespace anonymous_namespace_workaround_3d_spline_evaluator_derivatives_cpp {

struct DimX
{
    static constexpr bool PERIODIC = true;
};

struct DimY
{
    static constexpr bool PERIODIC = true;
};

struct DimZ
{
    static constexpr bool PERIODIC = true;
};

struct DDimBatch
{
};

constexpr std::size_t s_degree = DEGREE;

constexpr ddc::BoundCond s_bcl = ddc::BoundCond::PERIODIC;
constexpr ddc::BoundCond s_bcr = ddc::BoundCond::PERIODIC;

template <typename BSpX>
using GrevillePoints = ddc::GrevilleInterpolationPoints<BSpX, s_bcl, s_bcr>;

#if defined(BSPLINES_TYPE_UNIFORM)
template <typename X>
struct BSplines : ddc::UniformBSplines<X, s_degree>
{
};
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
struct BSplines : ddc::NonUniformBSplines<X, s_degree>
{
};
#endif

// In the dimensions of interest, the discrete dimension is deduced from the Greville points type.
template <typename X>
struct DDimGPS : GrevillePoints<BSplines<X>>::interpolation_discrete_dimension_type
{
};

template <typename DDim1, typename DDim2, typename DDim3>
using evaluator_type = Evaluator3D::Evaluator<
        CosineEvaluator::Evaluator<DDim1>,
        CosineEvaluator::Evaluator<DDim2>,
        CosineEvaluator::Evaluator<DDim3>>;

template <typename... DDimX>
using DElem = ddc::DiscreteElement<DDimX...>;
template <typename... DDimX>
using DVect = ddc::DiscreteVector<DDimX...>;
template <typename... X>
using Coord = ddc::Coordinate<X...>;

// Templated function giving first coordinate of the mesh in given dimension.
template <typename X>
KOKKOS_FUNCTION Coord<X> x0()
{
    return Coord<X>(0.);
}

// Templated function giving last coordinate of the mesh in given dimension.
template <typename X>
KOKKOS_FUNCTION Coord<X> xN()
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

template <
        class DDimI1,
        class DDimI2,
        class DDimI3,
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
        evaluator_type<DDimI1, DDimI2, DDimI3> const& evaluator,
        DElem const& deriv_order,
        std::size_t const ncells)
{
    using domain = decltype(spline_eval_deriv.domain());
    using I1 = typename DDimI1::continuous_dimension_type;
    using I2 = typename DDimI2::continuous_dimension_type;
    using I3 = typename DDimI3::continuous_dimension_type;

    ddc::parallel_fill(exec_space, spline_eval_deriv, 0.0);
    exec_space.fence();

    spline_evaluator.deriv(deriv_order, spline_eval_deriv, coords_eval, coef);

    auto const order1 = ddc::select_or(deriv_order, ddc::DiscreteElement<ddc::Deriv<I1>>(0)).uid();
    auto const order2 = ddc::select_or(deriv_order, ddc::DiscreteElement<ddc::Deriv<I2>>(0)).uid();
    auto const order3 = ddc::select_or(deriv_order, ddc::DiscreteElement<ddc::Deriv<I3>>(0)).uid();

    double const max_norm_error_diff = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(typename domain::discrete_element_type const e) {
                Coord<I1> const x = ddc::coordinate(ddc::DiscreteElement<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(ddc::DiscreteElement<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(ddc::DiscreteElement<DDimI3>(e));
                return Kokkos::abs(
                        spline_eval_deriv(e) - evaluator.deriv(x, y, z, order1, order2, order3));
            });

    double const max_norm_diff = evaluator.max_norm(order1, order2);

    SplineErrorBounds<evaluator_type<DDimI1, DDimI2, DDimI3>> const error_bounds(evaluator);

    EXPECT_LE(
            max_norm_error_diff,
            std::
                    max(error_bounds.error_bound(
                                std::array<ddc::DiscreteElementType, 3> {order1, order2, order3},
                                {dx<I1>(ncells), dx<I2>(ncells), dx<I3>(ncells)},
                                {s_degree, s_degree, s_degree}),
                        1e-11 * max_norm_diff));
}

template <
        typename I1,
        typename I2,
        typename I3,
        std::size_t order1,
        std::size_t order2,
        std::size_t order3>
auto make_deriv_order_delem()
{
    if constexpr (order1 == 0 && order2 == 0 && order3 == 0) {
        return ddc::DiscreteElement<>();
    } else if constexpr (order1 == 0 && order2 == 0 && order3 >= 1) {
        return ddc::DiscreteElement<ddc::Deriv<I3>>(order3);
    } else if constexpr (order1 == 0 && order2 >= 1 && order3 == 0) {
        return ddc::DiscreteElement<ddc::Deriv<I2>>(order2);
    } else if constexpr (order1 == 0 && order2 >= 1 && order3 >= 1) {
        return ddc::DiscreteElement<ddc::Deriv<I2>, ddc::Deriv<I3>>(order2, order3);
    } else if constexpr (order1 >= 1 && order2 == 0 && order3 == 0) {
        return ddc::DiscreteElement<ddc::Deriv<I1>>(order1);
    } else if constexpr (order1 >= 1 && order2 == 0 && order3 >= 1) {
        return ddc::DiscreteElement<ddc::Deriv<I1>, ddc::Deriv<I3>>(order1, order3);
    } else if constexpr (order1 >= 1 && order2 >= 1 && order3 == 0) {
        return ddc::DiscreteElement<ddc::Deriv<I1>, ddc::Deriv<I2>>(order1, order2);
    } else {
        return ddc::DiscreteElement<
                ddc::Deriv<I1>,
                ddc::Deriv<I2>,
                ddc::Deriv<I3>>(order1, order2, order3);
    }
}

template <
        std::size_t order1 = 0,
        std::size_t order2 = 0,
        std::size_t order3 = 0,
        class DDimI1,
        class DDimI2,
        class DDimI3,
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
        evaluator_type<DDimI1, DDimI2, DDimI3> const& evaluator,
        std::size_t const ncells)
{
    using I1 = typename DDimI1::continuous_dimension_type;
    using I2 = typename DDimI2::continuous_dimension_type;
    using I3 = typename DDimI3::continuous_dimension_type;
    constexpr std::size_t max_deriv_deg1 = BSplines<I1>::degree();
    constexpr std::size_t max_deriv_deg2 = BSplines<I2>::degree();
    constexpr std::size_t max_deriv_deg3 = BSplines<I3>::degree();

    if constexpr (order1 > max_deriv_deg1 || order2 > max_deriv_deg2 || order3 > max_deriv_deg3) {
        return;
    } else {
        // This is used to distribute the work between the 4 test executables.
        // For max deriv degrees (3,3,3), TEST1 will evaluate (0,0,0), TEST2 (0,0,1),
        // TEST3 (0,0,2), TEST4 (0,0,3), TEST1 (0,1,0), etc.
        static constexpr std::size_t test_idx =
#if defined(SPLINE_EVALUATOR_3D_DERIV_TEST_1)
                0;
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_2)
                1;
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_3)
                2;
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_4)
                3;
#endif

        constexpr std::size_t idx = order1 * (max_deriv_deg2 + 1) * (max_deriv_deg3 + 1)
                                    + order2 * (max_deriv_deg3 + 1) + order3;

        if constexpr (idx % 4 == test_idx) {
            test_deriv(
                    exec_space,
                    spline_evaluator,
                    coords_eval,
                    coef,
                    spline_eval_deriv,
                    evaluator,
                    make_deriv_order_delem<I1, I2, I3, order1, order2, order3>(),
                    ncells);
        }

        constexpr std::size_t next_order1
                = order1 + (order2 + order3) / (max_deriv_deg2 + max_deriv_deg3);
        constexpr std::size_t next_order2
                = (order2 + order3 / max_deriv_deg3) % (max_deriv_deg2 + 1);
        constexpr std::size_t next_order3 = (order3 + 1) % (max_deriv_deg3 + 1);
        launch_deriv_tests<next_order1, next_order2, next_order3>(
                exec_space,
                spline_evaluator,
                coords_eval,
                coef,
                spline_eval_deriv,
                evaluator,
                ncells);
    }
}

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <
        typename ExecSpace,
        typename MemorySpace,
        typename DDimI1,
        typename DDimI2,
        typename DDimI3,
        typename... DDims>
void SplineEvaluator3dDerivativesTest()
{
    using I1 = typename DDimI1::continuous_dimension_type;
    using I2 = typename DDimI2::continuous_dimension_type;
    using I3 = typename DDimI3::continuous_dimension_type;

    // Instantiate execution spaces and initialize spaces
    ExecSpace const exec_space;
    std::size_t const ncells = 10;
    InterestDimInitializer<DDimI1>(ncells);
    InterestDimInitializer<DDimI2>(ncells);
    InterestDimInitializer<DDimI3>(ncells);

    // Create the values domain (mesh)
    ddc::DiscreteDomain<DDimI1> const interpolation_domain1
            = GrevillePoints<BSplines<I1>>::template get_domain<DDimI1>();
    ddc::DiscreteDomain<DDimI2> const interpolation_domain2
            = GrevillePoints<BSplines<I2>>::template get_domain<DDimI2>();
    ddc::DiscreteDomain<DDimI3> const interpolation_domain3
            = GrevillePoints<BSplines<I3>>::template get_domain<DDimI3>();
    ddc::DiscreteDomain<DDimI1, DDimI2, DDimI3> const interpolation_domain(
            interpolation_domain1,
            interpolation_domain2,
            interpolation_domain3);
    // The following line creates a discrete domain over all dimensions (DDims...) except DDimI1, DDimI2 and DDimI3.
    auto const dom_vals_tmp
            = ddc::remove_dims_of_t<ddc::DiscreteDomain<DDims...>, DDimI1, DDimI2, DDimI3>(
                    ddc::DiscreteDomain<DDims>(DElem<DDims>(0), DVect<DDims>(ncells))...);
    ddc::DiscreteDomain<DDims...> const dom_vals(
            dom_vals_tmp,
            interpolation_domain1,
            interpolation_domain2,
            interpolation_domain3);

    // Create a SplineBuilder over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilder3D<
            ExecSpace,
            MemorySpace,
            BSplines<I1>,
            BSplines<I2>,
            BSplines<I3>,
            DDimI1,
            DDimI2,
            DDimI3,
            s_bcl,
            s_bcr,
            s_bcl,
            s_bcr,
            s_bcl,
            s_bcr,
            ddc::SplineSolver::GINKGO> const spline_builder(interpolation_domain);

    // Compute useful domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<DDimI1, DDimI2, DDimI3> const dom_interpolation
            = spline_builder.interpolation_domain();
    auto const dom_spline = spline_builder.batched_spline_domain(dom_vals);

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplicated along batch dimensions
    ddc::Chunk vals_1d_host_alloc(dom_interpolation, ddc::HostAllocator<double>());
    ddc::ChunkSpan const vals_1d_host = vals_1d_host_alloc.span_view();
    evaluator_type<DDimI1, DDimI2, DDimI3> const evaluator(dom_interpolation);
    evaluator(vals_1d_host);
    auto vals_1d_alloc = ddc::create_mirror_view_and_copy(exec_space, vals_1d_host);
    ddc::ChunkSpan const vals_1d = vals_1d_alloc.span_view();

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const vals = vals_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            vals.domain(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                vals(e) = vals_1d(DElem<DDimI1, DDimI2, DDimI3>(e));
            });

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
    spline_builder(coef, vals.span_cview());

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
    using extrapolation_rule_1_type = ddc::PeriodicExtrapolationRule<I1>;
    using extrapolation_rule_2_type = ddc::PeriodicExtrapolationRule<I2>;
    using extrapolation_rule_3_type = ddc::PeriodicExtrapolationRule<I3>;

    extrapolation_rule_1_type const extrapolation_rule_1;
    extrapolation_rule_2_type const extrapolation_rule_2;
    extrapolation_rule_3_type const extrapolation_rule_3;

    ddc::SplineEvaluator3D<
            ExecSpace,
            MemorySpace,
            BSplines<I1>,
            BSplines<I2>,
            BSplines<I3>,
            DDimI1,
            DDimI2,
            DDimI3,
            extrapolation_rule_1_type,
            extrapolation_rule_1_type,
            extrapolation_rule_2_type,
            extrapolation_rule_2_type,
            extrapolation_rule_3_type,
            extrapolation_rule_3_type> const
            spline_evaluator(
                    extrapolation_rule_1,
                    extrapolation_rule_1,
                    extrapolation_rule_2,
                    extrapolation_rule_2,
                    extrapolation_rule_3,
                    extrapolation_rule_3);

    // Instantiate chunk of coordinates of dom_interpolation
    ddc::Chunk coords_eval_alloc(dom_vals, ddc::KokkosAllocator<Coord<I1, I2, I3>, MemorySpace>());
    ddc::ChunkSpan const coords_eval = coords_eval_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            coords_eval.domain(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                coords_eval(e) = ddc::coordinate(DElem<DDimI1, DDimI2, DDimI3>(e));
            });


    // Instantiate chunks to receive outputs of spline_evaluator
    ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv = spline_eval_deriv_alloc.span_view();

    launch_deriv_tests(
            exec_space,
            spline_evaluator,
            coords_eval.span_cview(),
            coef.span_cview(),
            spline_eval_deriv,
            evaluator,
            ncells);
}

} // namespace anonymous_namespace_workaround_3d_spline_evaluator_derivatives_cpp

#if defined(SPLINE_EVALUATOR_3D_DERIV_TEST_1) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Periodic##Uniform1
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_1) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Periodic##NonUniform1
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_2) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Periodic##Uniform2
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_2) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Periodic##NonUniform2
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_3) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Periodic##Uniform3
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_3) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Periodic##NonUniform3
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_4) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Periodic##Uniform4
#elif defined(SPLINE_EVALUATOR_3D_DERIV_TEST_4) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Periodic##NonUniform4
#endif

TEST(SUFFIX(SplineEvaluator3dDerivativesHost), 3DXYZ)
{
    SplineEvaluator3dDerivativesTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>>();
}

TEST(SUFFIX(SplineEvaluator3dDerivativesDevice), 3DXYZ)
{
    SplineEvaluator3dDerivativesTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>>();
}

TEST(SUFFIX(SplineEvaluator3dDerivativesHost), 4DXYZB)
{
    SplineEvaluator3dDerivativesTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimBatch>();
}

TEST(SUFFIX(SplineEvaluator3dDerivativesHost), 4DXBYZ)
{
    SplineEvaluator3dDerivativesTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimBatch,
            DDimGPS<DimY>,
            DDimGPS<DimZ>>();
}

TEST(SUFFIX(SplineEvaluator3dDerivativesHost), 4DXYBZ)
{
    SplineEvaluator3dDerivativesTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimBatch,
            DDimGPS<DimZ>>();
}

TEST(SUFFIX(SplineEvaluator3dDerivativesHost), 4DBXYZ)
{
    SplineEvaluator3dDerivativesTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimBatch,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>>();
}
