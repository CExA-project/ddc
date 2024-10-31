// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <type_traits>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "evaluator_2d.hpp"
#if !defined(BC_PERIODIC)
#    include "polynomial_evaluator.hpp"
#endif

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(EXTRAPOLATION_RULE_CPP) {

#if defined(BC_PERIODIC)
struct DimX
{
    static constexpr bool PERIODIC = false;
};

struct DimY
{
    static constexpr bool PERIODIC = true;
};
#elif defined(BC_GREVILLE)

struct DimX
{
    static constexpr bool PERIODIC = false;
};

struct DimY
{
    static constexpr bool PERIODIC = false;
};
#endif

constexpr std::size_t s_degree = DEGREE;

#if defined(BC_PERIODIC)
constexpr ddc::BoundCond s_bcl1 = ddc::BoundCond::GREVILLE;
constexpr ddc::BoundCond s_bcr1 = ddc::BoundCond::GREVILLE;
constexpr ddc::BoundCond s_bcl2 = ddc::BoundCond::PERIODIC;
constexpr ddc::BoundCond s_bcr2 = ddc::BoundCond::PERIODIC;
#elif defined(BC_GREVILLE)
constexpr ddc::BoundCond s_bcl1 = ddc::BoundCond::GREVILLE;
constexpr ddc::BoundCond s_bcr1 = ddc::BoundCond::GREVILLE;
constexpr ddc::BoundCond s_bcl2 = ddc::BoundCond::GREVILLE;
constexpr ddc::BoundCond s_bcr2 = ddc::BoundCond::GREVILLE;
#endif

template <typename BSpX>
using GrevillePoints1 = ddc::GrevilleInterpolationPoints<BSpX, s_bcl1, s_bcr1>;

template <typename BSpX>
using GrevillePoints2 = ddc::GrevilleInterpolationPoints<BSpX, s_bcl2, s_bcr2>;

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
struct DDimGPS1 : GrevillePoints1<BSplines<X>>::interpolation_discrete_dimension_type
{
};
template <typename X>
struct DDimGPS2 : GrevillePoints2<BSplines<X>>::interpolation_discrete_dimension_type
{
};

#if defined(BC_PERIODIC)
template <typename DDim1, typename DDim2>
using evaluator_type = Evaluator2D::
        Evaluator<CosineEvaluator::Evaluator<DDim1>, CosineEvaluator::Evaluator<DDim2>>;
#else
template <typename DDim1, typename DDim2>
using evaluator_type = Evaluator2D::Evaluator<
        PolynomialEvaluator::Evaluator<DDim1, s_degree>,
        CosineEvaluator::Evaluator<DDim2>>;
#endif

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
}

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <
        typename ExecSpace,
        typename MemorySpace,
        typename DDimI1,
        typename DDimI2,
        typename... DDims>
void ExtrapolationRuleSplineTest()
{
    using I1 = typename DDimI1::continuous_dimension_type;
    using I2 = typename DDimI2::continuous_dimension_type;

    // Instantiate execution spaces and initialize spaces
    ExecSpace const exec_space;
    std::size_t const ncells = 10;
    InterestDimInitializer<DDimI1>(ncells);
    ddc::init_discrete_space<DDimI1>(
            GrevillePoints1<BSplines<I1>>::template get_sampling<DDimI1>());
    InterestDimInitializer<DDimI2>(ncells);
    ddc::init_discrete_space<DDimI2>(
            GrevillePoints2<BSplines<I2>>::template get_sampling<DDimI2>());

    // Create the values domain (mesh)
    ddc::DiscreteDomain<DDimI1, DDimI2> const interpolation_domain(
            GrevillePoints1<BSplines<I1>>::template get_domain<DDimI1>(),
            GrevillePoints2<BSplines<I2>>::template get_domain<DDimI2>());
    // The following line creates a discrete domain over all dimensions (DDims...) except DDimI1 and DDimI2.
    auto const dom_vals_tmp = ddc::remove_dims_of_t<ddc::DiscreteDomain<DDims...>, DDimI1, DDimI2>(
            ddc::DiscreteDomain<DDims>(DElem<DDims>(0), DVect<DDims>(ncells))...);
    ddc::DiscreteDomain<DDims...> const dom_vals(dom_vals_tmp, interpolation_domain);

    // Create a SplineBuilder over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilder2D<
            ExecSpace,
            MemorySpace,
            BSplines<I1>,
            BSplines<I2>,
            DDimI1,
            DDimI2,
            s_bcl1,
            s_bcr1,
            s_bcl2,
            s_bcr2,
            ddc::SplineSolver::GINKGO> const spline_builder(interpolation_domain);

    // Compute useful domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<DDimI1, DDimI2> const dom_interpolation
            = spline_builder.interpolation_domain();
    auto const dom_spline = spline_builder.batched_spline_domain(dom_vals);

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplicated along batch dimensions
    ddc::Chunk vals_1d_host_alloc(dom_interpolation, ddc::HostAllocator<double>());
    ddc::ChunkSpan const vals_1d_host = vals_1d_host_alloc.span_view();
    evaluator_type<DDimI1, DDimI2> const evaluator(dom_interpolation);
    evaluator(vals_1d_host);
    auto vals_1d_alloc = ddc::create_mirror_view_and_copy(exec_space, vals_1d_host);
    ddc::ChunkSpan const vals_1d = vals_1d_alloc.span_view();

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const vals = vals_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            vals.domain(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                vals(e) = vals_1d(DElem<DDimI1, DDimI2>(e));
            });

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
    spline_builder(coef, vals.span_cview());

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
#if defined(ER_NULL)
    using extrapolation_rule_dim_1_type = ddc::NullExtrapolationRule;
    extrapolation_rule_dim_1_type const extrapolation_rule_left_dim_1;
    extrapolation_rule_dim_1_type const extrapolation_rule_right_dim_1;
#    if defined(BC_PERIODIC)
    using extrapolation_rule_dim_2_type = ddc::PeriodicExtrapolationRule<I2>;
    extrapolation_rule_dim_2_type const extrapolation_rule_left_dim_2;
    extrapolation_rule_dim_2_type const extrapolation_rule_right_dim_2;
#    else
    using extrapolation_rule_dim_2_type = ddc::NullExtrapolationRule;
    extrapolation_rule_dim_2_type const extrapolation_rule_left_dim_2;
    extrapolation_rule_dim_2_type const extrapolation_rule_right_dim_2;
#    endif
#elif defined(ER_CONSTANT)
#    if defined(BC_PERIODIC)
    using extrapolation_rule_dim_1_type = ddc::ConstantExtrapolationRule<I1, I2>;
    using extrapolation_rule_dim_2_type = ddc::PeriodicExtrapolationRule<I2>;
    extrapolation_rule_dim_1_type const extrapolation_rule_left_dim_1(x0<I1>());
    extrapolation_rule_dim_1_type const extrapolation_rule_right_dim_1(xN<I1>());
    extrapolation_rule_dim_2_type const extrapolation_rule_left_dim_2;
    extrapolation_rule_dim_2_type const extrapolation_rule_right_dim_2;
#    else
    using extrapolation_rule_dim_1_type = ddc::ConstantExtrapolationRule<I1, I2>;
    using extrapolation_rule_dim_2_type = ddc::ConstantExtrapolationRule<I2, I1>;
    extrapolation_rule_dim_1_type const extrapolation_rule_left_dim_1(x0<I1>(), x0<I2>(), xN<I2>());
    extrapolation_rule_dim_1_type const
            extrapolation_rule_right_dim_1(xN<I1>(), x0<I2>(), xN<I2>());
    extrapolation_rule_dim_2_type const extrapolation_rule_left_dim_2(x0<I2>(), x0<I1>(), xN<I1>());
    extrapolation_rule_dim_2_type const
            extrapolation_rule_right_dim_2(xN<I2>(), x0<I1>(), xN<I1>());
#    endif
#endif

    ddc::SplineEvaluator2D<
            ExecSpace,
            MemorySpace,
            BSplines<I1>,
            BSplines<I2>,
            DDimI1,
            DDimI2,
            extrapolation_rule_dim_1_type,
            extrapolation_rule_dim_1_type,
            extrapolation_rule_dim_2_type,
            extrapolation_rule_dim_2_type> const
            spline_evaluator_batched(
                    extrapolation_rule_left_dim_1,
                    extrapolation_rule_right_dim_1,
                    extrapolation_rule_left_dim_2,
                    extrapolation_rule_right_dim_2);

    // Instantiate chunk of coordinates of dom_interpolation
    ddc::Chunk coords_eval_alloc(dom_vals, ddc::KokkosAllocator<Coord<I1, I2>, MemorySpace>());
    ddc::ChunkSpan const coords_eval = coords_eval_alloc.span_view();
    // Set coords_eval outside of the domain
    // - I1: +1 to ensure left bound is outside domain
    // - I2 this point should be found on the grid in the periodic case
    Coord<I1, I2> const displ(xN<I1>() - x0<I1>() + 1, 2 * xN<I2>() - x0<I2>());
    ddc::parallel_for_each(
            exec_space,
            coords_eval.domain(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                coords_eval(e) = ddc::coordinate(DElem<DDimI1, DDimI2>(e)) + displ;
            });


    // Instantiate chunks to receive outputs of spline_evaluator
    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval = spline_eval_alloc.span_view();

    // Call spline_evaluator on the same mesh we started with
    spline_evaluator_batched(spline_eval, coords_eval.span_cview(), coef.span_cview());

    // Checking errors (we recover the initial values)
    double const max_norm_error = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
#if defined(ER_NULL)
                return Kokkos::abs(spline_eval(e));
#elif defined(ER_CONSTANT)
                typename decltype(ddc::remove_dims_of(
                        vals.domain(),
                        vals.template domain<DDimI1>()))::discrete_element_type const
                        e_without_interest(e);
#    if defined(BC_PERIODIC)
                double const tmp = vals(vals.template domain<DDimI1>().back(), e_without_interest);
#    else
                double tmp;
                if (Coord<I2>(coords_eval(e)) > xN<I2>()) {
                    typename decltype(ddc::remove_dims_of(
                            vals.domain(),
                            vals.template domain<DDimI1, DDimI2>()))::discrete_element_type const
                            e_batch(e);
                    tmp = vals(vals.template domain<DDimI1, DDimI2>().back(), e_batch);
                } else {
                    tmp = vals(vals.template domain<DDimI1>().back(), e_without_interest);
                }
#    endif
                return Kokkos::abs(spline_eval(e) - tmp);
#endif
            });

    double const max_norm = evaluator.max_norm();

    EXPECT_LE(max_norm_error, 1.0e-14 * max_norm);
}

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(EXTRAPOLATION_RULE_CPP)

#if defined(ER_NULL) && defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Null##Periodic##Uniform
#elif defined(ER_NULL) && defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Null##Periodic##NonUniform
#elif defined(ER_NULL) && defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Null##Greville##Uniform
#elif defined(ER_NULL) && defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Null##Greville##NonUniform
#elif defined(ER_CONSTANT) && defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Constant##Periodic##Uniform
#elif defined(ER_CONSTANT) && defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Constant##Periodic##NonUniform
#elif defined(ER_CONSTANT) && defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Constant##Greville##Uniform
#elif defined(ER_CONSTANT) && defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Constant##Greville##NonUniform
#endif

TEST(SUFFIX(ExtrapolationRuleSplineHost), 2DXY)
{
    ExtrapolationRuleSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS1<DimX>,
            DDimGPS2<DimY>,
            DDimGPS1<DimX>,
            DDimGPS2<DimY>>();
}

TEST(SUFFIX(ExtrapolationRuleSplineDevice), 2DXY)
{
    ExtrapolationRuleSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS1<DimX>,
            DDimGPS2<DimY>,
            DDimGPS1<DimX>,
            DDimGPS2<DimY>>();
}
