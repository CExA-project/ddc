// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#if defined(BC_HERMITE)
#    include <optional>
#endif
#if defined(BSPLINES_TYPE_UNIFORM)
#    include <type_traits>
#endif
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "spline_error_bounds.hpp"

inline namespace anonymous_namespace_workaround_batched_spline_builder_cpp {

#if defined(BC_PERIODIC)
struct DimX
{
    static constexpr bool PERIODIC = true;
};
#else

struct DimX
{
    static constexpr bool PERIODIC = false;
};
#endif

struct DDimBatch1
{
};

struct DDimBatch2
{
};

constexpr std::size_t s_degree_x = DEGREE_X;

#if defined(BC_PERIODIC)
constexpr ddc::BoundCond s_bcl = ddc::BoundCond::PERIODIC;
constexpr ddc::BoundCond s_bcr = ddc::BoundCond::PERIODIC;
#elif defined(BC_GREVILLE)
constexpr ddc::BoundCond s_bcl = ddc::BoundCond::GREVILLE;
constexpr ddc::BoundCond s_bcr = ddc::BoundCond::GREVILLE;
#elif defined(BC_HERMITE)
constexpr ddc::BoundCond s_bcl = ddc::BoundCond::HERMITE;
constexpr ddc::BoundCond s_bcr = ddc::BoundCond::HERMITE;
#endif

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

#if defined(SOLVER_LAPACK)
constexpr ddc::SplineSolver s_spline_solver = ddc::SplineSolver::LAPACK;
#elif defined(SOLVER_GINKGO)
constexpr ddc::SplineSolver s_spline_solver = ddc::SplineSolver::GINKGO;
#endif

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

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename DDimI, typename... DDims>
void BatchedSplineTest()
{
    using I = typename DDimI::continuous_dimension_type;

    // Instantiate execution spaces and initialize spaces
    ExecSpace const exec_space;

    std::size_t const ncells = 10;
    InterestDimInitializer<DDimI>(ncells);

    // Create the values domain (mesh)
    ddc::DiscreteDomain<DDimI> const interpolation_domain
            = GrevillePoints<BSplines<I>>::template get_domain<DDimI>();
    // The following line creates a discrete domain over all dimensions (DDims...) except DDimI.
    auto const dom_vals_tmp = ddc::remove_dims_of_t<ddc::DiscreteDomain<DDims...>, DDimI>(
            ddc::DiscreteDomain<DDims>(DElem<DDims>(0), DVect<DDims>(ncells))...);
    ddc::DiscreteDomain<DDims...> const dom_vals(dom_vals_tmp, interpolation_domain);

    // Create the derivs domain
    ddc::StridedDiscreteDomain<ddc::Deriv<I>> const derivs_domain(
            DElem<ddc::Deriv<I>>(1),
            DVect<ddc::Deriv<I>>(s_degree_x / 2),
            DVect<ddc::Deriv<I>>(1));
    ddc::StridedDiscreteDomain<ddc::Deriv<I>, DDims...>
            whole_derivs_domain(derivs_domain, ddc::detail::to_strided_ddom(dom_vals));
    auto const dom_derivs = ddc::replace_dim_of<
            DDimI,
            ddc::Deriv<I>>(ddc::detail::to_strided_ddom(dom_vals), derivs_domain);

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
    auto const dom_batch = spline_builder.batch_domain(dom_vals);
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

    // Allocate and fill a chunk containing derivs to be passed as input to spline_builder.
    int const shift = s_degree_x % 2; // shift = 0 for even order, 1 for odd order
    ddc::Chunk derivs_alloc(whole_derivs_domain, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs = derivs_alloc.span_view();

    ddc::ChunkSpan const derivs_lhs_view
            = derivs[ddc::DiscreteElement<DDimI>(derivs.domain().front())];
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_lhs_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_lhs_host = derivs_lhs_host_alloc.span_view();
        for (int ii = 1; ii < derivs_lhs_host.domain().template extent<ddc::Deriv<I>>() + 1; ++ii) {
            derivs_lhs_host(typename decltype(derivs_lhs_host.domain())::discrete_element_type(ii))
                    = evaluator.deriv(x0<I>(), ii + shift - 1);
        }
        auto derivs_lhs_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs_lhs_host);
        ddc::ChunkSpan const derivs_lhs = derivs_lhs_alloc.span_view();
        ddc::parallel_for_each(
                exec_space,
                derivs_lhs_view.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs_lhs_view.domain())::discrete_element_type const
                                e) { derivs_lhs_view(e) = derivs_lhs(DElem<ddc::Deriv<I>>(e)); });
    }

    ddc::ChunkSpan const derivs_rhs_view
            = derivs[ddc::DiscreteElement<DDimI>(derivs.domain().back())];
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_rhs_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_rhs_host = derivs_rhs_host_alloc.span_view();
        for (int ii = 1; ii < derivs_rhs_host.domain().template extent<ddc::Deriv<I>>() + 1; ++ii) {
            derivs_rhs_host(typename decltype(derivs_rhs_host.domain())::discrete_element_type(ii))
                    = evaluator.deriv(xN<I>(), ii + shift - 1);
        }
        auto derivs_rhs_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs_rhs_host);
        ddc::ChunkSpan const derivs_rhs = derivs_rhs_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs_rhs_view.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs_rhs_view.domain())::discrete_element_type const
                                e) { derivs_rhs_view(e) = derivs_rhs(DElem<ddc::Deriv<I>>(e)); });
    }

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
    spline_builder(coef, vals.span_cview(), derivs.span_cview());

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
#if defined(BC_PERIODIC)
    using extrapolation_rule_type = ddc::PeriodicExtrapolationRule<I>;
#else
    using extrapolation_rule_type = ddc::NullExtrapolationRule;
#endif
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
    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval = spline_eval_alloc.span_view();
    ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv = spline_eval_deriv_alloc.span_view();
    ddc::Chunk spline_eval_integrals_alloc(dom_batch, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_integrals = spline_eval_integrals_alloc.span_view();

    // Call spline_evaluator on the same mesh we started with
    spline_evaluator_batched(spline_eval, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator_batched.deriv(spline_eval_deriv, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator_batched.integrate(spline_eval_integrals, coef.span_cview());

    // Checking errors (we recover the initial values)
    double const max_norm_error = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                return Kokkos::abs(spline_eval(e) - vals(e));
            });

    double const max_norm_error_diff = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I> const x = ddc::coordinate(DElem<DDimI>(e));
                return Kokkos::abs(spline_eval_deriv(e) - evaluator.deriv(x, 1));
            });
    double const max_norm_error_integ = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_integrals.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(
                    typename decltype(spline_builder)::template batch_domain_type<
                            ddc::DiscreteDomain<DDims...>>::discrete_element_type const e) {
                return Kokkos::abs(
                        spline_eval_integrals(e) - evaluator.deriv(xN<I>(), -1)
                        + evaluator.deriv(x0<I>(), -1));
            });

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type<DDimI>> const error_bounds(evaluator);
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(dx<I>(ncells), s_degree_x), 1.0e-14 * max_norm));
    EXPECT_LE(
            max_norm_error_diff,
            std::
                    max(error_bounds.error_bound_on_deriv(dx<I>(ncells), s_degree_x),
                        1e-12 * max_norm_diff));
    EXPECT_LE(
            max_norm_error_integ,
            std::
                    max(error_bounds.error_bound_on_int(dx<I>(ncells), s_degree_x),
                        1.0e-14 * max_norm_int));
}

} // namespace anonymous_namespace_workaround_batched_spline_builder_cpp

#if defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_LAPACK)
#    define SUFFIX(name) name##Lapack##Periodic##Uniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_LAPACK)
#    define SUFFIX(name) name##Lapack##Periodic##NonUniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_LAPACK)
#    define SUFFIX(name) name##Lapack##Greville##Uniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_LAPACK)
#    define SUFFIX(name) name##Lapack##Greville##NonUniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_LAPACK)
#    define SUFFIX(name) name##Lapack##Hermite##Uniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_LAPACK)
#    define SUFFIX(name) name##Lapack##Hermite##NonUniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_GINKGO)
#    define SUFFIX(name) name##Ginkgo##Periodic##Uniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_GINKGO)
#    define SUFFIX(name) name##Ginkgo##Periodic##NonUniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_GINKGO)
#    define SUFFIX(name) name##Ginkgo##Greville##Uniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_GINKGO)
#    define SUFFIX(name) name##Ginkgo##Greville##NonUniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_GINKGO)
#    define SUFFIX(name) name##Ginkgo##Hermite##Uniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_GINKGO)
#    define SUFFIX(name) name##Ginkgo##Hermite##NonUniform
#endif

TEST(SUFFIX(BatchedSplineHost), 1DX)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(BatchedSplineDevice), 1DX)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(BatchedSplineHost), 2DXB1)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1>();
}

TEST(SUFFIX(BatchedSplineHost), 2DB1X)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(BatchedSplineDevice), 2DXB1)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1>();
}

TEST(SUFFIX(BatchedSplineDevice), 2DB1X)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(BatchedSplineHost), 3DXB1B2)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2>();
}

TEST(SUFFIX(BatchedSplineHost), 3DB1XB2)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>,
            DDimBatch2>();
}

TEST(SUFFIX(BatchedSplineHost), 3DB1B2X)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2,
            DDimGPS<DimX>>();
}

TEST(SUFFIX(BatchedSplineDevice), 3DXB1B2)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2>();
}

TEST(SUFFIX(BatchedSplineDevice), 3DB1XB2)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimGPS<DimX>,
            DDimBatch2>();
}

TEST(SUFFIX(BatchedSplineDevice), 3DB1B2X)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimBatch1,
            DDimBatch2,
            DDimGPS<DimX>>();
}
