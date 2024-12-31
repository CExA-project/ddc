// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>
#if defined(BC_HERMITE)
#include <optional>
#endif
#if defined(BSPLINES_TYPE_UNIFORM)
#include <type_traits>
#endif
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "cosine_evaluator.hpp"
#include "spline_error_bounds.hpp"

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(BATCHED_SPLINE_BUILDER_CPP) {

#if defined(BC_PERIODIC)
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

struct DimT
{
    static constexpr bool PERIODIC = true;
};
#else

struct DimX
{
    static constexpr bool PERIODIC = false;
};

struct DimY
{
    static constexpr bool PERIODIC = false;
};

struct DimZ
{
    static constexpr bool PERIODIC = false;
};

struct DimT
{
    static constexpr bool PERIODIC = false;
};
#endif

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

// Gives discrete dimension. In the dimension of interest, it is deduced from the BSplines type. In the other dimensions, it has to be newly defined. In practice both types coincide in the test, but it may not be the case.
template <typename X>
struct DDimGPS : GrevillePoints<BSplines<X>>::interpolation_discrete_dimension_type
{
};
template <typename X>
struct DDimPS : ddc::UniformPointSampling<X>
{
};

template <typename X, typename Y>
using DDim = std::conditional_t<std::is_same_v<X, Y>, DDimGPS<X>, DDimPS<X>>;

#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
struct BSplines : ddc::NonUniformBSplines<X, s_degree_x>
{
};
template <typename X>
struct DDimPS : ddc::NonUniformPointSampling<X>
{
};

template <typename X, typename Y>
using DDim = DDimPS<X>;
#endif

template <typename DDimX>
using evaluator_type = CosineEvaluator::Evaluator<DDimX>;

template <typename... DDimX>
using Index = ddc::DiscreteElement<DDimX...>;
template <typename... DDimX>
using DVect = ddc::DiscreteVector<DDimX...>;
template <typename... X>
using Coord = ddc::Coordinate<X...>;

// Extract batch dimensions from DDim (remove dimension of interest). Usefull
template <typename X, typename... Y>
using BatchDims = ddc::type_seq_remove_t<ddc::detail::TypeSeq<Y...>, ddc::detail::TypeSeq<X>>;

// Templated function giving first coordinate of the mesh in given dimension.
template <typename X>
constexpr Coord<X> x0()
{
    return Coord<X>(0.);
}

// Templated function giving last coordinate of the mesh in given dimension.
template <typename X>
constexpr Coord<X> xN()
{
    return Coord<X>(1.);
}

// Templated function giving step of the mesh in given dimension.
template <typename X>
constexpr double dx(std::size_t ncells)
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
template <class DDimI, class T>
struct DimsInitializer;

template <class DDimI, class... DDimX>
struct DimsInitializer<DDimI, ddc::detail::TypeSeq<DDimX...>>
{
    void operator()(std::size_t const ncells)
    {
#if defined(BSPLINES_TYPE_UNIFORM)
        (ddc::init_discrete_space<DDimX>(DDimX::template init<DDimX>(
                 x0<typename DDimX::continuous_dimension_type>(),
                 xN<typename DDimX::continuous_dimension_type>(),
                 DVect<DDimX>(ncells))),
         ...);
        ddc::init_discrete_space<BSplines<typename DDimI::continuous_dimension_type>>(
                x0<typename DDimI::continuous_dimension_type>(),
                xN<typename DDimI::continuous_dimension_type>(),
                ncells);
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
        (ddc::init_discrete_space<DDimX>(breaks<typename DDimX::continuous_dimension_type>(ncells)),
         ...);
        ddc::init_discrete_space<BSplines<typename DDimI::continuous_dimension_type>>(
                breaks<typename DDimI::continuous_dimension_type>(ncells));
#endif
        ddc::init_discrete_space<DDimI>(
                GrevillePoints<BSplines<typename DDimI::continuous_dimension_type>>::
                        template get_sampling<DDimI>());
    }
};

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename I, typename... X>
void BatchedSplineTest()
{
    // Instantiate execution spaces and initialize spaces
    Kokkos::DefaultHostExecutionSpace const host_exec_space;
    ExecSpace const exec_space;

    std::size_t constexpr ncells = 10;
    DimsInitializer<DDim<I, I>, BatchDims<DDim<I, I>, DDim<X, I>...>> dims_initializer;
    dims_initializer(ncells);

    // Create the values domain (mesh)
    ddc::DiscreteDomain<DDim<I, I>> const interpolation_domain
            = GrevillePoints<BSplines<I>>::template get_domain<DDim<I, I>>();
    // If we remove auto using the constructor syntax, nvcc does not compile
    auto const dom_vals_tmp = ddc::DiscreteDomain<DDim<X, void>...>(
            ddc::DiscreteDomain<
                    DDim<X, void>>(Index<DDim<X, void>>(0), DVect<DDim<X, void>>(ncells))...);
    ddc::DiscreteDomain<DDim<X, I>...> const dom_vals
            = ddc::replace_dim_of<DDim<I, void>, DDim<I, I>>(dom_vals_tmp, interpolation_domain);

#if defined(BC_HERMITE)
    // Create the derivs domain
    ddc::DiscreteDomain<ddc::Deriv<I>> const
            derivs_domain(Index<ddc::Deriv<I>>(1), DVect<ddc::Deriv<I>>(s_degree_x / 2));
    auto const dom_derivs = ddc::replace_dim_of<DDim<I, I>, ddc::Deriv<I>>(dom_vals, derivs_domain);
#endif

    // Create a SplineBuilder over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilder<
            ExecSpace,
            MemorySpace,
            BSplines<I>,
            DDim<I, I>,
            s_bcl,
            s_bcr,
#if defined(SOLVER_LAPACK)
            ddc::SplineSolver::LAPACK,
#elif defined(SOLVER_GINKGO)
            ddc::SplineSolver::GINKGO,
#endif
            DDim<X, I>...> const spline_builder(dom_vals);

    // Compute usefull domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<DDim<I, I>> const dom_interpolation = spline_builder.interpolation_domain();
    auto const dom_batch = spline_builder.batch_domain();
    auto const dom_spline = spline_builder.batched_spline_domain();

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplicated along batch dimensions
    ddc::Chunk vals_1d_host_alloc(
            dom_interpolation,
            ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan const vals_1d_host = vals_1d_host_alloc.span_view();
    evaluator_type<DDim<I, I>> const evaluator(dom_interpolation);
    evaluator(vals_1d_host);
    auto vals_1d_alloc = ddc::create_mirror_view_and_copy(exec_space, vals_1d_host);
    ddc::ChunkSpan const vals_1d = vals_1d_alloc.span_view();

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const vals = vals_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            vals.domain(),
            KOKKOS_LAMBDA(Index<DDim<X, I>...> const e) {
                vals(e) = vals_1d(Index<DDim<I, I>>(e));
            });

#if defined(BC_HERMITE)
    // Allocate and fill a chunk containing derivs to be passed as input to spline_builder.
    int constexpr shift = s_degree_x % 2; // shift = 0 for even order, 1 for odd order
    ddc::Chunk derivs_lhs_alloc(dom_derivs, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_lhs = derivs_lhs_alloc.span_view();
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_lhs1_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_lhs1_host = derivs_lhs1_host_alloc.span_view();
        for (int ii = 1; ii < derivs_lhs1_host.domain().template extent<ddc::Deriv<I>>() + 1;
             ++ii) {
            derivs_lhs1_host(
                    typename decltype(derivs_lhs1_host.domain())::discrete_element_type(ii))
                    = evaluator.deriv(x0<I>(), ii + shift - 1);
        }
        auto derivs_lhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs_lhs1_host);
        ddc::ChunkSpan const derivs_lhs1 = derivs_lhs1_alloc.span_view();
        ddc::parallel_for_each(
                exec_space,
                derivs_lhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs_lhs.domain())::discrete_element_type const e) {
                    derivs_lhs(e) = derivs_lhs1(Index<ddc::Deriv<I>>(e));
                });
    }

    ddc::Chunk derivs_rhs_alloc(dom_derivs, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_rhs = derivs_rhs_alloc.span_view();
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_rhs1_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_rhs1_host = derivs_rhs1_host_alloc.span_view();
        for (int ii = 1; ii < derivs_rhs1_host.domain().template extent<ddc::Deriv<I>>() + 1;
             ++ii) {
            derivs_rhs1_host(
                    typename decltype(derivs_rhs1_host.domain())::discrete_element_type(ii))
                    = evaluator.deriv(xN<I>(), ii + shift - 1);
        }
        auto derivs_rhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs_rhs1_host);
        ddc::ChunkSpan const derivs_rhs1 = derivs_rhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs_rhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs_rhs.domain())::discrete_element_type const e) {
                    derivs_rhs(e) = derivs_rhs1(Index<ddc::Deriv<I>>(e));
                });
    }
#endif

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
#if defined(BC_HERMITE)
    spline_builder(
            coef,
            vals.span_cview(),
            std::optional(derivs_lhs.span_cview()),
            std::optional(derivs_rhs.span_cview()));
#else
    spline_builder(coef, vals.span_cview());
#endif

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
#if defined(BC_PERIODIC)
    ddc::PeriodicExtrapolationRule<I> const extrapolation_rule;
#else
    ddc::NullExtrapolationRule const extrapolation_rule;
#endif
    ddc::SplineEvaluator<
            ExecSpace,
            MemorySpace,
            BSplines<I>,
            DDim<I, I>,
#if defined(BC_PERIODIC)
            ddc::PeriodicExtrapolationRule<I>,
            ddc::PeriodicExtrapolationRule<I>,
#else
            ddc::NullExtrapolationRule,
            ddc::NullExtrapolationRule,
#endif
            DDim<X, I>...> const spline_evaluator_batched(extrapolation_rule, extrapolation_rule);

    // Instantiate chunk of coordinates of dom_interpolation
    ddc::Chunk coords_eval_alloc(dom_vals, ddc::KokkosAllocator<Coord<X...>, MemorySpace>());
    ddc::ChunkSpan const coords_eval = coords_eval_alloc.span_view();
    ddc::parallel_for_each(
            exec_space,
            coords_eval.domain(),
            KOKKOS_LAMBDA(Index<DDim<X, I>...> const e) { coords_eval(e) = ddc::coordinate(e); });


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
            KOKKOS_LAMBDA(Index<DDim<X, I>...> const e) {
                return Kokkos::abs(spline_eval(e) - vals(e));
            });

    double const max_norm_error_diff = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(Index<DDim<X, I>...> const e) {
                Coord<I> const x = ddc::coordinate(Index<DDim<I, I>>(e));
                return Kokkos::abs(spline_eval_deriv(e) - evaluator.deriv(x, 1));
            });
    double const max_norm_error_integ = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_integrals.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(typename decltype(spline_builder)::batch_domain_type::
                                  discrete_element_type const e) {
                return Kokkos::abs(
                        spline_eval_integrals(e) - evaluator.deriv(xN<I>(), -1)
                        + evaluator.deriv(x0<I>(), -1));
            });

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type<DDim<I, I>>> error_bounds(evaluator);
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

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(BATCHED_SPLINE_BUILDER_CPP)

#if defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_LAPACK)
#define SUFFIX(name) name##Lapack##Periodic##Uniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_LAPACK)
#define SUFFIX(name) name##Lapack##Periodic##NonUniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_LAPACK)
#define SUFFIX(name) name##Lapack##Greville##Uniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_LAPACK)
#define SUFFIX(name) name##Lapack##Greville##NonUniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_LAPACK)
#define SUFFIX(name) name##Lapack##Hermite##Uniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_LAPACK)
#define SUFFIX(name) name##Lapack##Hermite##NonUniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_GINKGO)
#define SUFFIX(name) name##Ginkgo##Periodic##Uniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_GINKGO)
#define SUFFIX(name) name##Ginkgo##Periodic##NonUniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_GINKGO)
#define SUFFIX(name) name##Ginkgo##Greville##Uniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_GINKGO)
#define SUFFIX(name) name##Ginkgo##Greville##NonUniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_UNIFORM) && defined(SOLVER_GINKGO)
#define SUFFIX(name) name##Ginkgo##Hermite##Uniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_NON_UNIFORM) && defined(SOLVER_GINKGO)
#define SUFFIX(name) name##Ginkgo##Hermite##NonUniform
#endif

TEST(SUFFIX(BatchedSplineHost), 1DX)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX>();
}

TEST(SUFFIX(BatchedSplineDevice), 1DX)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX>();
}

TEST(SUFFIX(BatchedSplineHost), 2DX)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY>();
}

TEST(SUFFIX(BatchedSplineHost), 2DY)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY>();
}

TEST(SUFFIX(BatchedSplineDevice), 2DX)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY>();
}

TEST(SUFFIX(BatchedSplineDevice), 2DY)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY>();
}

TEST(SUFFIX(BatchedSplineHost), 3DX)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(BatchedSplineHost), 3DY)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(BatchedSplineHost), 3DZ)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(BatchedSplineDevice), 3DX)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(BatchedSplineDevice), 3DY)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(BatchedSplineDevice), 3DZ)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}


TEST(SUFFIX(BatchedSplineHost), 4DX)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineHost), 4DY)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineHost), 4DZ)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineHost), 4DT)
{
    BatchedSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimT,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineDevice), 4DX)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineDevice), 4DY)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineDevice), 4DZ)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(BatchedSplineDevice), 4DT)
{
    BatchedSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimT,
            DimX,
            DimY,
            DimZ,
            DimT>();
}
