#include <algorithm>
#include <array>
#include <cmath>
#include <iosfwd>
#include <vector>

#include <experimental/mdspan>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include "ddc/coordinate.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/for_each.hpp"
#include "ddc/uniform_point_sampling.hpp"

#include "cosine_evaluator.hpp"
#include "evaluator_2d.hpp"
#include "polynomial_evaluator.hpp"
#include "spline_error_bounds.hpp"

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

static constexpr std::size_t s_degree = DEGREE;

#if defined(BC_PERIODIC)
static constexpr ddc::BoundCond s_bcl = ddc::BoundCond::PERIODIC;
static constexpr ddc::BoundCond s_bcr = ddc::BoundCond::PERIODIC;
#elif defined(BC_GREVILLE)
static constexpr ddc::BoundCond s_bcl = ddc::BoundCond::GREVILLE;
static constexpr ddc::BoundCond s_bcr = ddc::BoundCond::GREVILLE;
#elif defined(BC_HERMITE)
static constexpr ddc::BoundCond s_bcl = ddc::BoundCond::HERMITE;
static constexpr ddc::BoundCond s_bcr = ddc::BoundCond::HERMITE;
#endif

template <typename BSpX>
using GrevillePoints = ddc::GrevilleInterpolationPoints<BSpX, s_bcl, s_bcr>;

#if defined(BSPLINES_TYPE_UNIFORM)
template <typename X>
using BSplines = ddc::UniformBSplines<X, s_degree>;

// Gives discrete dimension. In the dimension of interest, it is deduced from the BSplines type. In the other dimensions, it has to be newly defined. In practice both types coincide in the test, but it may not be the case.
template <typename X, typename I1, typename I2>
using IDim = std::conditional_t<
        std::disjunction_v<std::is_same<X, I1>, std::is_same<X, I2>>,
        typename GrevillePoints<BSplines<X>>::interpolation_mesh_type,
        ddc::UniformPointSampling<X>>;

#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
using BSplines = ddc::NonUniformBSplines<X, s_degree>;

template <typename X, typename I1, typename I2>
using IDim = std::conditional_t<
        std::disjunction_v<std::is_same<X, I1>, std::is_same<X, I2>>,
        typename GrevillePoints<BSplines<X>>::interpolation_mesh_type,
        ddc::NonUniformPointSampling<X>>;
#endif

#if defined(BC_HERMITE)
template <typename I>
using IDimDeriv = ddc::UniformPointSampling<ddc::Deriv<I>>;
#endif

template <typename IDim1, typename IDim2>
using evaluator_type = Evaluator2D::Evaluator<
        PolynomialEvaluator::Evaluator<IDim1, s_degree>,
        PolynomialEvaluator::Evaluator<IDim2, s_degree>>;

template <typename... IDimX>
using Index = ddc::DiscreteElement<IDimX...>;
template <typename... IDimX>
using DVect = ddc::DiscreteVector<IDimX...>;
template <typename... X>
using Coord = ddc::Coordinate<X...>;

// Extract batch dimensions from IDim (remove dimension of interest). Usefull
template <typename I1, typename I2, typename... X>
using BatchDims = ddc::type_seq_remove_t<ddc::detail::TypeSeq<X...>, ddc::detail::TypeSeq<I1, I2>>;

// Templated function giving first coordinate of the mesh in given dimension.
template <typename X>
static constexpr Coord<X> x0()
{
    return Coord<X>(0.);
}

// Templated function giving last coordinate of the mesh in given dimension.
template <typename X>
static constexpr Coord<X> xN()
{
    return Coord<X>(1.);
}

// Templated function giving step of the mesh in given dimension.
template <typename X>
static constexpr double dx(double ncells)
{
    return (xN<X>() - x0<X>()) / ncells;
}

// Templated function giving break points of mesh in given dimension for non-uniform case.
template <typename X>
static constexpr std::vector<Coord<X>> breaks(double ncells)
{
    std::vector<Coord<X>> out(ncells + 1);
    for (int i(0); i < ncells + 1; ++i) {
        out[i] = x0<X>() + i * dx<X>(ncells);
    }
    return out;
}

// Helper to initialize space
template <class IDimI1, class IDimI2, class T>
struct DimsInitializer;

template <class IDimI1, class IDimI2, class... IDimX>
struct DimsInitializer<IDimI1, IDimI2, ddc::detail::TypeSeq<IDimX...>>
{
    void operator()(std::size_t const ncells)
    {
#if defined(BSPLINES_TYPE_UNIFORM)
        (ddc::init_discrete_space(IDimX::
                                          init(x0<typename IDimX::continuous_dimension_type>(),
                                               xN<typename IDimX::continuous_dimension_type>(),
                                               DVect<IDimX>(ncells))),
         ...);
        ddc::init_discrete_space<BSplines<typename IDimI1::continuous_dimension_type>>(
                x0<typename IDimI1::continuous_dimension_type>(),
                xN<typename IDimI1::continuous_dimension_type>(),
                ncells);
        ddc::init_discrete_space<BSplines<typename IDimI2::continuous_dimension_type>>(
                x0<typename IDimI2::continuous_dimension_type>(),
                xN<typename IDimI2::continuous_dimension_type>(),
                ncells);
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
        (ddc::init_discrete_space<IDimX>(breaks<typename IDimX::continuous_dimension_type>(ncells)),
         ...);
        ddc::init_discrete_space<BSplines<typename IDimI1::continuous_dimension_type>>(
                breaks<typename IDimI1::continuous_dimension_type>(ncells));
        ddc::init_discrete_space<BSplines<typename IDimI2::continuous_dimension_type>>(
                breaks<typename IDimI2::continuous_dimension_type>(ncells));
#endif
        ddc::init_discrete_space<IDimI1>(
                GrevillePoints<
                        BSplines<typename IDimI1::continuous_dimension_type>>::get_sampling());
        ddc::init_discrete_space<IDimI2>(
                GrevillePoints<
                        BSplines<typename IDimI2::continuous_dimension_type>>::get_sampling());

#if defined(BC_HERMITE)
        ddc::init_discrete_space(
                IDimDeriv<typename IDimI1::continuous_dimension_type>::
                        init(Coord<ddc::Deriv<typename IDimI1::continuous_dimension_type>>(1),
                             Coord<ddc::Deriv<typename IDimI1::continuous_dimension_type>>(
                                     std::max(2, (int)s_degree / 2)),
                             DVect<IDimDeriv<typename IDimI1::continuous_dimension_type>>(
                                     std::max(2, (int)s_degree / 2))));
        ddc::init_discrete_space(
                IDimDeriv<typename IDimI2::continuous_dimension_type>::
                        init(Coord<ddc::Deriv<typename IDimI2::continuous_dimension_type>>(1),
                             Coord<ddc::Deriv<typename IDimI2::continuous_dimension_type>>(
                                     std::max(2, (int)s_degree / 2)),
                             DVect<IDimDeriv<typename IDimI2::continuous_dimension_type>>(
                                     std::max(2, (int)s_degree / 2))));

#endif
    }
};

// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename I1, typename I2, typename... X>
static void Batched2dSplineTest()
{
    // Instantiate execution spaces and initialize spaces
    Kokkos::DefaultHostExecutionSpace host_exec_space = Kokkos::DefaultHostExecutionSpace();
    ExecSpace exec_space = ExecSpace();
    std::size_t constexpr ncells = 10;
    DimsInitializer<
            IDim<I1, I1, I2>,
            IDim<I2, I1, I2>,
            BatchDims<IDim<I1, I1, I2>, IDim<I2, I1, I2>, IDim<X, I1, I2>...>>
            dims_initializer;
    dims_initializer(ncells);

    // Create the values domain (mesh)
    auto interpolation_domain1
            = ddc::DiscreteDomain<IDim<I1, I1, I2>>(GrevillePoints<BSplines<I1>>::get_domain());
    auto interpolation_domain2
            = ddc::DiscreteDomain<IDim<I2, I1, I2>>(GrevillePoints<BSplines<I2>>::get_domain());
    auto interpolation_domain = ddc::DiscreteDomain<IDim<I1, I1, I2>, IDim<I2, I1, I2>>(
            GrevillePoints<BSplines<I1>>::get_domain(),
            GrevillePoints<BSplines<I2>>::get_domain());
    ddc::DiscreteDomain<IDim<X, void, void>...> const dom_vals_tmp = ddc::DiscreteDomain<
            IDim<X, void, void>...>(
            ddc::DiscreteDomain<IDim<
                    X,
                    void,
                    void>>(Index<IDim<X, void, void>>(0), DVect<IDim<X, void, void>>(ncells))...);
    ddc::DiscreteDomain<IDim<X, I1, I2>...> const dom_vals
            = ddc::replace_dim_of<IDim<I1, void, void>, IDim<I1, I1, I2>>(
                    ddc::replace_dim_of<
                            IDim<I2, void, void>,
                            IDim<I2, I1, I2>>(dom_vals_tmp, interpolation_domain),
                    interpolation_domain);

#if defined(BC_HERMITE)
    // Create the derivs domain
    ddc::DiscreteDomain<IDimDeriv<I1>> const derivs_domain1 = ddc::DiscreteDomain<
            IDimDeriv<I1>>(Index<IDimDeriv<I1>>(0), DVect<IDimDeriv<I1>>(s_degree / 2));
    ddc::DiscreteDomain<IDimDeriv<I2>> const derivs_domain2 = ddc::DiscreteDomain<
            IDimDeriv<I2>>(Index<IDimDeriv<I2>>(0), DVect<IDimDeriv<I2>>(s_degree / 2));
    ddc::DiscreteDomain<IDimDeriv<I1>, IDimDeriv<I2>> const derivs_domain
            = ddc::DiscreteDomain<IDimDeriv<I1>, IDimDeriv<I2>>(derivs_domain1, derivs_domain2);

    auto const dom_derivs1
            = ddc::replace_dim_of<IDim<I1, I1, I2>, IDimDeriv<I1>>(dom_vals, derivs_domain1);
    auto const dom_derivs2
            = ddc::replace_dim_of<IDim<I2, I1, I2>, IDimDeriv<I2>>(dom_vals, derivs_domain2);
    auto const dom_mixed_derivs
            = ddc::replace_dim_of<IDim<I2, I1, I2>, IDimDeriv<I2>>(dom_derivs1, derivs_domain2);
#endif

    // Create a SplineBuilderBatched over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilder2DBatched<
            ExecSpace,
            MemorySpace,
            BSplines<I1>,
            BSplines<I2>,
            IDim<I1, I1, I2>,
            IDim<I2, I1, I2>,
            s_bcl,
            s_bcr,
            s_bcl,
            s_bcr,
            IDim<X, I1, I2>...>
            spline_builder(dom_vals);

    // Compute usefull domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<IDim<I1, I1, I2>, IDim<I2, I1, I2>> const dom_interpolation
            = spline_builder.interpolation_domain();
    auto const dom_batch = spline_builder.batch_domain();
    auto const dom_spline = spline_builder.spline_domain();

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplicated along batch dimensions
    ddc::Chunk vals1_cpu_alloc(
            dom_interpolation,
            ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan vals1_cpu = vals1_cpu_alloc.span_view();
    evaluator_type<IDim<I1, I1, I2>, IDim<I2, I1, I2>> evaluator(dom_interpolation);
    evaluator(vals1_cpu);
    ddc::Chunk vals1_alloc(dom_interpolation, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan vals1 = vals1_alloc.span_view();
    ddc::deepcopy(vals1, vals1_cpu);

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan vals = vals_alloc.span_view();
    ddc::for_each(
            ddc::policies::policy(exec_space),
            vals.domain(),
            DDC_LAMBDA(Index<IDim<X, I1, I2>...> const e) {
                vals(e) = vals1(ddc::select<IDim<I1, I1, I2>, IDim<I2, I1, I2>>(e));
            });

#if defined(BC_HERMITE)
    // Allocate and fill a chunk containing derivs to be passed as input to spline_builder.
    int constexpr shift = s_degree % 2; // shift = 0 for even order, 1 for odd order
    ddc::Chunk Sderiv1_lhs_alloc(dom_derivs1, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan Sderiv1_lhs = Sderiv1_lhs_alloc.span_view();
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk Sderiv1_lhs1_cpu_alloc(
                ddc::DiscreteDomain<
                        IDimDeriv<I1>,
                        IDim<I2, I1, I2>>(derivs_domain1, interpolation_domain2),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan Sderiv1_lhs1_cpu = Sderiv1_lhs1_cpu_alloc.span_view();
        for (int ii = 0; ii < Sderiv1_lhs1_cpu.domain().template extent<IDimDeriv<I1>>(); ++ii) {
            for (std::size_t jj = 0;
                 jj < Sderiv1_lhs1_cpu.domain().template extent<IDim<I2, I1, I2>>();
                 ++jj) {
                Sderiv1_lhs1_cpu(
                        typename decltype(Sderiv1_lhs1_cpu.domain())::discrete_element_type(ii, jj))
                        = evaluator.deriv(x0<I1>(), x0<I2>(), ii + shift, 0);
            }
        }
        ddc::Chunk Sderiv1_lhs1_alloc(
                ddc::DiscreteDomain<
                        IDimDeriv<I1>,
                        IDim<I2, I1, I2>>(derivs_domain1, interpolation_domain2),
                ddc::KokkosAllocator<double, MemorySpace>());
        ddc::ChunkSpan Sderiv1_lhs1 = Sderiv1_lhs1_alloc.span_view();
        ddc::deepcopy(Sderiv1_lhs1, Sderiv1_lhs1_cpu);

        ddc::for_each(
                ddc::policies::policy(exec_space),
                Sderiv1_lhs.domain(),
                DDC_LAMBDA(typename decltype(Sderiv1_lhs.domain())::discrete_element_type const e) {
                    Sderiv1_lhs(e) = Sderiv1_lhs1(ddc::select<IDimDeriv<I1>, IDim<I2, I1, I2>>(e));
                });
    }
/*
    ddc::Chunk Sderiv_rhs_alloc(dom_derivs1, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan Sderiv_rhs = Sderiv_rhs_alloc.span_view();
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk Sderiv_rhs1_cpu_alloc(derivs_domain1, ddc::HostAllocator<double>());
        ddc::ChunkSpan Sderiv_rhs1_cpu = Sderiv_rhs1_cpu_alloc.span_view();
        for (int ii = 0; ii < Sderiv_rhs1_cpu.domain().template extent<IDimDeriv<I1>>(); ++ii) {
            Sderiv_rhs1_cpu(typename decltype(Sderiv_rhs1_cpu.domain())::discrete_element_type(ii))
                    = evaluator.deriv(x0<I1>(), ii + shift);
        }
        ddc::Chunk Sderiv_rhs1_alloc(derivs_domain1, ddc::KokkosAllocator<double, MemorySpace>());
        ddc::ChunkSpan Sderiv_rhs1 = Sderiv_rhs1_alloc.span_view();
        ddc::deepcopy(Sderiv_rhs1, Sderiv_rhs1_cpu);

        ddc::for_each(
                ddc::policies::policy(exec_space),
                Sderiv_rhs.domain(),
                DDC_LAMBDA(typename decltype(Sderiv_rhs.domain())::discrete_element_type const e) {
                    Sderiv_rhs(e) = Sderiv_rhs1(ddc::select<IDimDeriv<I1>>(e));
                });
    }
				*/
#endif

    // Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
#if 0
#if defined(BC_HERMITE)
    spline_builder(coef, vals, std::optional(Sderiv_lhs), std::optional(Sderiv_rhs));
#else
    spline_builder(coef, vals);
#endif

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
    ddc::SplineEvaluatorBatched<
            ddc::SplineEvaluator<ExecSpace, MemorySpace, BSplines<I>, IDim<I, I>>,
            IDim<X, I>...>
            spline_evaluator_batched(
                    coef.domain(),
                    ddc::g_null_boundary<BSplines<I>>,
                    ddc::g_null_boundary<BSplines<I>>);

    // Instantiate chunk of coordinates of dom_interpolation
    ddc::Chunk coords_eval_alloc(dom_vals, ddc::KokkosAllocator<Coord<X...>, MemorySpace>());
    ddc::ChunkSpan coords_eval = coords_eval_alloc.span_view();
    ddc::for_each(
            ddc::policies::policy(exec_space),
            coords_eval.domain(),
            DDC_LAMBDA(Index<IDim<X, I>...> const e) { coords_eval(e) = ddc::coordinate(e); });


    // Instantiate chunks to receive outputs of spline_evaluator
    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan spline_eval = spline_eval_alloc.span_view();
    ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan spline_eval_deriv = spline_eval_deriv_alloc.span_view();
    ddc::Chunk spline_eval_integrals_alloc(dom_batch, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan spline_eval_integrals = spline_eval_integrals_alloc.span_view();

    // Call spline_evaluator on the same mesh we started with
    spline_evaluator_batched(spline_eval, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator_batched.deriv(spline_eval_deriv, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator_batched.integrate(spline_eval_integrals, coef.span_cview());

    // Checking errors (we recover the initial values)
    double max_norm_error = ddc::transform_reduce(
            ddc::policies::policy(exec_space),
            spline_eval.domain(),
            0.,
            ddc::reducer::max<double>(),
            DDC_LAMBDA(Index<IDim<X, I>...> const e) {
                return Kokkos::abs(spline_eval(e) - vals(e));
            });

    double max_norm_error_diff = ddc::transform_reduce(
            ddc::policies::policy(exec_space),
            spline_eval_deriv.domain(),
            0.,
            ddc::reducer::max<double>(),
            DDC_LAMBDA(Index<IDim<X, I>...> const e) {
                Coord<I> const x = ddc::coordinate(ddc::select<IDim<I, I>>(e));
                return Kokkos::abs(spline_eval_deriv(e) - evaluator.deriv(x, 1));
            });
    double max_norm_error_integ = ddc::transform_reduce(
            ddc::policies::policy(exec_space),
            spline_eval_integrals.domain(),
            0.,
            ddc::reducer::max<double>(),
            DDC_LAMBDA(typename decltype(spline_builder)::batch_domain_type::
                               discrete_element_type const e) {
                return Kokkos::abs(
                        spline_eval_integrals(e) - evaluator.deriv(xN<I>(), -1)
                        + evaluator.deriv(x0<I>(), -1));
            });

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff = evaluator.max_norm(1);
    double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type<IDim<I, I>>> error_bounds(evaluator);
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(dx<I>(ncells), s_degree), 1.0e-14 * max_norm));
    EXPECT_LE(
            max_norm_error_diff,
            std::
                    max(error_bounds.error_bound_on_deriv(dx<I>(ncells), s_degree),
                        1e-12 * max_norm_diff));
    EXPECT_LE(
            max_norm_error_integ,
            std::
                    max(error_bounds.error_bound_on_int(dx<I>(ncells), s_degree),
                        1.0e-14 * max_norm_int));
#endif
}

#if defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM)
#define SUFFIX(name) name##Periodic##Uniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM)
#define SUFFIX(name) name##Periodic##NonUniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM)
#define SUFFIX(name) name##Greville##Uniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM)
#define SUFFIX(name) name##Greville##NonUniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_UNIFORM)
#define SUFFIX(name) name##Hermite##Uniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_NON_UNIFORM)
#define SUFFIX(name) name##Hermite##NonUniform
#endif

TEST(SUFFIX(Batched2dSplineHost), 2DXY)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimY,
            DimX,
            DimY>();
}

TEST(SUFFIX(Batched2dSplineDevice), 2DXY)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimY,
            DimX,
            DimY>();
}

TEST(SUFFIX(Batched2dSplineHost), 3DXY)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineHost), 3DXZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineHost), 3DYZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineDevice), 3DXY)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineDevice), 3DXZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineDevice), 3DYZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}
#if 0
TEST(SUFFIX(Batched2dSplineDevice), 3DX)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineDevice), 3DY)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(SUFFIX(Batched2dSplineDevice), 3DZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}


TEST(SUFFIX(Batched2dSplineHost), 4DX)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineHost), 4DY)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineHost), 4DZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineHost), 4DT)
{
    Batched2dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimT,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineDevice), 4DX)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineDevice), 4DY)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineDevice), 4DZ)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(SUFFIX(Batched2dSplineDevice), 4DT)
{
    Batched2dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimT,
            DimX,
            DimY,
            DimZ,
            DimT>();
}
#endif
