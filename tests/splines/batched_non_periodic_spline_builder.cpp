#include <algorithm>
#include <array>
#include <cmath>
#include <iosfwd>
#include <vector>

#include <experimental/mdspan>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include "ddc/discrete_domain.hpp"

#include "cosine_evaluator.hpp"
#include "polynomial_evaluator.hpp"
#include "spline_error_bounds.hpp"

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

static constexpr std::size_t s_degree_x = DEGREE_X;

#if defined(BC_GREVILLE)
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
using BSplines = ddc::UniformBSplines<X, s_degree_x>;

template <typename X, typename I>
using IDim = std::conditional_t<
        std::is_same_v<X, I>,
        typename GrevillePoints<BSplines<X>>::interpolation_mesh_type,
        ddc::UniformPointSampling<X>>;

#elif defined(BSPLINES_TYPE_NON_UNIFORM)
template <typename X>
using BSplines = ddc::NonUniformBSplines<X, s_degree_x>;

template <typename X, typename I>
using IDim = std::conditional_t<
        std::is_same_v<X, I>,
        typename GrevillePoints<BSplines<X>>::interpolation_mesh_type,
        ddc::NonUniformPointSampling<X>>;
#endif

template <typename DerivI>
using IDimDeriv = ddc::UniformPointSampling<DerivI>;

template <typename IDimX>
using evaluator_type = CosineEvaluator::Evaluator<IDimX>;

template <typename... IDimX>
using Index = ddc::DiscreteElement<IDimX...>;
template <typename... IDimX>
using DVect = ddc::DiscreteVector<IDimX...>;
template <typename... X>
using Coord = ddc::Coordinate<X...>;

template <typename I, typename... X>
using BatchDims = ddc::type_seq_remove_t<ddc::detail::TypeSeq<X...>, ddc::detail::TypeSeq<I>>;

template <typename X>
static constexpr Coord<X> x0()
{
    return Coord<X>(0.);
}

template <typename X>
static constexpr Coord<X> xN()
{
    return Coord<X>(1.);
}

template <typename X>
static constexpr double dx(double ncells)
{
    return (xN<X>() - x0<X>()) / ncells;
}

template <typename X>
static constexpr std::vector<Coord<X>> breaks(double ncells)
{
    std::vector<Coord<X>> out(ncells + 1);
    for (int i(0); i < ncells + 1; ++i) {
        out[i] = x0<X>() + i * dx<X>(ncells);
    }
    return out;
}

template <class IDimI, class T>
struct DimsInitializer;

template <class IDimI, class... IDimX>
struct DimsInitializer<IDimI, ddc::detail::TypeSeq<IDimX...>>
{
    void operator()(std::size_t const ncells)
    {
#if defined(BSPLINES_TYPE_UNIFORM)
        (ddc::init_discrete_space(IDimX::
                                          init(x0<typename IDimX::continuous_dimension_type>(),
                                               xN<typename IDimX::continuous_dimension_type>(),
                                               DVect<IDimX>(ncells))),
         ...);
        ddc::init_discrete_space<BSplines<typename IDimI::continuous_dimension_type>>(
                x0<typename IDimI::continuous_dimension_type>(),
                xN<typename IDimI::continuous_dimension_type>(),
                ncells);
#elif defined(BSPLINES_TYPE_NON_UNIFORM)
        (ddc::init_discrete_space<IDimX>(breaks<typename IDimX::continuous_dimension_type>(ncells)),
         ...);
        ddc::init_discrete_space<BSplines<typename IDimI::continuous_dimension_type>>(
                breaks<typename IDimI::continuous_dimension_type>(ncells));
#endif
        ddc::init_discrete_space<IDimI>(
                GrevillePoints<
                        BSplines<typename IDimI::continuous_dimension_type>>::get_sampling());
        ddc::init_discrete_space(
                IDimDeriv<ddc::Deriv<typename IDimI::continuous_dimension_type>>::
                        init(Coord<ddc::Deriv<typename IDimI::continuous_dimension_type>>(1),
                             Coord<ddc::Deriv<typename IDimI::continuous_dimension_type>>(
                                     s_degree_x),
                             DVect<IDimDeriv<
                                     ddc::Deriv<typename IDimI::continuous_dimension_type>>>(
                                     s_degree_x)));
    }
};


// Checks that when evaluating the spline at interpolation points one
// recovers values that were used to build the spline
template <typename ExecSpace, typename MemorySpace, typename I, typename... X>
static void BatchedNonPeriodicSplineTest()
{
    // Instantiate execution spaces and initialize spaces
    Kokkos::DefaultHostExecutionSpace host_exec_space = Kokkos::DefaultHostExecutionSpace();
    ExecSpace exec_space = ExecSpace();
    std::size_t constexpr ncells = 10;

    // Initialize spaces
    DimsInitializer<IDim<I, I>, BatchDims<IDim<I, I>, IDim<X, I>...>> dims_initializer;
    dims_initializer(ncells);

    // Create the values domain (mesh)
    ddc::DiscreteDomain<IDim<I, I>> interpolation_domain
            = GrevillePoints<BSplines<I>>::get_domain();
    ddc::DiscreteDomain<IDim<X, void>...> const dom_vals_tmp = ddc::DiscreteDomain<
            IDim<X, void>...>(
            ddc::DiscreteDomain<
                    IDim<X, void>>(Index<IDim<X, void>>(0), DVect<IDim<X, void>>(ncells))...);
    ddc::DiscreteDomain<IDim<X, I>...> const dom_vals
            = ddc::replace_dim_of<IDim<I, void>, IDim<I, I>>(dom_vals_tmp, interpolation_domain);

    ddc::DiscreteDomain<IDimDeriv<ddc::Deriv<I>>> const derivs_domain
            = ddc::DiscreteDomain<IDimDeriv<ddc::Deriv<I>>>(
                    Index<IDimDeriv<ddc::Deriv<I>>>(0),
                    DVect<IDimDeriv<ddc::Deriv<I>>>(s_degree_x));

    auto const dom_derivs
            = ddc::replace_dim_of<IDim<I, I>, IDimDeriv<ddc::Deriv<I>>>(dom_vals, derivs_domain);

    // Create a SplineBuilderBatched over BSplines<I> and batched along other dimensions using some boundary conditions
    ddc::SplineBuilderBatched<
            ddc::SplineBuilder<ExecSpace, MemorySpace, BSplines<I>, IDim<I, I>, s_bcl, s_bcr>,
            IDim<X, I>...>
            spline_builder(dom_vals);

    // Compute usefull domains (dom_interpolation, dom_batch, dom_bsplines and dom_spline)
    ddc::DiscreteDomain<IDim<I, I>> const dom_interpolation = spline_builder.interpolation_domain();
    auto const dom_batch = spline_builder.batch_domain();
    ddc::DiscreteDomain<BSplines<I>> const dom_bsplines = spline_builder.bsplines_domain();
    auto const dom_spline = spline_builder.spline_domain();

    // Allocate and fill a chunk containing values to be passed as input to spline_builder. Those are values of cosine along interest dimension duplic  ated along batch dimensions
    ddc::Chunk vals1_cpu_alloc(
            dom_interpolation,
            ddc::KokkosAllocator<double, Kokkos::DefaultHostExecutionSpace::memory_space>());
    ddc::ChunkSpan vals1_cpu = vals1_cpu_alloc.span_view();
    evaluator_type<IDim<I, I>> evaluator(dom_interpolation);
    evaluator(vals1_cpu);
    ddc::Chunk vals1_alloc(dom_interpolation, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan vals1 = vals1_alloc.span_view();
    ddc::deepcopy(vals1, vals1_cpu);

    ddc::Chunk vals_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan vals = vals_alloc.span_view();
    ddc::for_each(
            ddc::policies::policy(exec_space),
            vals.domain(),
            DDC_LAMBDA(Index<IDim<X, I>...> const e) {
                vals(e) = vals1(ddc::select<IDim<I, I>>(e));
            });
 
    int constexpr shift = s_degree_x % 2; // shift = 0 for even order, 1 for odd order
    ddc::Chunk Sderiv_lhs_alloc(dom_derivs, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan Sderiv_lhs = Sderiv_lhs_alloc.span_view();
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk Sderiv_lhs1_cpu_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan Sderiv_lhs1_cpu = Sderiv_lhs1_cpu_alloc.span_view();
        for (int ii = 0; ii < Sderiv_lhs1_cpu.domain().template extent<IDimDeriv<ddc::Deriv<I>>>();
             ++ii) {
            Sderiv_lhs1_cpu(typename decltype(Sderiv_lhs1_cpu.domain())::discrete_element_type(ii))
                    = evaluator.deriv(x0<I>(), ii + shift);
        }
        ddc::Chunk Sderiv_lhs1_alloc(derivs_domain, ddc::KokkosAllocator<double, MemorySpace>());
        ddc::ChunkSpan Sderiv_lhs1 = Sderiv_lhs1_alloc.span_view();
        ddc::deepcopy(Sderiv_lhs1, Sderiv_lhs1_cpu);

        ddc::for_each(
                ddc::policies::policy(exec_space),
                Sderiv_lhs.domain(),
                DDC_LAMBDA(typename decltype(Sderiv_lhs.domain())::discrete_element_type const e) {
                    Sderiv_lhs(e) = Sderiv_lhs1(ddc::select<IDimDeriv<ddc::Deriv<I>>>(e));
                });
    }

    ddc::Chunk Sderiv_rhs_alloc(dom_derivs, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan Sderiv_rhs = Sderiv_rhs_alloc.span_view();
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk Sderiv_rhs1_cpu_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan Sderiv_rhs1_cpu = Sderiv_rhs1_cpu_alloc.span_view();
        for (int ii = 0; ii < Sderiv_rhs1_cpu.domain().template extent<IDimDeriv<ddc::Deriv<I>>>();
             ++ii) {
            Sderiv_rhs1_cpu(typename decltype(Sderiv_rhs1_cpu.domain())::discrete_element_type(ii))
                    = evaluator.deriv(x0<I>(), ii + shift);
        }
        ddc::Chunk Sderiv_rhs1_alloc(derivs_domain, ddc::KokkosAllocator<double, MemorySpace>());
        ddc::ChunkSpan Sderiv_rhs1 = Sderiv_rhs1_alloc.span_view();
        ddc::deepcopy(Sderiv_rhs1, Sderiv_rhs1_cpu);

        ddc::for_each(
                ddc::policies::policy(exec_space),
                Sderiv_rhs.domain(),
                DDC_LAMBDA(typename decltype(Sderiv_rhs.domain())::discrete_element_type const e) {
                    Sderiv_rhs(e) = Sderiv_rhs1(ddc::select<IDimDeriv<ddc::Deriv<I>>>(e));
                });
    }

	// Instantiate chunk of spline coefs to receive output of spline_builder
    ddc::Chunk coef_alloc(dom_spline, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan coef = coef_alloc.span_view();

    // Finally compute the spline by filling `coef`
    spline_builder(coef, vals, std::optional(Sderiv_lhs), std::optional(Sderiv_rhs));

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
	/*
    ddc::Chunk spline_eval_deriv_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan spline_eval_deriv = spline_eval_deriv_alloc.span_view();
    ddc::Chunk spline_eval_integrals_alloc(dom_batch, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan spline_eval_integrals = spline_eval_integrals_alloc.span_view();
	*/

	// Call spline_evaluator on the same mesh we started with
    spline_evaluator_batched(spline_eval, coords_eval.span_cview(), coef.span_cview());
    // spline_evaluator_batched.deriv(spline_eval_deriv, coords_eval.span_cview(), coef.span_cview());
    // spline_evaluator_batched.integrate(spline_eval_integrals, coef.span_cview());

	// Checking errors (we recover the initial values)
    double max_norm_error = ddc::transform_reduce(
            ddc::policies::policy(exec_space),
            spline_eval.domain(),
            0.,
            ddc::reducer::max<double>(),
            DDC_LAMBDA(Index<IDim<X, I>...> const e) {
                return Kokkos::abs(spline_eval(e) - vals(e));
            });
	/*
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
	*/
    double const max_norm = evaluator.max_norm();
    // double const max_norm_diff = evaluator.max_norm(1);
    // double const max_norm_int = evaluator.max_norm(-1);

    SplineErrorBounds<evaluator_type<IDim<I, I>>> error_bounds(evaluator);
    EXPECT_LE(
            max_norm_error,
            std::max(error_bounds.error_bound(dx<I>(ncells), s_degree_x), 1.0e-14 * max_norm));
	/*
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
	*/
    /*
    ddc::DiscreteDomain<BSplinesX> const dom_bsplines_x(
            ddc::discrete_space<BSplinesX>().full_domain());

    // 2. Create a Spline represented by a chunk over BSplines
    // The chunk is filled with garbage data, we need to initialize it
	ddc::Chunk coef(dom_bsplines_x, ddc::KokkosAllocator<double, Kokkos::HostSpace>());

    // 3. Create the interpolation domain
    ddc::init_discrete_space<IDimX>(GrevillePoints::get_sampling());
    ddc::DiscreteDomain<IDimX> interpolation_domain(GrevillePoints::get_domain());

    // 4. Create a SplineBuilder over BSplines using some boundary conditions
    ddc::SplineBuilder<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::HostSpace,
            BSplinesX,
            IDimX,
            s_bcl,
            s_bcr>
            spline_builder(interpolation_domain);

    // 5. Allocate and fill a chunk over the interpolation domain
	ddc::Chunk yvals(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::HostSpace>());
    evaluator_type evaluator(interpolation_domain);
    evaluator(yvals.span_view());

    int constexpr shift = s_degree_x % 2; // shift = 0 for even order, 1 for odd order
    std::array<double, s_degree_x / 2> Sderiv_lhs_data;
	ddc::DSpan1D Sderiv_lhs(Sderiv_lhs_data.data(), Sderiv_lhs_data.size());
    std::optional<ddc::DSpan1D> deriv_l;
    if (s_bcl == ddc::BoundCond::HERMITE) {
        for (std::size_t ii = 0; ii < Sderiv_lhs.extent(0); ++ii) {
            Sderiv_lhs(ii) = evaluator.deriv(x0, ii + shift);
        }
        deriv_l = Sderiv_lhs;
    }

    std::array<double, s_degree_x / 2> Sderiv_rhs_data;
	ddc::DSpan1D Sderiv_rhs(Sderiv_rhs_data.data(), Sderiv_rhs_data.size());
    std::optional<ddc::DSpan1D> deriv_r;
    if (s_bcr == ddc::BoundCond::HERMITE) {
        for (std::size_t ii = 0; ii < Sderiv_rhs.extent(0); ++ii) {
            Sderiv_rhs(ii) = evaluator.deriv(xN, ii + shift);
        }
        deriv_r = Sderiv_rhs;
    }

    // 6. Finally build the spline by filling `coef`
    spline_builder(coef.span_view(), yvals.span_view(), deriv_l, deriv_r);

    // 7. Create a SplineEvaluator to evaluate the spline at any point in the domain of the BSplines
    ddc::SplineEvaluator<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace, BSplinesX, IDimX>
            spline_evaluator(ddc::g_null_boundary<BSplinesX>, ddc::g_null_boundary<BSplinesX>);

    ddc::Chunk<ddc::Coordinate<DimX>, ddc::DiscreteDomain<IDimX>> coords_eval(interpolation_domain);
    for (IndexX const ix : interpolation_domain) {
        coords_eval(ix) = ddc::coordinate(ix);
    }

	ddc::Chunk spline_eval(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::HostSpace>());
    spline_evaluator(spline_eval.span_view(), coords_eval.span_cview(), coef.span_cview());

	ddc::Chunk spline_eval_deriv(interpolation_domain, ddc::KokkosAllocator<double, Kokkos::HostSpace>());
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
    if constexpr (std::is_same_v<
                          evaluator_type,
                          PolynomialEvaluator::Evaluator<IDimX, s_degree_x>>) {
        EXPECT_LE(max_norm_error / max_norm, 1.0e-14);
        EXPECT_LE(max_norm_error_diff / max_norm_diff, 1.0e-12);
        EXPECT_LE(max_norm_error_integ / max_norm_int, 1.0e-14);
    } else {
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
    }
	*/
}

TEST(BatchedNonPeriodicSplineHost, 1DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX>();
}

TEST(BatchedNonPeriodicSplineDevice, 1DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX>();
}

TEST(BatchedNonPeriodicSplineHost, 2DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY>();
}

TEST(BatchedNonPeriodicSplineHost, 2DY)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY>();
}

TEST(BatchedNonPeriodicSplineDevice, 2DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY>();
}

TEST(BatchedNonPeriodicSplineDevice, 2DY)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY>();
}

TEST(BatchedNonPeriodicSplineHost, 3DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ>();
}

TEST(BatchedNonPeriodicSplineHost, 3DY)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(BatchedNonPeriodicSplineHost, 3DZ)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}

TEST(BatchedNonPeriodicSplineDevice, 3DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ>();
}

TEST(BatchedNonPeriodicSplineDevice, 3DY)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ>();
}

TEST(BatchedNonPeriodicSplineDevice, 3DZ)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ>();
}


TEST(BatchedNonPeriodicSplineHost, 4DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineHost, 4DY)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineHost, 4DZ)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineHost, 4DT)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DimT,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineDevice, 4DX)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimX,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineDevice, 4DY)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimY,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineDevice, 4DZ)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimZ,
            DimX,
            DimY,
            DimZ,
            DimT>();
}

TEST(BatchedNonPeriodicSplineDevice, 4DT)
{
    BatchedNonPeriodicSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DimT,
            DimX,
            DimY,
            DimZ,
            DimT>();
}
