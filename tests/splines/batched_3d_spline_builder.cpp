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

#include "evaluator_3d.hpp"
#if defined(BC_PERIODIC)
#    include "cosine_evaluator.hpp"
#else
#    include "polynomial_evaluator.hpp"
#endif
#include "spline_error_bounds.hpp"

inline namespace anonymous_namespace_workaround_batched_3d_spline_builder_cpp {

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
#endif

struct DDimBatch
{
};

constexpr std::size_t s_degree = DEGREE;

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

#if defined(BC_PERIODIC)
template <typename DDim1, typename DDim2, typename DDim3>
using evaluator_type = Evaluator3D::Evaluator<
        CosineEvaluator::Evaluator<DDim1>,
        CosineEvaluator::Evaluator<DDim2>,
        CosineEvaluator::Evaluator<DDim3>>;
#else
template <typename DDim1, typename DDim2, typename DDim3>
using evaluator_type = Evaluator3D::Evaluator<
        PolynomialEvaluator::Evaluator<DDim1, s_degree>,
        PolynomialEvaluator::Evaluator<DDim2, s_degree>,
        PolynomialEvaluator::Evaluator<DDim3, s_degree>>;
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
    ddc::init_discrete_space<DDim>(GrevillePoints<BSplines<CDim>>::template get_sampling<DDim>());
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
void Batched3dSplineTest()
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

#if defined(BC_HERMITE)
    // Create the derivs domain
    ddc::DiscreteDomain<ddc::Deriv<I1>> const
            derivs_domain1(DElem<ddc::Deriv<I1>>(1), DVect<ddc::Deriv<I1>>(s_degree / 2));
    ddc::DiscreteDomain<ddc::Deriv<I2>> const
            derivs_domain2(DElem<ddc::Deriv<I2>>(1), DVect<ddc::Deriv<I2>>(s_degree / 2));
    ddc::DiscreteDomain<ddc::Deriv<I3>> const
            derivs_domain3(DElem<ddc::Deriv<I3>>(1), DVect<ddc::Deriv<I3>>(s_degree / 2));
    ddc::DiscreteDomain<ddc::Deriv<I1>, ddc::Deriv<I2>, DDimI3> const
            derivs_domain12(derivs_domain1, derivs_domain2, interpolation_domain3);
    ddc::DiscreteDomain<DDimI1, ddc::Deriv<I2>, ddc::Deriv<I3>> const
            derivs_domain23(interpolation_domain1, derivs_domain2, derivs_domain3);
    ddc::DiscreteDomain<ddc::Deriv<I1>, DDimI2, ddc::Deriv<I3>> const
            derivs_domain13(derivs_domain1, interpolation_domain2, derivs_domain3);
    ddc::DiscreteDomain<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>> const
            derivs_domain(derivs_domain1, derivs_domain2, derivs_domain3);

    auto const dom_derivs1 = ddc::replace_dim_of<DDimI1, ddc::Deriv<I1>>(dom_vals, derivs_domain1);
    auto const dom_derivs2 = ddc::replace_dim_of<DDimI2, ddc::Deriv<I2>>(dom_vals, derivs_domain2);
    auto const dom_derivs3 = ddc::replace_dim_of<DDimI3, ddc::Deriv<I3>>(dom_vals, derivs_domain3);
    auto const dom_derivs12
            = ddc::replace_dim_of<DDimI2, ddc::Deriv<I2>>(dom_derivs1, derivs_domain2);
    auto const dom_derivs23
            = ddc::replace_dim_of<DDimI3, ddc::Deriv<I3>>(dom_derivs2, derivs_domain3);
    auto const dom_derivs13
            = ddc::replace_dim_of<DDimI3, ddc::Deriv<I3>>(dom_derivs1, derivs_domain3);
    auto const dom_derivs = ddc::replace_dim_of<DDimI3, ddc::Deriv<I3>>(
            ddc::replace_dim_of<DDimI2, ddc::Deriv<I2>>(dom_derivs1, derivs_domain2),
            derivs_domain3);
#endif

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

#if defined(BC_HERMITE)
    int const shift = s_degree % 2; // shift = 0 for even order, 1 for odd order

    ddc::Chunk derivs1_lhs_alloc(dom_derivs1, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs1_lhs = derivs1_lhs_alloc.span_view();
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs1_lhs1_host_alloc(
                ddc::DiscreteDomain<
                        ddc::Deriv<I1>,
                        DDimI2,
                        DDimI3>(derivs_domain1, interpolation_domain2, interpolation_domain3),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs1_lhs1_host = derivs1_lhs1_host_alloc.span_view();
        ddc::for_each(
                derivs1_lhs1_host.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<ddc::Deriv<I1>, DDimI2, DDimI3> const e) {
                    auto deriv_idx = ddc::DiscreteElement<ddc::Deriv<I1>>(e).uid();
                    auto x2 = ddc::coordinate(ddc::DiscreteElement<DDimI2>(e));
                    auto x3 = ddc::coordinate(ddc::DiscreteElement<DDimI3>(e));
                    derivs1_lhs1_host(e)
                            = evaluator.deriv(x0<I1>(), x2, x3, deriv_idx + shift - 1, 0, 0);
                });
        auto derivs1_lhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs1_lhs1_host);
        ddc::ChunkSpan const derivs1_lhs1 = derivs1_lhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs1_lhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs1_lhs.domain())::discrete_element_type const e) {
                    derivs1_lhs(e) = derivs1_lhs1(DElem<ddc::Deriv<I1>, DDimI2, DDimI3>(e));
                });
    }

    ddc::Chunk derivs1_rhs_alloc(dom_derivs1, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs1_rhs = derivs1_rhs_alloc.span_view();
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs1_rhs1_host_alloc(
                ddc::DiscreteDomain<
                        ddc::Deriv<I1>,
                        DDimI2,
                        DDimI3>(derivs_domain1, interpolation_domain2, interpolation_domain3),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs1_rhs1_host = derivs1_rhs1_host_alloc.span_view();
        ddc::for_each(
                derivs1_rhs1_host.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<ddc::Deriv<I1>, DDimI2, DDimI3> const e) {
                    auto deriv_idx = ddc::DiscreteElement<ddc::Deriv<I1>>(e).uid();
                    auto x2 = ddc::coordinate(ddc::DiscreteElement<DDimI2>(e));
                    auto x3 = ddc::coordinate(ddc::DiscreteElement<DDimI3>(e));
                    derivs1_rhs1_host(e)
                            = evaluator.deriv(xN<I1>(), x2, x3, deriv_idx + shift - 1, 0, 0);
                });
        auto derivs1_rhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs1_rhs1_host);
        ddc::ChunkSpan const derivs1_rhs1 = derivs1_rhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs1_rhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs1_rhs.domain())::discrete_element_type const e) {
                    derivs1_rhs(e) = derivs1_rhs1(DElem<ddc::Deriv<I1>, DDimI2, DDimI3>(e));
                });
    }

    ddc::Chunk derivs2_lhs_alloc(dom_derivs2, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs2_lhs = derivs2_lhs_alloc.span_view();
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs2_lhs1_host_alloc(
                ddc::DiscreteDomain<
                        DDimI1,
                        ddc::Deriv<I2>,
                        DDimI3>(interpolation_domain1, derivs_domain2, interpolation_domain3),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs2_lhs1_host = derivs2_lhs1_host_alloc.span_view();
        ddc::for_each(
                derivs2_lhs1_host.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimI1, ddc::Deriv<I2>, DDimI3> const e) {
                    auto x1 = ddc::coordinate(ddc::DiscreteElement<DDimI1>(e));
                    auto deriv_idx = ddc::DiscreteElement<ddc::Deriv<I2>>(e).uid();
                    auto x3 = ddc::coordinate(ddc::DiscreteElement<DDimI3>(e));
                    derivs2_lhs1_host(e)
                            = evaluator.deriv(x1, x0<I2>(), x3, 0, deriv_idx + shift - 1, 0);
                });

        auto derivs2_lhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs2_lhs1_host);
        ddc::ChunkSpan const derivs2_lhs1 = derivs2_lhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs2_lhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs2_lhs.domain())::discrete_element_type const e) {
                    derivs2_lhs(e) = derivs2_lhs1(DElem<DDimI1, ddc::Deriv<I2>, DDimI3>(e));
                });
    }

    ddc::Chunk derivs2_rhs_alloc(dom_derivs2, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs2_rhs = derivs2_rhs_alloc.span_view();
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs2_rhs1_host_alloc(
                ddc::DiscreteDomain<
                        DDimI1,
                        ddc::Deriv<I2>,
                        DDimI3>(interpolation_domain1, derivs_domain2, interpolation_domain3),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs2_rhs1_host = derivs2_rhs1_host_alloc.span_view();
        ddc::for_each(
                derivs2_rhs1_host.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimI1, ddc::Deriv<I2>, DDimI3> const e) {
                    auto x1 = ddc::coordinate(ddc::DiscreteElement<DDimI1>(e));
                    auto deriv_idx = ddc::DiscreteElement<ddc::Deriv<I2>>(e).uid();
                    auto x3 = ddc::coordinate(ddc::DiscreteElement<DDimI3>(e));
                    derivs2_rhs1_host(e)
                            = evaluator.deriv(x1, xN<I2>(), x3, 0, deriv_idx + shift - 1, 0);
                });

        auto derivs2_rhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs2_rhs1_host);
        ddc::ChunkSpan const derivs2_rhs1 = derivs2_rhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs2_rhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs2_rhs.domain())::discrete_element_type const e) {
                    derivs2_rhs(e) = derivs2_rhs1(DElem<DDimI1, ddc::Deriv<I2>, DDimI3>(e));
                });
    }

    ddc::Chunk derivs3_lhs_alloc(dom_derivs3, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs3_lhs = derivs3_lhs_alloc.span_view();
    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs3_lhs1_host_alloc(
                ddc::DiscreteDomain<
                        DDimI1,
                        DDimI2,
                        ddc::Deriv<
                                I3>>(interpolation_domain1, interpolation_domain2, derivs_domain3),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs3_lhs1_host = derivs3_lhs1_host_alloc.span_view();
        ddc::for_each(
                derivs3_lhs1_host.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimI1, DDimI2, ddc::Deriv<I3>> const e) {
                    auto x1 = ddc::coordinate(ddc::DiscreteElement<DDimI1>(e));
                    auto x2 = ddc::coordinate(ddc::DiscreteElement<DDimI2>(e));
                    auto deriv_idx = ddc::DiscreteElement<ddc::Deriv<I3>>(e).uid();
                    derivs3_lhs1_host(e)
                            = evaluator.deriv(x1, x2, x0<I3>(), 0, 0, deriv_idx + shift - 1);
                });

        auto derivs3_lhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs3_lhs1_host);
        ddc::ChunkSpan const derivs3_lhs1 = derivs3_lhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs3_lhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs3_lhs.domain())::discrete_element_type const e) {
                    derivs3_lhs(e) = derivs3_lhs1(DElem<DDimI1, DDimI2, ddc::Deriv<I3>>(e));
                });
    }

    ddc::Chunk derivs3_rhs_alloc(dom_derivs3, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs3_rhs = derivs3_rhs_alloc.span_view();
    if (s_bcr == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs3_rhs1_host_alloc(
                ddc::DiscreteDomain<
                        DDimI1,
                        DDimI2,
                        ddc::Deriv<
                                I3>>(interpolation_domain1, interpolation_domain2, derivs_domain3),
                ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs3_rhs1_host = derivs3_rhs1_host_alloc.span_view();
        ddc::for_each(
                derivs3_rhs1_host.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimI1, DDimI2, ddc::Deriv<I3>> const e) {
                    auto x1 = ddc::coordinate(ddc::DiscreteElement<DDimI1>(e));
                    auto x2 = ddc::coordinate(ddc::DiscreteElement<DDimI2>(e));
                    auto deriv_idx = ddc::DiscreteElement<ddc::Deriv<I3>>(e).uid();
                    derivs3_rhs1_host(e)
                            = evaluator.deriv(x1, x2, xN<I3>(), 0, 0, deriv_idx + shift - 1);
                });

        auto derivs3_rhs1_alloc = ddc::create_mirror_view_and_copy(exec_space, derivs3_rhs1_host);
        ddc::ChunkSpan const derivs3_rhs1 = derivs3_rhs1_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                derivs3_rhs.domain(),
                KOKKOS_LAMBDA(
                        typename decltype(derivs3_rhs.domain())::discrete_element_type const e) {
                    derivs3_rhs(e) = derivs3_rhs1(DElem<DDimI1, DDimI2, ddc::Deriv<I3>>(e));
                });
    }

    ddc::Chunk
            derivs_mixed_lhs1_lhs2_alloc(dom_derivs12, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_lhs2 = derivs_mixed_lhs1_lhs2_alloc.span_view();
    ddc::Chunk
            derivs_mixed_lhs1_rhs2_alloc(dom_derivs12, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_rhs2 = derivs_mixed_lhs1_rhs2_alloc.span_view();
    ddc::Chunk
            derivs_mixed_rhs1_lhs2_alloc(dom_derivs12, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_lhs2 = derivs_mixed_rhs1_lhs2_alloc.span_view();
    ddc::Chunk
            derivs_mixed_rhs1_rhs2_alloc(dom_derivs12, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_rhs2 = derivs_mixed_rhs1_rhs2_alloc.span_view();

    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_mixed_lhs1_lhs2_host_alloc(derivs_domain12, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_host
                = derivs_mixed_lhs1_lhs2_host_alloc.span_view();
        ddc::Chunk derivs_mixed_lhs1_rhs2_host_alloc(derivs_domain12, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_host
                = derivs_mixed_lhs1_rhs2_host_alloc.span_view();
        ddc::Chunk derivs_mixed_rhs1_lhs2_host_alloc(derivs_domain12, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_host
                = derivs_mixed_rhs1_lhs2_host_alloc.span_view();
        ddc::Chunk derivs_mixed_rhs1_rhs2_host_alloc(derivs_domain12, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_host
                = derivs_mixed_rhs1_rhs2_host_alloc.span_view();

        ddc::for_each(
                derivs_domain12,
                KOKKOS_LAMBDA(
                        ddc::DiscreteElement<ddc::Deriv<I1>, ddc::Deriv<I2>, DDimI3> const e) {
                    auto deriv_idx1 = ddc::DiscreteElement<ddc::Deriv<I1>>(e).uid();
                    auto deriv_idx2 = ddc::DiscreteElement<ddc::Deriv<I2>>(e).uid();
                    auto x3 = ddc::coordinate(ddc::DiscreteElement<DDimI3>(e));
                    derivs_mixed_lhs1_lhs2_host(e) = evaluator
                                                             .deriv(x0<I1>(),
                                                                    x0<I2>(),
                                                                    x3,
                                                                    deriv_idx1 + shift - 1,
                                                                    deriv_idx2 + shift - 1,
                                                                    0);
                    derivs_mixed_lhs1_rhs2_host(e) = evaluator
                                                             .deriv(x0<I1>(),
                                                                    xN<I2>(),
                                                                    x3,
                                                                    deriv_idx1 + shift - 1,
                                                                    deriv_idx2 + shift - 1,
                                                                    0);
                    derivs_mixed_rhs1_lhs2_host(e) = evaluator
                                                             .deriv(xN<I1>(),
                                                                    x0<I2>(),
                                                                    x3,
                                                                    deriv_idx1 + shift - 1,
                                                                    deriv_idx2 + shift - 1,
                                                                    0);
                    derivs_mixed_rhs1_rhs2_host(e) = evaluator
                                                             .deriv(xN<I1>(),
                                                                    xN<I2>(),
                                                                    x3,
                                                                    deriv_idx1 + shift - 1,
                                                                    deriv_idx2 + shift - 1,
                                                                    0);
                });

        auto derivs_mixed_lhs1_lhs2_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_lhs2_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_ = derivs_mixed_lhs1_lhs2_alloc.span_view();
        auto derivs_mixed_lhs1_rhs2_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_rhs2_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_ = derivs_mixed_lhs1_rhs2_alloc.span_view();
        auto derivs_mixed_rhs1_lhs2_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_lhs2_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_ = derivs_mixed_rhs1_lhs2_alloc.span_view();
        auto derivs_mixed_rhs1_rhs2_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_rhs2_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_ = derivs_mixed_rhs1_rhs2_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                dom_derivs12,
                KOKKOS_LAMBDA(typename decltype(dom_derivs12)::discrete_element_type const e) {
                    derivs_mixed_lhs1_lhs2(e) = derivs_mixed_lhs1_lhs2_(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, DDimI3>(e));
                    derivs_mixed_lhs1_rhs2(e) = derivs_mixed_lhs1_rhs2_(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, DDimI3>(e));
                    derivs_mixed_rhs1_lhs2(e) = derivs_mixed_rhs1_lhs2_(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, DDimI3>(e));
                    derivs_mixed_rhs1_rhs2(e) = derivs_mixed_rhs1_rhs2_(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, DDimI3>(e));
                });
    }

    ddc::Chunk
            derivs_mixed_lhs2_lhs3_alloc(dom_derivs23, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs2_lhs3 = derivs_mixed_lhs2_lhs3_alloc.span_view();
    ddc::Chunk
            derivs_mixed_lhs2_rhs3_alloc(dom_derivs23, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs2_rhs3 = derivs_mixed_lhs2_rhs3_alloc.span_view();
    ddc::Chunk
            derivs_mixed_rhs2_lhs3_alloc(dom_derivs23, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs2_lhs3 = derivs_mixed_rhs2_lhs3_alloc.span_view();
    ddc::Chunk
            derivs_mixed_rhs2_rhs3_alloc(dom_derivs23, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs2_rhs3 = derivs_mixed_rhs2_rhs3_alloc.span_view();

    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_mixed_lhs2_lhs3_host_alloc(derivs_domain23, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs2_lhs3_host
                = derivs_mixed_lhs2_lhs3_host_alloc.span_view();
        ddc::Chunk derivs_mixed_lhs2_rhs3_host_alloc(derivs_domain23, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs2_rhs3_host
                = derivs_mixed_lhs2_rhs3_host_alloc.span_view();
        ddc::Chunk derivs_mixed_rhs2_lhs3_host_alloc(derivs_domain23, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs2_lhs3_host
                = derivs_mixed_rhs2_lhs3_host_alloc.span_view();
        ddc::Chunk derivs_mixed_rhs2_rhs3_host_alloc(derivs_domain23, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs2_rhs3_host
                = derivs_mixed_rhs2_rhs3_host_alloc.span_view();

        ddc::for_each(
                derivs_domain23,
                KOKKOS_LAMBDA(
                        ddc::DiscreteElement<DDimI1, ddc::Deriv<I2>, ddc::Deriv<I3>> const e) {
                    auto x1 = ddc::coordinate(ddc::DiscreteElement<DDimI1>(e));
                    auto deriv_idx2 = ddc::DiscreteElement<ddc::Deriv<I2>>(e).uid();
                    auto deriv_idx3 = ddc::DiscreteElement<ddc::Deriv<I3>>(e).uid();
                    derivs_mixed_lhs2_lhs3_host(e) = evaluator
                                                             .deriv(x1,
                                                                    x0<I2>(),
                                                                    x0<I3>(),
                                                                    0,
                                                                    deriv_idx2 + shift - 1,
                                                                    deriv_idx3 + shift - 1);
                    derivs_mixed_lhs2_rhs3_host(e) = evaluator
                                                             .deriv(x1,
                                                                    x0<I2>(),
                                                                    xN<I3>(),
                                                                    0,
                                                                    deriv_idx2 + shift - 1,
                                                                    deriv_idx3 + shift - 1);
                    derivs_mixed_rhs2_lhs3_host(e) = evaluator
                                                             .deriv(x1,
                                                                    xN<I2>(),
                                                                    x0<I3>(),
                                                                    0,
                                                                    deriv_idx2 + shift - 1,
                                                                    deriv_idx3 + shift - 1);
                    derivs_mixed_rhs2_rhs3_host(e) = evaluator
                                                             .deriv(x1,
                                                                    xN<I2>(),
                                                                    xN<I3>(),
                                                                    0,
                                                                    deriv_idx2 + shift - 1,
                                                                    deriv_idx3 + shift - 1);
                });

        auto derivs_mixed_lhs2_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs2_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs2_lhs3_ = derivs_mixed_lhs2_lhs3_alloc.span_view();
        auto derivs_mixed_lhs2_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs2_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs2_rhs3_ = derivs_mixed_lhs2_rhs3_alloc.span_view();
        auto derivs_mixed_rhs2_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs2_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs2_lhs3_ = derivs_mixed_rhs2_lhs3_alloc.span_view();
        auto derivs_mixed_rhs2_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs2_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs2_rhs3_ = derivs_mixed_rhs2_rhs3_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                dom_derivs23,
                KOKKOS_LAMBDA(typename decltype(dom_derivs23)::discrete_element_type const e) {
                    derivs_mixed_lhs2_lhs3(e) = derivs_mixed_lhs2_lhs3_(
                            DElem<DDimI1, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_lhs2_rhs3(e) = derivs_mixed_lhs2_rhs3_(
                            DElem<DDimI1, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs2_lhs3(e) = derivs_mixed_rhs2_lhs3_(
                            DElem<DDimI1, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs2_rhs3(e) = derivs_mixed_rhs2_rhs3_(
                            DElem<DDimI1, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                });
    }

    ddc::Chunk
            derivs_mixed_lhs1_lhs3_alloc(dom_derivs13, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_lhs3 = derivs_mixed_lhs1_lhs3_alloc.span_view();
    ddc::Chunk
            derivs_mixed_lhs1_rhs3_alloc(dom_derivs13, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_rhs3 = derivs_mixed_lhs1_rhs3_alloc.span_view();
    ddc::Chunk
            derivs_mixed_rhs1_lhs3_alloc(dom_derivs13, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_lhs3 = derivs_mixed_rhs1_lhs3_alloc.span_view();
    ddc::Chunk
            derivs_mixed_rhs1_rhs3_alloc(dom_derivs13, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_rhs3 = derivs_mixed_rhs1_rhs3_alloc.span_view();

    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk derivs_mixed_lhs1_lhs3_host_alloc(derivs_domain13, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs3_host
                = derivs_mixed_lhs1_lhs3_host_alloc.span_view();
        ddc::Chunk derivs_mixed_lhs1_rhs3_host_alloc(derivs_domain13, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs3_host
                = derivs_mixed_lhs1_rhs3_host_alloc.span_view();
        ddc::Chunk derivs_mixed_rhs1_lhs3_host_alloc(derivs_domain13, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs3_host
                = derivs_mixed_rhs1_lhs3_host_alloc.span_view();
        ddc::Chunk derivs_mixed_rhs1_rhs3_host_alloc(derivs_domain13, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs3_host
                = derivs_mixed_rhs1_rhs3_host_alloc.span_view();

        ddc::for_each(
                derivs_domain13,
                KOKKOS_LAMBDA(
                        ddc::DiscreteElement<ddc::Deriv<I1>, DDimI2, ddc::Deriv<I3>> const e) {
                    auto deriv_idx1 = ddc::DiscreteElement<ddc::Deriv<I1>>(e).uid();
                    auto x2 = ddc::coordinate(ddc::DiscreteElement<DDimI2>(e));
                    auto deriv_idx3 = ddc::DiscreteElement<ddc::Deriv<I3>>(e).uid();
                    derivs_mixed_lhs1_lhs3_host(e) = evaluator
                                                             .deriv(x0<I1>(),
                                                                    x2,
                                                                    x0<I3>(),
                                                                    deriv_idx1 + shift - 1,
                                                                    0,
                                                                    deriv_idx3 + shift - 1);
                    derivs_mixed_lhs1_rhs3_host(e) = evaluator
                                                             .deriv(x0<I1>(),
                                                                    x2,
                                                                    xN<I3>(),
                                                                    deriv_idx1 + shift - 1,
                                                                    0,
                                                                    deriv_idx3 + shift - 1);
                    derivs_mixed_rhs1_lhs3_host(e) = evaluator
                                                             .deriv(xN<I1>(),
                                                                    x2,
                                                                    x0<I3>(),
                                                                    deriv_idx1 + shift - 1,
                                                                    0,
                                                                    deriv_idx3 + shift - 1);
                    derivs_mixed_rhs1_rhs3_host(e) = evaluator
                                                             .deriv(xN<I1>(),
                                                                    x2,
                                                                    xN<I3>(),
                                                                    deriv_idx1 + shift - 1,
                                                                    0,
                                                                    deriv_idx3 + shift - 1);
                });

        auto derivs_mixed_lhs1_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs3_ = derivs_mixed_lhs1_lhs3_alloc.span_view();
        auto derivs_mixed_lhs1_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs3_ = derivs_mixed_lhs1_rhs3_alloc.span_view();
        auto derivs_mixed_rhs1_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs3_ = derivs_mixed_rhs1_lhs3_alloc.span_view();
        auto derivs_mixed_rhs1_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs3_ = derivs_mixed_rhs1_rhs3_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                dom_derivs13,
                KOKKOS_LAMBDA(typename decltype(dom_derivs13)::discrete_element_type const e) {
                    derivs_mixed_lhs1_lhs3(e) = derivs_mixed_lhs1_lhs3_(
                            DElem<ddc::Deriv<I1>, DDimI2, ddc::Deriv<I3>>(e));
                    derivs_mixed_lhs1_rhs3(e) = derivs_mixed_lhs1_rhs3_(
                            DElem<ddc::Deriv<I1>, DDimI2, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs1_lhs3(e) = derivs_mixed_rhs1_lhs3_(
                            DElem<ddc::Deriv<I1>, DDimI2, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs1_rhs3(e) = derivs_mixed_rhs1_rhs3_(
                            DElem<ddc::Deriv<I1>, DDimI2, ddc::Deriv<I3>>(e));
                });
    }

    ddc::Chunk derivs_mixed_lhs1_lhs2_lhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_lhs3
            = derivs_mixed_lhs1_lhs2_lhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_rhs1_lhs2_lhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_lhs3
            = derivs_mixed_rhs1_lhs2_lhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_lhs1_rhs2_lhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_lhs3
            = derivs_mixed_lhs1_rhs2_lhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_rhs1_rhs2_lhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_lhs3
            = derivs_mixed_rhs1_rhs2_lhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_lhs1_lhs2_rhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_rhs3
            = derivs_mixed_lhs1_lhs2_rhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_rhs1_lhs2_rhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_rhs3
            = derivs_mixed_rhs1_lhs2_rhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_lhs1_rhs2_rhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_rhs3
            = derivs_mixed_lhs1_rhs2_rhs3_alloc.span_view();
    ddc::Chunk derivs_mixed_rhs1_rhs2_rhs3_alloc(
            dom_derivs,
            ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_rhs3
            = derivs_mixed_rhs1_rhs2_rhs3_alloc.span_view();

    if (s_bcl == ddc::BoundCond::HERMITE) {
        ddc::Chunk
                derivs_mixed_lhs1_lhs2_lhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_lhs3_host
                = derivs_mixed_lhs1_lhs2_lhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_rhs1_lhs2_lhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_lhs3_host
                = derivs_mixed_rhs1_lhs2_lhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_lhs1_rhs2_lhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_lhs3_host
                = derivs_mixed_lhs1_rhs2_lhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_rhs1_rhs2_lhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_lhs3_host
                = derivs_mixed_rhs1_rhs2_lhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_lhs1_lhs2_rhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_rhs3_host
                = derivs_mixed_lhs1_lhs2_rhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_rhs1_lhs2_rhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_rhs3_host
                = derivs_mixed_rhs1_lhs2_rhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_lhs1_rhs2_rhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_rhs3_host
                = derivs_mixed_lhs1_rhs2_rhs3_host_alloc.span_view();
        ddc::Chunk
                derivs_mixed_rhs1_rhs2_rhs3_host_alloc(derivs_domain, ddc::HostAllocator<double>());
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_rhs3_host
                = derivs_mixed_rhs1_rhs2_rhs3_host_alloc.span_view();

        for (std::size_t ii = 1;
             ii < static_cast<std::size_t>(derivs_domain.template extent<ddc::Deriv<I1>>()) + 1;
             ++ii) {
            for (std::size_t jj = 1;
                 jj < static_cast<std::size_t>(derivs_domain.template extent<ddc::Deriv<I2>>()) + 1;
                 ++jj) {
                for (std::size_t kk = 1;
                     kk < static_cast<std::size_t>(derivs_domain.template extent<ddc::Deriv<I3>>())
                                  + 1;
                     ++kk) {
                    derivs_mixed_lhs1_lhs2_lhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(x0<I1>(),
                                             x0<I2>(),
                                             x0<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_rhs1_lhs2_lhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(xN<I1>(),
                                             x0<I2>(),
                                             x0<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_lhs1_rhs2_lhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(x0<I1>(),
                                             xN<I2>(),
                                             x0<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_rhs1_rhs2_lhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(xN<I1>(),
                                             xN<I2>(),
                                             x0<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_lhs1_lhs2_rhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(x0<I1>(),
                                             x0<I2>(),
                                             xN<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_rhs1_lhs2_rhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(xN<I1>(),
                                             x0<I2>(),
                                             xN<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_lhs1_rhs2_rhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(x0<I1>(),
                                             xN<I2>(),
                                             xN<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                    derivs_mixed_rhs1_rhs2_rhs3_host(
                            typename decltype(derivs_domain)::discrete_element_type(ii, jj, kk))
                            = evaluator
                                      .deriv(xN<I1>(),
                                             xN<I2>(),
                                             xN<I3>(),
                                             ii + shift - 1,
                                             jj + shift - 1,
                                             kk + shift - 1);
                }
            }
        }
        auto derivs_mixed_lhs1_lhs2_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_lhs2_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_lhs31
                = derivs_mixed_lhs1_lhs2_lhs3_alloc.span_view();
        auto derivs_mixed_rhs1_lhs2_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_lhs2_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_lhs31
                = derivs_mixed_rhs1_lhs2_lhs3_alloc.span_view();
        auto derivs_mixed_lhs1_rhs2_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_rhs2_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_lhs31
                = derivs_mixed_lhs1_rhs2_lhs3_alloc.span_view();
        auto derivs_mixed_rhs1_rhs2_lhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_rhs2_lhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_lhs31
                = derivs_mixed_rhs1_rhs2_lhs3_alloc.span_view();
        auto derivs_mixed_lhs1_lhs2_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_lhs2_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_lhs2_rhs31
                = derivs_mixed_lhs1_lhs2_rhs3_alloc.span_view();
        auto derivs_mixed_rhs1_lhs2_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_lhs2_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_lhs2_rhs31
                = derivs_mixed_rhs1_lhs2_rhs3_alloc.span_view();
        auto derivs_mixed_lhs1_rhs2_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_lhs1_rhs2_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_lhs1_rhs2_rhs31
                = derivs_mixed_lhs1_rhs2_rhs3_alloc.span_view();
        auto derivs_mixed_rhs1_rhs2_rhs3_alloc
                = ddc::create_mirror_view_and_copy(exec_space, derivs_mixed_rhs1_rhs2_rhs3_host);
        ddc::ChunkSpan const derivs_mixed_rhs1_rhs2_rhs31
                = derivs_mixed_rhs1_rhs2_rhs3_alloc.span_view();

        ddc::parallel_for_each(
                exec_space,
                dom_derivs,
                KOKKOS_LAMBDA(typename decltype(dom_derivs)::discrete_element_type const e) {
                    derivs_mixed_lhs1_lhs2_lhs3(e) = derivs_mixed_lhs1_lhs2_lhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs1_lhs2_lhs3(e) = derivs_mixed_rhs1_lhs2_lhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_lhs1_rhs2_lhs3(e) = derivs_mixed_lhs1_rhs2_lhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs1_rhs2_lhs3(e) = derivs_mixed_rhs1_rhs2_lhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_lhs1_lhs2_rhs3(e) = derivs_mixed_lhs1_lhs2_rhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs1_lhs2_rhs3(e) = derivs_mixed_rhs1_lhs2_rhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_lhs1_rhs2_rhs3(e) = derivs_mixed_lhs1_rhs2_rhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
                    derivs_mixed_rhs1_rhs2_rhs3(e) = derivs_mixed_rhs1_rhs2_rhs31(
                            DElem<ddc::Deriv<I1>, ddc::Deriv<I2>, ddc::Deriv<I3>>(e));
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
            std::optional(derivs1_lhs.span_cview()),
            std::optional(derivs1_rhs.span_cview()),
            std::optional(derivs2_lhs.span_cview()),
            std::optional(derivs2_rhs.span_cview()),
            std::optional(derivs3_lhs.span_cview()),
            std::optional(derivs3_rhs.span_cview()),
            std::optional(derivs_mixed_lhs1_lhs2.span_cview()),
            std::optional(derivs_mixed_lhs1_rhs2.span_cview()),
            std::optional(derivs_mixed_rhs1_lhs2.span_cview()),
            std::optional(derivs_mixed_rhs1_rhs2.span_cview()),
            std::optional(derivs_mixed_lhs2_lhs3.span_cview()),
            std::optional(derivs_mixed_lhs2_rhs3.span_cview()),
            std::optional(derivs_mixed_rhs2_lhs3.span_cview()),
            std::optional(derivs_mixed_rhs2_rhs3.span_cview()),
            std::optional(derivs_mixed_lhs1_lhs3.span_cview()),
            std::optional(derivs_mixed_lhs1_rhs3.span_cview()),
            std::optional(derivs_mixed_rhs1_lhs3.span_cview()),
            std::optional(derivs_mixed_rhs1_rhs3.span_cview()),
            std::optional(derivs_mixed_lhs1_lhs2_lhs3.span_cview()),
            std::optional(derivs_mixed_lhs1_rhs2_lhs3.span_cview()),
            std::optional(derivs_mixed_rhs1_lhs2_lhs3.span_cview()),
            std::optional(derivs_mixed_rhs1_rhs2_lhs3.span_cview()),
            std::optional(derivs_mixed_lhs1_lhs2_rhs3.span_cview()),
            std::optional(derivs_mixed_lhs1_rhs2_rhs3.span_cview()),
            std::optional(derivs_mixed_rhs1_lhs2_rhs3.span_cview()),
            std::optional(derivs_mixed_rhs1_rhs2_rhs3.span_cview()));
#else
    spline_builder(coef, vals.span_cview());
#endif

    // Instantiate a SplineEvaluator over interest dimension and batched along other dimensions
#if defined(BC_PERIODIC)
    using extrapolation_rule_1_type = ddc::PeriodicExtrapolationRule<I1>;
    using extrapolation_rule_2_type = ddc::PeriodicExtrapolationRule<I2>;
    using extrapolation_rule_3_type = ddc::PeriodicExtrapolationRule<I3>;
#else
    using extrapolation_rule_1_type = ddc::NullExtrapolationRule;
    using extrapolation_rule_2_type = ddc::NullExtrapolationRule;
    using extrapolation_rule_3_type = ddc::NullExtrapolationRule;
#endif
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
    ddc::Chunk spline_eval_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval = spline_eval_alloc.span_view();
    ddc::Chunk spline_eval_deriv1_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv1 = spline_eval_deriv1_alloc.span_view();
    ddc::Chunk spline_eval_deriv2_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv2 = spline_eval_deriv2_alloc.span_view();
    ddc::Chunk spline_eval_deriv3_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv3 = spline_eval_deriv3_alloc.span_view();
    ddc::Chunk spline_eval_deriv12_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv12 = spline_eval_deriv12_alloc.span_view();
    ddc::Chunk spline_eval_deriv23_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv23 = spline_eval_deriv23_alloc.span_view();
    ddc::Chunk spline_eval_deriv13_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv13 = spline_eval_deriv13_alloc.span_view();
    ddc::Chunk spline_eval_deriv123_alloc(dom_vals, ddc::KokkosAllocator<double, MemorySpace>());
    ddc::ChunkSpan const spline_eval_deriv123 = spline_eval_deriv123_alloc.span_view();

    // Call spline_evaluator on the same mesh we started with
    spline_evaluator(spline_eval, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator
            .template deriv<I1>(spline_eval_deriv1, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator
            .template deriv<I2>(spline_eval_deriv2, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator
            .template deriv<I3>(spline_eval_deriv3, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator.template deriv2<
            I1,
            I2>(spline_eval_deriv12, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator.template deriv2<
            I2,
            I3>(spline_eval_deriv23, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator.template deriv2<
            I1,
            I3>(spline_eval_deriv13, coords_eval.span_cview(), coef.span_cview());
    spline_evaluator.template deriv3<
            I1,
            I2,
            I3>(spline_eval_deriv123, coords_eval.span_cview(), coef.span_cview());

    // Checking errors (we recover the initial values)
    double const max_norm_error = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                return Kokkos::abs(spline_eval(e) - vals(e));
            });
    double const max_norm_error_diff1 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv1.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv1(e) - evaluator.deriv(x, y, z, 1, 0, 0));
            });
    double const max_norm_error_diff2 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv2.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv2(e) - evaluator.deriv(x, y, z, 0, 1, 0));
            });
    double const max_norm_error_diff3 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv3.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv3(e) - evaluator.deriv(x, y, z, 0, 0, 1));
            });
    double const max_norm_error_diff12 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv12.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv12(e) - evaluator.deriv(x, y, z, 1, 1, 0));
            });
    double const max_norm_error_diff23 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv23.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv23(e) - evaluator.deriv(x, y, z, 0, 1, 1));
            });
    double const max_norm_error_diff13 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv13.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv13(e) - evaluator.deriv(x, y, z, 1, 0, 1));
            });
    double const max_norm_error_diff123 = ddc::parallel_transform_reduce(
            exec_space,
            spline_eval_deriv123.domain(),
            0.,
            ddc::reducer::max<double>(),
            KOKKOS_LAMBDA(DElem<DDims...> const e) {
                Coord<I1> const x = ddc::coordinate(DElem<DDimI1>(e));
                Coord<I2> const y = ddc::coordinate(DElem<DDimI2>(e));
                Coord<I3> const z = ddc::coordinate(DElem<DDimI3>(e));
                return Kokkos::abs(spline_eval_deriv123(e) - evaluator.deriv(x, y, z, 1, 1, 1));
            });

    double const max_norm = evaluator.max_norm();
    double const max_norm_diff1 = evaluator.max_norm(1, 0, 0);
    double const max_norm_diff2 = evaluator.max_norm(0, 1, 0);
    double const max_norm_diff3 = evaluator.max_norm(0, 0, 1);
    double const max_norm_diff12 = evaluator.max_norm(1, 1, 0);
    double const max_norm_diff23 = evaluator.max_norm(0, 1, 1);
    double const max_norm_diff13 = evaluator.max_norm(1, 0, 1);
    double const max_norm_diff123 = evaluator.max_norm(1, 1, 1);

    SplineErrorBounds<evaluator_type<DDimI1, DDimI2, DDimI3>> const error_bounds(evaluator);
    EXPECT_LE(
            max_norm_error,
            std::
                    max(error_bounds.error_bound(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1.0e-14 * max_norm));
    EXPECT_LE(
            max_norm_error_diff1,
            std::
                    max(error_bounds.error_bound_on_deriv_1(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1e-12 * max_norm_diff1));
    EXPECT_LE(
            max_norm_error_diff2,
            std::
                    max(error_bounds.error_bound_on_deriv_2(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1e-12 * max_norm_diff2));
    EXPECT_LE(
            max_norm_error_diff3,
            std::
                    max(error_bounds.error_bound_on_deriv_3(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1e-12 * max_norm_diff3));
    EXPECT_LE(
            max_norm_error_diff12,
            std::
                    max(error_bounds.error_bound_on_deriv_12(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1e-11 * max_norm_diff12));
    EXPECT_LE(
            max_norm_error_diff23,
            std::
                    max(error_bounds.error_bound_on_deriv_23(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1e-11 * max_norm_diff23));
    EXPECT_LE(
            max_norm_error_diff13,
            std::
                    max(error_bounds.error_bound_on_deriv_13(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        1e-11 * max_norm_diff13));
    EXPECT_LE(
            max_norm_error_diff123,
            std::
                    max(error_bounds.error_bound_on_deriv_123(
                                dx<I1>(ncells),
                                dx<I2>(ncells),
                                dx<I3>(ncells),
                                s_degree,
                                s_degree,
                                s_degree),
                        5e-10 * max_norm_diff123));
}

} // namespace anonymous_namespace_workaround_batched_3d_spline_builder_cpp

#if defined(BC_PERIODIC) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Periodic##Uniform
#elif defined(BC_PERIODIC) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Periodic##NonUniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Greville##Uniform
#elif defined(BC_GREVILLE) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Greville##NonUniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_UNIFORM)
#    define SUFFIX(name) name##Hermite##Uniform
#elif defined(BC_HERMITE) && defined(BSPLINES_TYPE_NON_UNIFORM)
#    define SUFFIX(name) name##Hermite##NonUniform
#endif

TEST(SUFFIX(Batched3dSplineHost), 3DXYZ)
{
    Batched3dSplineTest<
            Kokkos::DefaultHostExecutionSpace,
            Kokkos::DefaultHostExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>>();
}

TEST(SUFFIX(Batched3dSplineDevice), 3DXYZ)
{
    Batched3dSplineTest<
            Kokkos::DefaultExecutionSpace,
            Kokkos::DefaultExecutionSpace::memory_space,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>,
            DDimGPS<DimX>,
            DDimGPS<DimY>,
            DDimGPS<DimZ>>();
}

TEST(SUFFIX(Batched3dSplineHost), 4DXYZB)
{
    Batched3dSplineTest<
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

TEST(SUFFIX(Batched3dSplineHost), 4DXBYZ)
{
    Batched3dSplineTest<
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

TEST(SUFFIX(Batched3dSplineHost), 4DXYBZ)
{
    Batched3dSplineTest<
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

TEST(SUFFIX(Batched3dSplineHost), 4DBXYZ)
{
    Batched3dSplineTest<
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
