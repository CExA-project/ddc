// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "test_utils.hpp"

template <class T>
struct BSplinesFixture;

template <std::size_t D, std::size_t Nc, bool periodic>
struct BSplinesFixture<std::tuple<
        std::integral_constant<std::size_t, D>,
        std::integral_constant<std::size_t, Nc>,
        std::integral_constant<bool, periodic>>> : public testing::Test
{
    struct DimX
    {
        static constexpr bool PERIODIC = periodic;
    };
    struct UBSplinesX : ddc::UniformBSplines<DimX, D>
    {
    };
    struct NUBSplinesX : ddc::NonUniformBSplines<DimX, D>
    {
    };
    static constexpr std::size_t spline_degree = D;
    static constexpr std::size_t ncells = Nc;
};

using degrees = std::integer_sequence<std::size_t, 1, 2, 3, 4, 5, 6>;
using ncells = std::integer_sequence<std::size_t, 10, 20, 100>;
using periodicity = std::integer_sequence<bool, true, false>;

using Cases = tuple_to_types_t<cartesian_product_t<degrees, ncells, periodicity>>;

// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(BSplinesFixture, Cases, );

TYPED_TEST(BSplinesFixture, PartitionOfUnityUniform)
{
    std::size_t constexpr degree = TestFixture::spline_degree;
    using DimX = typename TestFixture::DimX;
    using BSplinesX = typename TestFixture::UBSplinesX;
    using CoordX = ddc::Coordinate<DimX>;
    static constexpr CoordX xmin = CoordX(0.0);
    static constexpr CoordX xmax = CoordX(0.2);
    static constexpr std::size_t ncells = TestFixture::ncells;
    ddc::init_discrete_space<BSplinesX>(xmin, xmax, ncells);

    std::array<double, degree + 1> values_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree + 1>> const values(
            values_ptr.data());

    std::size_t const n_test_points = ncells * 30;
    double const dx = (xmax - xmin) / (n_test_points - 1);

    for (std::size_t i(0); i < n_test_points; ++i) {
        CoordX const test_point(xmin + dx * i);
        ddc::discrete_space<BSplinesX>().eval_basis(values, test_point);
        double sum = 0.0;
        for (std::size_t j(0); j < degree + 1; ++j) {
            sum += DDC_MDSPAN_ACCESS_OP(values, j);
        }
        EXPECT_LE(fabs(sum - 1.0), 1.0e-15);
    }
}

TYPED_TEST(BSplinesFixture, PartitionOfUnityNonUniform)
{
    std::size_t constexpr degree = TestFixture::spline_degree;
    using DimX = typename TestFixture::DimX;
    using BSplinesX = typename TestFixture::NUBSplinesX;
    using CoordX = ddc::Coordinate<DimX>;
    static constexpr CoordX xmin = CoordX(0.0);
    static constexpr CoordX xmax = CoordX(0.2);
    static constexpr std::size_t ncells = TestFixture::ncells;
    std::vector<CoordX> breaks(ncells + 1);
    double dx = (xmax - xmin) / ncells;
    for (std::size_t i(0); i < ncells + 1; ++i) {
        breaks[i] = CoordX(xmin + i * dx);
    }
    ddc::init_discrete_space<BSplinesX>(breaks);

    std::array<double, degree + 1> values_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree + 1>> const values(
            values_ptr.data());


    std::size_t const n_test_points = ncells * 30;
    dx = (xmax - xmin) / (n_test_points - 1);

    for (std::size_t i(0); i < n_test_points; ++i) {
        CoordX const test_point(xmin + dx * i);
        ddc::discrete_space<BSplinesX>().eval_basis(values, test_point);
        double sum = 0.0;
        for (std::size_t j(0); j < degree + 1; ++j) {
            sum += DDC_MDSPAN_ACCESS_OP(values, j);
        }
        EXPECT_LE(fabs(sum - 1.0), 1.0e-15);
    }
}

TEST(KnotDiscreteDimension, Type)
{
    struct DDim1 : ddc::UniformBSplines<struct X, 1>
    {
    };
    EXPECT_TRUE((std::is_same_v<
                 ddc::knot_discrete_dimension_t<DDim1>,
                 ddc::UniformBsplinesKnots<DDim1>>));

    struct DDim2 : ddc::NonUniformBSplines<struct X, 1>
    {
    };
    EXPECT_TRUE((std::is_same_v<
                 ddc::knot_discrete_dimension_t<DDim2>,
                 ddc::NonUniformBsplinesKnots<DDim2>>));
}

TYPED_TEST(BSplinesFixture, RoundingNonUniform)
{
    std::size_t constexpr degree = TestFixture::spline_degree;
    using DimX = typename TestFixture::DimX;
    using BSplinesX = typename TestFixture::NUBSplinesX;
    using CoordX = ddc::Coordinate<DimX>;
    static constexpr CoordX xmin = CoordX(0.0);
    static constexpr CoordX xmax = CoordX(0.2);
    static constexpr std::size_t ncells = TestFixture::ncells;
    std::vector<CoordX> breaks(ncells + 1);
    double const dx = (xmax - xmin) / ncells;
    for (std::size_t i(0); i < ncells + 1; ++i) {
        breaks[i] = CoordX(xmin + i * dx);
    }
    ddc::init_discrete_space<BSplinesX>(breaks);

    ddc::DiscreteDomain<BSplinesX> const bspl_full_domain
            = ddc::discrete_space<BSplinesX>().full_domain();

    std::array<double, degree + 1> values_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree + 1>> const values(
            values_ptr.data());

    CoordX const test_point_min(xmin - std::numeric_limits<double>::epsilon());
    ddc::DiscreteElement<BSplinesX> const front_idx
            = ddc::discrete_space<BSplinesX>().eval_basis(values, test_point_min);
    EXPECT_EQ(front_idx, bspl_full_domain.front());

    CoordX const test_point_max(xmax + std::numeric_limits<double>::epsilon());
    ddc::DiscreteElement<BSplinesX> const back_idx
            = ddc::discrete_space<BSplinesX>().eval_basis(values, test_point_max);
    EXPECT_EQ(back_idx, bspl_full_domain.back() - BSplinesX::degree());
}

TYPED_TEST(BSplinesFixture, RoundingUniform)
{
    std::size_t constexpr degree = TestFixture::spline_degree;
    using DimX = typename TestFixture::DimX;
    using BSplinesX = typename TestFixture::UBSplinesX;
    using CoordX = ddc::Coordinate<DimX>;
    static constexpr CoordX xmin = CoordX(0.0);
    static constexpr CoordX xmax = CoordX(0.2);
    static constexpr std::size_t ncells = TestFixture::ncells;
    ddc::init_discrete_space<BSplinesX>(xmin, xmax, ncells);

    ddc::DiscreteDomain<BSplinesX> const bspl_full_domain
            = ddc::discrete_space<BSplinesX>().full_domain();

    std::array<double, degree + 1> values_ptr;
    Kokkos::mdspan<double, Kokkos::extents<std::size_t, degree + 1>> const values(
            values_ptr.data());

    CoordX const test_point_min(xmin - std::numeric_limits<double>::epsilon());
    ddc::DiscreteElement<BSplinesX> const front_idx
            = ddc::discrete_space<BSplinesX>().eval_basis(values, test_point_min);
    EXPECT_EQ(front_idx, bspl_full_domain.front());

    CoordX const test_point_max(xmax + std::numeric_limits<double>::epsilon());
    ddc::DiscreteElement<BSplinesX> const back_idx
            = ddc::discrete_space<BSplinesX>().eval_basis(values, test_point_max);
    EXPECT_EQ(back_idx, bspl_full_domain.back() - BSplinesX::degree());
}
