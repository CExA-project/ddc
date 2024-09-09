// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include "test_utils.hpp"

namespace {

template <class T>
struct UniformBSplinesFixture;

template <bool IsPeriodic>
struct UniformBSplinesFixture<std::tuple<std::integral_constant<bool, IsPeriodic>>>
    : public testing::Test
{
    struct DimX
    {
        static constexpr bool PERIODIC = IsPeriodic;
    };

    struct DDimX : ddc::UniformPointSampling<DimX>
    {
    };

    static constexpr ddc::BoundCond Bc
            = IsPeriodic ? ddc::BoundCond::PERIODIC : ddc::BoundCond::HERMITE;

    struct BSplinesX : ddc::UniformBSplines<DimX, 2>
    {
    };
};

template <class T>
struct NonUniformBSplinesFixture;

template <bool IsPeriodic>
struct NonUniformBSplinesFixture<std::tuple<std::integral_constant<bool, IsPeriodic>>>
    : public testing::Test
{
    struct DimX
    {
        static constexpr bool PERIODIC = IsPeriodic;
    };

    struct DDimX : ddc::NonUniformPointSampling<DimX>
    {
    };

    static constexpr ddc::BoundCond Bc
            = IsPeriodic ? ddc::BoundCond::PERIODIC : ddc::BoundCond::HERMITE;

    struct BSplinesX : ddc::NonUniformBSplines<DimX, 2>
    {
    };
};

} // namespace

using periodicity = std::integer_sequence<bool, true, false>;

using Cases = tuple_to_types_t<cartesian_product_t<periodicity>>;

// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(UniformBSplinesFixture, Cases, );

TYPED_TEST(UniformBSplinesFixture, KnotsAsInterpolationPoints)
{
    using DimX = typename TestFixture::DimX;
    using DDimX = typename TestFixture::DDimX;
    using BSplinesX = typename TestFixture::BSplinesX;
    using CoordX = ddc::Coordinate<DimX>;
    constexpr ddc::BoundCond Bc = TestFixture::Bc;
    constexpr CoordX xmin(0.);
    constexpr CoordX xmax(1.);
    std::size_t const ncells = 20;
    ddc::init_discrete_space<BSplinesX>(xmin, xmax, ncells);
    ddc::init_discrete_space<DDimX>(
            ddc::KnotsAsInterpolationPoints<BSplinesX, Bc, Bc>::template get_sampling<DDimX>());
    auto const interp_points_dom
            = ddc::KnotsAsInterpolationPoints<BSplinesX, Bc, Bc>::template get_domain<DDimX>();
    auto break_points_dom = ddc::discrete_space<BSplinesX>().break_point_domain();
    if (BSplinesX::is_periodic()) {
        break_points_dom = break_points_dom.remove_last(
                ddc::DiscreteVector<ddc::knot_discrete_dimension_t<BSplinesX>>(1));
    }
    ASSERT_EQ(interp_points_dom.size(), break_points_dom.size());
    for (int i = 0; i != interp_points_dom.size(); ++i) {
        EXPECT_DOUBLE_EQ(
                ddc::coordinate(interp_points_dom[i]),
                ddc::coordinate(break_points_dom[i]));
    }
}

// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(NonUniformBSplinesFixture, Cases, );

TYPED_TEST(NonUniformBSplinesFixture, KnotsAsInterpolationPoints)
{
    using DimX = typename TestFixture::DimX;
    using DDimX = typename TestFixture::DDimX;
    using BSplinesX = typename TestFixture::BSplinesX;
    using CoordX = ddc::Coordinate<DimX>;
    constexpr ddc::BoundCond Bc = TestFixture::Bc;
    constexpr CoordX xmin(0.);
    constexpr CoordX xmax(1.);
    std::size_t const ncells = 20;
    std::vector<CoordX> breaks(ncells + 1);
    double const dx = (xmax - xmin) / ncells;
    for (std::size_t i = 0; i < ncells + 1; ++i) {
        breaks[i] = CoordX(xmin + i * dx);
    }
    ddc::init_discrete_space<BSplinesX>(breaks);
    ddc::init_discrete_space<DDimX>(
            ddc::KnotsAsInterpolationPoints<BSplinesX, Bc, Bc>::template get_sampling<DDimX>());
    auto const interp_points_dom
            = ddc::KnotsAsInterpolationPoints<BSplinesX, Bc, Bc>::template get_domain<DDimX>();
    auto break_points_dom = ddc::discrete_space<BSplinesX>().break_point_domain();
    if (BSplinesX::is_periodic()) {
        break_points_dom = break_points_dom.remove_last(
                ddc::DiscreteVector<ddc::knot_discrete_dimension_t<BSplinesX>>(1));
    }
    ASSERT_EQ(interp_points_dom.size(), break_points_dom.size());
    for (int i = 0; i != interp_points_dom.size(); ++i) {
        EXPECT_DOUBLE_EQ(
                ddc::coordinate(interp_points_dom[i]),
                ddc::coordinate(break_points_dom[i]));
    }
}
