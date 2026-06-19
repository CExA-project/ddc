// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "test_utils.hpp"

inline namespace anonymous_namespace_workaround_spline_traits_cpp {

struct DimX
{
};

struct DDimX : ddc::NonUniformPointSampling<DimX>
{
};

struct DimY
{
};

struct DDimY : ddc::NonUniformPointSampling<DimY>
{
};

struct DimZ
{
};

struct DDimZ : ddc::NonUniformPointSampling<DimZ>
{
};

template <typename T>
struct BSplinesTraits
{
};

template <
        typename ExecSpace1,
        std::size_t D1,
        typename ExecSpace2,
        std::size_t D2,
        bool LegacyEvaluator>
struct BSplinesTraits<std::tuple<
        ExecSpace1,
        std::integral_constant<std::size_t, D1>,
        ExecSpace2,
        std::integral_constant<std::size_t, D2>,
        std::bool_constant<LegacyEvaluator>>> : public ::testing::Test
{
    using execution_space1 = ExecSpace1;
    using execution_space2 = ExecSpace2;
    using memory_space1 = ExecSpace1::memory_space;
    using memory_space2 = ExecSpace2::memory_space;
    static constexpr std::size_t m_spline_degree1 = D1;
    static constexpr std::size_t m_spline_degree2 = D2;

    struct BSplinesX1 : ddc::UniformBSplines<DimX, D1, true>
    {
    };

    struct BSplinesX2 : ddc::UniformBSplines<DimX, D2, true>
    {
    };

    struct BSplinesY : ddc::UniformBSplines<DimY, D1, true>
    {
    };

    struct BSplinesZ : ddc::UniformBSplines<DimZ, D1, true>
    {
    };

    using Builder1D_1 = ddc::SplineBuilder<
            execution_space1,
            memory_space1,
            BSplinesX1,
            DDimX,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC>;

    using Evaluator1D_1 = std::conditional_t<
            LegacyEvaluator,
            ddc::SplineEvaluator<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    DDimX,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>>,
            ddc::SplineEvaluatorND<
                    execution_space1,
                    memory_space1,
                    ddc::TypeSeq<BSplinesX1>,
                    ddc::TypeSeq<DDimX>,
                    ddc::TypeSeq<
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimX>>>>;

    using Builder1D_2 = ddc::SplineBuilder<
            execution_space2,
            memory_space2,
            BSplinesX2,
            DDimX,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC>;

    using Evaluator1D_2 = std::conditional_t<
            LegacyEvaluator,
            ddc::SplineEvaluator<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    DDimX,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>>,
            ddc::SplineEvaluatorND<
                    execution_space2,
                    memory_space2,
                    ddc::TypeSeq<BSplinesX2>,
                    ddc::TypeSeq<DDimX>,
                    ddc::TypeSeq<
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimX>>>>;

    using Builder2D_1 = ddc::SplineBuilder2D<
            execution_space1,
            memory_space1,
            BSplinesX1,
            BSplinesY,
            DDimX,
            DDimY,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC>;

    using Evaluator2D_1 = std::conditional_t<
            LegacyEvaluator,
            ddc::SplineEvaluator2D<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    BSplinesY,
                    DDimX,
                    DDimY,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimY>>,
            ddc::SplineEvaluatorND<
                    execution_space1,
                    memory_space1,
                    ddc::TypeSeq<BSplinesX1, BSplinesY>,
                    ddc::TypeSeq<DDimX, DDimY>,
                    ddc::TypeSeq<
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimY>,
                            ddc::PeriodicExtrapolationRule<DimY>>>>;

    using Builder2D_2 = ddc::SplineBuilder2D<
            execution_space2,
            memory_space2,
            BSplinesX2,
            BSplinesY,
            DDimX,
            DDimY,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC>;

    using Evaluator2D_2 = std::conditional_t<
            LegacyEvaluator,
            ddc::SplineEvaluator2D<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    BSplinesY,
                    DDimX,
                    DDimY,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimY>>,
            ddc::SplineEvaluatorND<
                    execution_space2,
                    memory_space2,
                    ddc::TypeSeq<BSplinesX2, BSplinesY>,
                    ddc::TypeSeq<DDimX, DDimY>,
                    ddc::TypeSeq<
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimY>,
                            ddc::PeriodicExtrapolationRule<DimY>>>>;

    using Builder3D_1 = ddc::SplineBuilder3D<
            execution_space1,
            memory_space1,
            BSplinesX1,
            BSplinesY,
            BSplinesZ,
            DDimX,
            DDimY,
            DDimZ,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineSolver::LAPACK>;

    using Evaluator3D_1 = std::conditional_t<
            LegacyEvaluator,
            ddc::SplineEvaluator3D<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    BSplinesY,
                    BSplinesZ,
                    DDimX,
                    DDimY,
                    DDimZ,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimZ>,
                    ddc::PeriodicExtrapolationRule<DimZ>>,
            ddc::SplineEvaluatorND<
                    execution_space1,
                    memory_space1,
                    ddc::TypeSeq<BSplinesX1, BSplinesY, BSplinesZ>,
                    ddc::TypeSeq<DDimX, DDimY, DDimZ>,
                    ddc::TypeSeq<
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimY>,
                            ddc::PeriodicExtrapolationRule<DimY>,
                            ddc::PeriodicExtrapolationRule<DimZ>,
                            ddc::PeriodicExtrapolationRule<DimZ>>>>;

    using Builder3D_2 = ddc::SplineBuilder3D<
            execution_space2,
            memory_space2,
            BSplinesX2,
            BSplinesY,
            BSplinesZ,
            DDimX,
            DDimY,
            DDimZ,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineBuilderClosure::PERIODIC,
            ddc::SplineSolver::LAPACK>;

    using Evaluator3D_2 = std::conditional_t<
            LegacyEvaluator,
            ddc::SplineEvaluator3D<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    BSplinesY,
                    BSplinesZ,
                    DDimX,
                    DDimY,
                    DDimZ,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimZ>,
                    ddc::PeriodicExtrapolationRule<DimZ>>,
            ddc::SplineEvaluatorND<
                    execution_space2,
                    memory_space2,
                    ddc::TypeSeq<BSplinesX2, BSplinesY, BSplinesZ>,
                    ddc::TypeSeq<DDimX, DDimY, DDimZ>,
                    ddc::TypeSeq<
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimX>,
                            ddc::PeriodicExtrapolationRule<DimY>,
                            ddc::PeriodicExtrapolationRule<DimY>,
                            ddc::PeriodicExtrapolationRule<DimZ>,
                            ddc::PeriodicExtrapolationRule<DimZ>>>>;
};

struct BSplinesTraitsNames
{
    template <typename T>
    static std::string GetName(int const i)
    {
        std::stringstream ss;
        ss << "(ExecSpace1=" << std::tuple_element_t<0, T>::name();
        ss << ",Degree1=" << std::tuple_element_t<1, T>::value;
        ss << ",ExecSpace2=" << std::tuple_element_t<2, T>::name();
        ss << ",Degree2=" << std::tuple_element_t<3, T>::value;
        ss << ",LegacyEvaluator=" << std::tuple_element_t<4, T>::value;
        ss << ")/" << i;

        return ss.str();
    }
};

#if defined(KOKKOS_ENABLE_SERIAL)
using execution_space_types = std::
        tuple<Kokkos::Serial, Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
#else
using execution_space_types
        = std::tuple<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
#endif

using spline_degrees = std::integer_sequence<std::size_t, 2, 3>;

using TestTypes = tuple_to_types_t<cartesian_product_t<
        execution_space_types,
        spline_degrees,
        execution_space_types,
        spline_degrees,
        std::tuple<std::false_type, std::true_type>>>;

} // namespace anonymous_namespace_workaround_spline_traits_cpp

TYPED_TEST_SUITE(BSplinesTraits, TestTypes, BSplinesTraitsNames);

TYPED_TEST(BSplinesTraits, IsSplineBuilder)
{
    using Builder1D = TestFixture::Builder1D_1;
    using Evaluator1D = TestFixture::Evaluator1D_1;
    using Builder2D = TestFixture::Builder2D_1;
    using Evaluator2D = TestFixture::Evaluator2D_1;
    EXPECT_TRUE(ddc::is_spline_builder_v<Builder1D>);
    EXPECT_FALSE(ddc::is_spline_builder_v<Builder2D>);
    EXPECT_FALSE(ddc::is_spline_builder_v<Evaluator1D>);
    EXPECT_FALSE(ddc::is_spline_builder_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineBuilder2D)
{
    using Builder1D = TestFixture::Builder1D_1;
    using Evaluator1D = TestFixture::Evaluator1D_1;
    using Builder2D = TestFixture::Builder2D_1;
    using Evaluator2D = TestFixture::Evaluator2D_1;
    EXPECT_FALSE(ddc::is_spline_builder2d_v<Builder1D>);
    EXPECT_TRUE(ddc::is_spline_builder2d_v<Builder2D>);
    EXPECT_FALSE(ddc::is_spline_builder2d_v<Evaluator1D>);
    EXPECT_FALSE(ddc::is_spline_builder2d_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineBuilder3D)
{
    using Builder1D = TestFixture::Builder1D_1;
    using Evaluator1D = TestFixture::Evaluator1D_1;
    using Builder3D = TestFixture::Builder3D_1;
    using Evaluator3D = TestFixture::Evaluator3D_1;
    EXPECT_FALSE(ddc::is_spline_builder3d_v<Builder1D>);
    EXPECT_TRUE(ddc::is_spline_builder3d_v<Builder3D>);
    EXPECT_FALSE(ddc::is_spline_builder3d_v<Evaluator1D>);
    EXPECT_FALSE(ddc::is_spline_builder3d_v<Evaluator3D>);
}

TYPED_TEST(BSplinesTraits, IsSplineEvaluator)
{
    using Builder1D = TestFixture::Builder1D_1;
    using Evaluator1D = TestFixture::Evaluator1D_1;
    using Builder2D = TestFixture::Builder2D_1;
    using Evaluator2D = TestFixture::Evaluator2D_1;
    EXPECT_FALSE(ddc::is_spline_evaluator_v<Builder1D>);
    EXPECT_FALSE(ddc::is_spline_evaluator_v<Builder2D>);
    EXPECT_TRUE(ddc::is_spline_evaluator_v<Evaluator1D>);
    EXPECT_FALSE(ddc::is_spline_evaluator_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineEvaluator2D)
{
    using Builder1D = TestFixture::Builder1D_1;
    using Evaluator1D = TestFixture::Evaluator1D_1;
    using Builder2D = TestFixture::Builder2D_1;
    using Evaluator2D = TestFixture::Evaluator2D_1;
    EXPECT_FALSE(ddc::is_spline_evaluator2d_v<Builder1D>);
    EXPECT_FALSE(ddc::is_spline_evaluator2d_v<Builder2D>);
    EXPECT_FALSE(ddc::is_spline_evaluator2d_v<Evaluator1D>);
    EXPECT_TRUE(ddc::is_spline_evaluator2d_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineEvaluator3D)
{
    using Builder1D = TestFixture::Builder1D_1;
    using Evaluator1D = TestFixture::Evaluator1D_1;
    using Builder3D = TestFixture::Builder3D_1;
    using Evaluator3D = TestFixture::Evaluator3D_1;
    EXPECT_FALSE(ddc::is_spline_evaluator3d_v<Builder1D>);
    EXPECT_FALSE(ddc::is_spline_evaluator3d_v<Builder3D>);
    EXPECT_FALSE(ddc::is_spline_evaluator3d_v<Evaluator1D>);
    EXPECT_TRUE(ddc::is_spline_evaluator3d_v<Evaluator3D>);
}

TYPED_TEST(BSplinesTraits, IsAdmissible1D)
{
    using Builder1D_1 = TestFixture::Builder1D_1;
    using Evaluator1D_1 = TestFixture::Evaluator1D_1;
    using Builder1D_2 = TestFixture::Builder1D_2;
    using Evaluator1D_2 = TestFixture::Evaluator1D_2;

    // Builders are not compatible
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder1D_1, Builder1D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder1D_1, Builder1D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder1D_2, Builder1D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder1D_2, Builder1D_2>));

    // Evaluators are not compatible
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_1, Evaluator1D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_1, Evaluator1D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_2, Evaluator1D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_2, Evaluator1D_2>));

    // Compatible builder and evaluator pairs
    EXPECT_TRUE((ddc::is_evaluator_admissible_v<Builder1D_1, Evaluator1D_1>));
    EXPECT_TRUE((ddc::is_evaluator_admissible_v<Builder1D_2, Evaluator1D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_1, Builder1D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_2, Builder1D_2>));

    // Incompatible builder and evaluator pairs
    using execution_space1 = TestFixture::execution_space1;
    using execution_space2 = TestFixture::execution_space2;
    std::size_t constexpr m_spline_degree1 = TestFixture::m_spline_degree1;
    std::size_t constexpr m_spline_degree2 = TestFixture::m_spline_degree2;

    if ((!std::is_same_v<execution_space1, execution_space2>)
        || (m_spline_degree1 != m_spline_degree2)) {
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder1D_1, Evaluator1D_2>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_2, Builder1D_1>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder1D_2, Evaluator1D_1>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator1D_1, Builder1D_2>));
    }
}

TYPED_TEST(BSplinesTraits, IsAdmissible2D)
{
    using Builder2D_1 = TestFixture::Builder2D_1;
    using Evaluator2D_1 = TestFixture::Evaluator2D_1;
    using Builder2D_2 = TestFixture::Builder2D_2;
    using Evaluator2D_2 = TestFixture::Evaluator2D_2;

    // Builders are not compatible
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder2D_1, Builder2D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder2D_1, Builder2D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder2D_2, Builder2D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder2D_2, Builder2D_2>));

    // Evaluators are not compatible
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_1, Evaluator2D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_1, Evaluator2D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_2, Evaluator2D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_2, Evaluator2D_2>));

    // Compatible builder and evaluator pairs
    EXPECT_TRUE((ddc::is_evaluator_admissible_v<Builder2D_1, Evaluator2D_1>));
    EXPECT_TRUE((ddc::is_evaluator_admissible_v<Builder2D_2, Evaluator2D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_1, Builder2D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_2, Builder2D_2>));

    // Incompatible builder and evaluator pairs
    using execution_space1 = TestFixture::execution_space1;
    using execution_space2 = TestFixture::execution_space2;
    std::size_t constexpr m_spline_degree1 = TestFixture::m_spline_degree1;
    std::size_t constexpr m_spline_degree2 = TestFixture::m_spline_degree2;

    if ((!std::is_same_v<execution_space1, execution_space2>)
        || (m_spline_degree1 != m_spline_degree2)) {
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder2D_1, Evaluator2D_2>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_2, Builder2D_1>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder2D_2, Evaluator2D_1>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator2D_1, Builder2D_2>));
    }
}

TYPED_TEST(BSplinesTraits, IsAdmissible3D)
{
    using Builder3D_1 = TestFixture::Builder3D_1;
    using Evaluator3D_1 = TestFixture::Evaluator3D_1;
    using Builder3D_2 = TestFixture::Builder3D_2;
    using Evaluator3D_2 = TestFixture::Evaluator3D_2;

    // Builders are not compatible
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder3D_1, Builder3D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder3D_1, Builder3D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder3D_2, Builder3D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder3D_2, Builder3D_2>));

    // Evaluators are not compatible
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_1, Evaluator3D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_1, Evaluator3D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_2, Evaluator3D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_2, Evaluator3D_2>));

    // Compatible builder and evaluator pairs
    EXPECT_TRUE((ddc::is_evaluator_admissible_v<Builder3D_1, Evaluator3D_1>));
    EXPECT_TRUE((ddc::is_evaluator_admissible_v<Builder3D_2, Evaluator3D_2>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_1, Builder3D_1>));
    EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_2, Builder3D_2>));

    // Incompatible builder and evaluator pairs
    using execution_space1 = TestFixture::execution_space1;
    using execution_space2 = TestFixture::execution_space2;
    std::size_t constexpr m_spline_degree1 = TestFixture::m_spline_degree1;
    std::size_t constexpr m_spline_degree2 = TestFixture::m_spline_degree2;

    if ((!std::is_same_v<execution_space1, execution_space2>)
        || (m_spline_degree1 != m_spline_degree2)) {
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder3D_1, Evaluator3D_2>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_2, Builder3D_1>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Builder3D_2, Evaluator3D_1>));
        EXPECT_FALSE((ddc::is_evaluator_admissible_v<Evaluator3D_1, Builder3D_2>));
    }
}
