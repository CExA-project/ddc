// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include "test_utils.hpp"

struct DimX
{
    static constexpr bool PERIODIC = true;
};

struct DDimX : ddc::NonUniformPointSampling<DimX>
{
};

struct DimY
{
    static constexpr bool PERIODIC = true;
};

struct DDimY : ddc::NonUniformPointSampling<DimY>
{
};

template <typename T>
struct BSplinesTraits;

template <typename ExecSpace1, std::size_t D1, typename ExecSpace2, std::size_t D2>
struct BSplinesTraits<std::tuple<ExecSpace1, std::integral_constant<std::size_t, D1>, ExecSpace2, std::integral_constant<std::size_t, D2>>> : public ::testing::Test {
  using execution_space1 = ExecSpace1;
  using execution_space2 = ExecSpace2;
  using memory_space1 = typename ExecSpace1::memory_space;
  using memory_space2 = typename ExecSpace2::memory_space;
  static constexpr std::size_t m_spline_degree1 = D1;
  static constexpr std::size_t m_spline_degree2 = D2;

  struct BSplinesX1 : ddc::UniformBSplines<DimX, D1>
  {
  };

  struct BSplinesX2 : ddc::UniformBSplines<DimX, D2>
  {
  };

  struct BSplinesY : ddc::UniformBSplines<DimY, D1>
  {
  };
  
  using Builder1D_1 = ddc::SplineBuilder<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    DDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::LAPACK>;

  using Evaluator1D_1 = ddc::SplineEvaluator<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    DDimX,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>>;

  using Builder1D_2 = ddc::SplineBuilder<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    DDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::LAPACK>;

  using Evaluator1D_2 = ddc::SplineEvaluator<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    DDimX,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>>;

  using Builder2D_1 = ddc::SplineBuilder2D<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    BSplinesY,
                    DDimX,
                    DDimY,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::LAPACK>;

  using Evaluator2D_1 = ddc::SplineEvaluator2D<
                    execution_space1,
                    memory_space1,
                    BSplinesX1,
                    BSplinesY,
                    DDimX,
                    DDimY,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimY>>;

  using Builder2D_2 = ddc::SplineBuilder2D<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    BSplinesY,
                    DDimX,
                    DDimY,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::LAPACK>;

  using Evaluator2D_2 = ddc::SplineEvaluator2D<
                    execution_space2,
                    memory_space2,
                    BSplinesX2,
                    BSplinesY,
                    DDimX,
                    DDimY,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimX>,
                    ddc::PeriodicExtrapolationRule<DimY>,
                    ddc::PeriodicExtrapolationRule<DimY>>;
};

#if defined(KOKKOS_ENABLE_SERIAL)
using execution_space_types = std::tuple<Kokkos::Serial, Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
#else
using execution_space_types = std::tuple<Kokkos::DefaultHostExecutionSpace, Kokkos::DefaultExecutionSpace>;
#endif

using spline_degrees = std::integer_sequence<std::size_t, 2, 3>;

using TestTypes = tuple_to_types_t<cartesian_product_t<execution_space_types, spline_degrees, execution_space_types, spline_degrees>>;

TYPED_TEST_SUITE(BSplinesTraits, TestTypes, );

TYPED_TEST(BSplinesTraits, IsSplineBuilder)
{
    using Builder1D = typename TestFixture::Builder1D_1;
    using Evaluator1D = typename TestFixture::Evaluator1D_1;
    using Builder2D = typename TestFixture::Builder2D_1;
    using Evaluator2D = typename TestFixture::Evaluator2D_1;
    ASSERT_TRUE(ddc::is_spline_builder_v<Builder1D>);
    ASSERT_FALSE(ddc::is_spline_builder_v<Builder2D>);
    ASSERT_FALSE(ddc::is_spline_builder_v<Evaluator1D>);
    ASSERT_FALSE(ddc::is_spline_builder_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineBuilder2D)
{
    using Builder1D = typename TestFixture::Builder1D_1;
    using Evaluator1D = typename TestFixture::Evaluator1D_1;
    using Builder2D = typename TestFixture::Builder2D_1;
    using Evaluator2D = typename TestFixture::Evaluator2D_1;
    ASSERT_FALSE(ddc::is_spline_builder2D_v<Builder1D>);
    ASSERT_TRUE(ddc::is_spline_builder2D_v<Builder2D>);
    ASSERT_FALSE(ddc::is_spline_builder2D_v<Evaluator1D>);
    ASSERT_FALSE(ddc::is_spline_builder2D_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineEvaluator)
{
    using Builder1D = typename TestFixture::Builder1D_1;
    using Evaluator1D = typename TestFixture::Evaluator1D_1;
    using Builder2D = typename TestFixture::Builder2D_1;
    using Evaluator2D = typename TestFixture::Evaluator2D_1;
    ASSERT_FALSE(ddc::is_spline_evaluator_v<Builder1D>);
    ASSERT_FALSE(ddc::is_spline_evaluator_v<Builder2D>);
    ASSERT_TRUE(ddc::is_spline_evaluator_v<Evaluator1D>);
    ASSERT_FALSE(ddc::is_spline_evaluator_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsSplineEvaluator2D)
{
    using Builder1D = typename TestFixture::Builder1D_1;
    using Evaluator1D = typename TestFixture::Evaluator1D_1;
    using Builder2D = typename TestFixture::Builder2D_1;
    using Evaluator2D = typename TestFixture::Evaluator2D_1;
    ASSERT_FALSE(ddc::is_spline_evaluator2D_v<Builder1D>);
    ASSERT_FALSE(ddc::is_spline_evaluator2D_v<Builder2D>);
    ASSERT_FALSE(ddc::is_spline_evaluator2D_v<Evaluator1D>);
    ASSERT_TRUE(ddc::is_spline_evaluator2D_v<Evaluator2D>);
}

TYPED_TEST(BSplinesTraits, IsCompatible1D)
{
    using Builder1D_1 = typename TestFixture::Builder1D_1;
    using Evaluator1D_1 = typename TestFixture::Evaluator1D_1;
    using Builder1D_2 = typename TestFixture::Builder1D_2;
    using Evaluator1D_2 = typename TestFixture::Evaluator1D_2;

    // Builders are not compatible
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder1D_1, Builder1D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder1D_1, Builder1D_2>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder1D_2, Builder1D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder1D_2, Builder1D_2>));

    // Evaluators are not compatible
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator1D_1, Evaluator1D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator1D_1, Evaluator1D_2>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator1D_2, Evaluator1D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator1D_2, Evaluator1D_2>));

    // Compatible builder and evaluator pairs
    ASSERT_TRUE((ddc::is_spline_compatible_v<Builder1D_1, Evaluator1D_1>));
    ASSERT_TRUE((ddc::is_spline_compatible_v<Evaluator1D_1, Builder1D_1>));
    ASSERT_TRUE((ddc::is_spline_compatible_v<Builder1D_2, Evaluator1D_2>));
    ASSERT_TRUE((ddc::is_spline_compatible_v<Evaluator1D_2, Builder1D_2>));

    // Incompatible builder and evaluator pairs
    using execution_space1 = typename TestFixture::execution_space1;
    using execution_space2 = typename TestFixture::execution_space2;
    std::size_t constexpr m_spline_degree1 = TestFixture::m_spline_degree1;
    std::size_t constexpr m_spline_degree2 = TestFixture::m_spline_degree2;

    if ((!std::is_same_v<execution_space1, execution_space2>) || (m_spline_degree1 != m_spline_degree2)) {
      ASSERT_FALSE((ddc::is_spline_compatible_v<Builder1D_1, Evaluator1D_2>));
      ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator1D_2, Builder1D_1>));
      ASSERT_FALSE((ddc::is_spline_compatible_v<Builder1D_2, Evaluator1D_1>));
      ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator1D_1, Builder1D_2>));
    }
}

TYPED_TEST(BSplinesTraits, IsCompatible2D)
{
    using Builder2D_1 = typename TestFixture::Builder2D_1;
    using Evaluator2D_1 = typename TestFixture::Evaluator2D_1;
    using Builder2D_2 = typename TestFixture::Builder2D_2;
    using Evaluator2D_2 = typename TestFixture::Evaluator2D_2;

    // Builders are not compatible
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder2D_1, Builder2D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder2D_1, Builder2D_2>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder2D_2, Builder2D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Builder2D_2, Builder2D_2>));

    // Evaluators are not compatible
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator2D_1, Evaluator2D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator2D_1, Evaluator2D_2>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator2D_2, Evaluator2D_1>));
    ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator2D_2, Evaluator2D_2>));

    // Compatible builder and evaluator pairs
    ASSERT_TRUE((ddc::is_spline_compatible_v<Builder2D_1, Evaluator2D_1>));
    ASSERT_TRUE((ddc::is_spline_compatible_v<Evaluator2D_1, Builder2D_1>));
    ASSERT_TRUE((ddc::is_spline_compatible_v<Builder2D_2, Evaluator2D_2>));
    ASSERT_TRUE((ddc::is_spline_compatible_v<Evaluator2D_2, Builder2D_2>));

    // Incompatible builder and evaluator pairs
    using execution_space1 = typename TestFixture::execution_space1;
    using execution_space2 = typename TestFixture::execution_space2;
    std::size_t constexpr m_spline_degree1 = TestFixture::m_spline_degree1;
    std::size_t constexpr m_spline_degree2 = TestFixture::m_spline_degree2;

    if ((!std::is_same_v<execution_space1, execution_space2>) || (m_spline_degree1 != m_spline_degree2)) {
      ASSERT_FALSE((ddc::is_spline_compatible_v<Builder2D_1, Evaluator2D_2>));
      ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator2D_2, Builder2D_1>));
      ASSERT_FALSE((ddc::is_spline_compatible_v<Builder2D_2, Evaluator2D_1>));
      ASSERT_FALSE((ddc::is_spline_compatible_v<Evaluator2D_1, Builder2D_2>));
    }
}

