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

struct DimX
{
    static constexpr bool PERIODIC = true;
};

using CoordX = ddc::Coordinate<DimX>;

static constexpr std::size_t s_degree_x = 2;

struct ShortBSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};

struct ShortDDimX : ddc::NonUniformPointSampling<DimX>
{
};

struct LongBSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};

struct LongDDimX : ddc::NonUniformPointSampling<DimX>
{
};

struct BadBSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};

struct BadDDimX : ddc::NonUniformPointSampling<DimX>
{
};

struct CorrectBSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};

struct CorrectDDimX : ddc::NonUniformPointSampling<DimX>
{
};


using execution_space = Kokkos::DefaultHostExecutionSpace;
using memory_space = Kokkos::HostSpace;

TEST(SplineBuilder, ShortInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<ShortBSplinesX>(x0, xN, ncells);

    // One point missing
    std::vector<double> const range {0.1, 0.3, 0.5, 0.7};

    ddc::DiscreteDomain<ShortDDimX> const interpolation_domain
            = ddc::init_discrete_space<ShortDDimX>(ShortDDimX::init<ShortDDimX>(range));

    EXPECT_THROW(
            (ddc::SplineBuilder<
                    execution_space,
                    memory_space,
                    ShortBSplinesX,
                    ShortDDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::GINKGO,
                    ShortDDimX>(interpolation_domain)),
            std::runtime_error);
}

TEST(SplineBuilder, LongInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<LongBSplinesX>(x0, xN, ncells);

    // One point too much
    std::vector<double> const range {0.1, 0.3, 0.5, 0.7, 0.9, 0.95};

    ddc::DiscreteDomain<LongDDimX> const interpolation_domain
            = ddc::init_discrete_space<LongDDimX>(LongDDimX::init<LongDDimX>(range));

    EXPECT_THROW(
            (ddc::SplineBuilder<
                    execution_space,
                    memory_space,
                    LongBSplinesX,
                    LongDDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::GINKGO,
                    LongDDimX>(interpolation_domain)),
            std::runtime_error);
}

TEST(SplineBuilder, BadShapeInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<BadBSplinesX>(x0, xN, ncells);

    // All points end up in the first cell ]0, 0.2[
    std::vector<double> const range {0.1, 0.11, 0.12, 0.13, 0.14};

    ddc::DiscreteDomain<BadDDimX> const interpolation_domain
            = ddc::init_discrete_space<BadDDimX>(BadDDimX::init<BadDDimX>(range));

    EXPECT_THROW(
            (ddc::SplineBuilder<
                    execution_space,
                    memory_space,
                    BadBSplinesX,
                    BadDDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::GINKGO,
                    BadDDimX>(interpolation_domain)),
            std::runtime_error);
}

TEST(SplineBuilder, CorrectInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<CorrectBSplinesX>(x0, xN, ncells);

    std::vector<double> const range {0.05, 0.15, 0.5, 0.85, 0.95};

    ddc::DiscreteDomain<CorrectDDimX> const interpolation_domain
            = ddc::init_discrete_space<CorrectDDimX>(CorrectDDimX::init<CorrectDDimX>(range));

    EXPECT_NO_THROW((ddc::SplineBuilder<
                     execution_space,
                     memory_space,
                     CorrectBSplinesX,
                     CorrectDDimX,
                     ddc::BoundCond::PERIODIC,
                     ddc::BoundCond::PERIODIC,
                     ddc::SplineSolver::GINKGO,
                     CorrectDDimX>(interpolation_domain)));
}
