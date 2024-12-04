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

struct BSplinesX : ddc::UniformBSplines<DimX, s_degree_x>
{
};

struct IDimX : ddc::NonUniformPointSampling<DimX>
{
};

using execution_space = Kokkos::DefaultHostExecutionSpace;
using memory_space = Kokkos::HostSpace;

TEST(SplineBuilder, ShortInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<BSplinesX>(x0, xN, ncells);

    // One point missing
    std::vector<double> const range {0.1, 0.3, 0.5, 0.7};

    ddc::DiscreteDomain<IDimX> const interpolation_domain
            = ddc::init_discrete_space<IDimX>(IDimX::init<IDimX>(range));

    EXPECT_THROW(
            (ddc::SplineBuilder<
                    execution_space,
                    memory_space,
                    BSplinesX,
                    IDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::GINKGO,
                    IDimX>(interpolation_domain)),
            std::runtime_error);
}

TEST(SplineBuilder, LongInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<BSplinesX>(x0, xN, ncells);

    // One point too much
    std::vector<double> const range {0.1, 0.3, 0.5, 0.7, 0.9, 0.95};

    ddc::DiscreteDomain<IDimX> const interpolation_domain
            = ddc::init_discrete_space<IDimX>(IDimX::init<IDimX>(range));

    EXPECT_THROW(
            (ddc::SplineBuilder<
                    execution_space,
                    memory_space,
                    BSplinesX,
                    IDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::GINKGO,
                    IDimX>(interpolation_domain)),
            std::runtime_error);
}

TEST(SplineBuilder, BadShapeInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<BSplinesX>(x0, xN, ncells);

    // All points end up in the first cell ]0, 0.2[
    std::vector<double> const range {0.1, 0.11, 0.12, 0.13, 0.14};

    ddc::DiscreteDomain<IDimX> const interpolation_domain
            = ddc::init_discrete_space<IDimX>(IDimX::init<IDimX>(range));

    EXPECT_THROW(
            (ddc::SplineBuilder<
                    execution_space,
                    memory_space,
                    BSplinesX,
                    IDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC,
                    ddc::SplineSolver::GINKGO,
                    IDimX>(interpolation_domain)),
            std::runtime_error);
}

TEST(SplineBuilder, CorrectInterpolationGrid)
{
    CoordX const x0(0.);
    CoordX const xN(1.);
    std::size_t const ncells = 5;

    ddc::init_discrete_space<BSplinesX>(x0, xN, ncells);

    std::vector<double> const range {0.05, 0.15, 0.5, 0.85, 0.95};

    ddc::DiscreteDomain<IDimX> const interpolation_domain
            = ddc::init_discrete_space<IDimX>(IDimX::init<IDimX>(range));

    EXPECT_NO_THROW((ddc::SplineBuilder<
                     execution_space,
                     memory_space,
                     BSplinesX,
                     IDimX,
                     ddc::BoundCond::PERIODIC,
                     ddc::BoundCond::PERIODIC,
                     ddc::SplineSolver::GINKGO,
                     IDimX>(interpolation_domain)));
}
