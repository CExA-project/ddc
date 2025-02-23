// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "quadrature.hpp"

TEST(Quadrature, VersionB)
{
    ddc::DiscreteDomain<GridVx>
            gridvx(ddc::DiscreteElement<GridVx>(0), ddc::DiscreteVector<GridVx>(100));

    ddc::Chunk quadrature_coeffs(gridvx, ddc::DeviceAllocator<double>());
    ddc::parallel_fill(quadrature_coeffs, 3.);
    Quadrature const integrate_v(quadrature_coeffs.span_cview());

    ddc::Chunk fdistribu_alloc(gridvx, ddc::DeviceAllocator<double>());
    ddc::parallel_fill(fdistribu_alloc.span_view(), 2);

    // density
    double const density_res
            = integrate_v(Kokkos::DefaultExecutionSpace(), fdistribu_alloc.span_view());
}
