// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

TEST(DdcToKokkosExecutionPolicy, Dim0)
{
    Kokkos::DefaultExecutionSpace const space;
    auto const kokkos_range = ddc::detail::
            ddc_to_kokkos_execution_policy(space, std::array<ddc::DiscreteVectorElement, 0> {});
    EXPECT_EQ(kokkos_range.space(), space);
    EXPECT_EQ(kokkos_range.begin(), 0);
    EXPECT_EQ(kokkos_range.end(), 1);
}

TEST(DdcToKokkosExecutionPolicy, Dim1)
{
    ddc::DiscreteVectorElement const n0 = 3;
    Kokkos::DefaultExecutionSpace const space;
    auto const kokkos_range = ddc::detail::
            ddc_to_kokkos_execution_policy(space, std::array<ddc::DiscreteVectorElement, 1> {n0});
    EXPECT_EQ(kokkos_range.space(), space);
    EXPECT_EQ(kokkos_range.begin(), 0);
    EXPECT_EQ(kokkos_range.end(), n0);
}

TEST(DdcToKokkosExecutionPolicy, Dim2)
{
    ddc::DiscreteVectorElement const n0 = 3;
    ddc::DiscreteVectorElement const n1 = 4;
    Kokkos::DefaultExecutionSpace const space;
    auto const kokkos_range = ddc::detail::ddc_to_kokkos_execution_policy(
            space,
            std::array<ddc::DiscreteVectorElement, 2> {n0, n1});
    EXPECT_EQ(kokkos_range.space(), space);
    EXPECT_EQ(kokkos_range.m_lower[0], 0);
    EXPECT_EQ(kokkos_range.m_lower[1], 0);
    EXPECT_EQ(kokkos_range.m_upper[0], n0);
    EXPECT_EQ(kokkos_range.m_upper[1], n1);
}
