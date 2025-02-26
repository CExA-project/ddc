// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include "sum.hpp"

TEST(KokkosSumVersionA, Small)
{
    int const n = 10;

    Kokkos::View<int*, Kokkos::LayoutRight> view("", n);
    Kokkos::deep_copy(view, 1);

    EXPECT_EQ(sum(view), n);
}

TEST(KokkosSumVersionA, Bigger)
{
    int const n = 1000;

    Kokkos::View<int*, Kokkos::LayoutRight> view("", n);
    Kokkos::deep_copy(view, 1);

    EXPECT_EQ(sum(view), n);
}
