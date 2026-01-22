// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#define DDC_TESTS_MAIN                                                                             \
    int main(int argc, char** argv)                                                                \
    {                                                                                              \
        ::testing::InitGoogleTest(&argc, argv);                                                    \
        Kokkos::ScopeGuard const kokkos_scope(argc, argv);                                         \
        ddc::ScopeGuard const ddc_scope(argc, argv);                                               \
        return RUN_ALL_TESTS();                                                                    \
    }
