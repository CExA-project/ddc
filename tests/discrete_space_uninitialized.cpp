// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>

#include <gtest/gtest.h>

import ddc.core;
import ddc.core.detail;

TEST(DiscreteSpace, UninitializedDisplayDiscretizationStore)
{
    std::stringstream ss;
    ddc::detail::display_discretization_store(ss);
    EXPECT_EQ(ss.str(), "The host discretization store is not initialized:\n");
}
