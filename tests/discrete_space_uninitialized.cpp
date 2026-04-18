// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <string>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

TEST(DiscreteSpace, UninitializedDisplayDiscretizationStore)
{
    std::stringstream ss;
    ddc::detail::display_discretization_store(ss);
    EXPECT_EQ(ss.str(), "The host discretization store is not initialized:\n");
}
