// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <stdexcept>
#include <string>

#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

TEST(SplineBoundaryConditions, StreamOperator)
{
    std::stringstream ss;
    ss << ddc::BoundCond::GREVILLE;
    EXPECT_EQ("GREVILLE", ss.str());

    ss.str("");
    ss << ddc::BoundCond::HERMITE;
    EXPECT_EQ("HERMITE", ss.str());

    ss.str("");
    ss << ddc::BoundCond::HOMOGENEOUS_HERMITE;
    EXPECT_EQ("HOMOGENEOUS_HERMITE", ss.str());

    ss.str("");
    ss << ddc::BoundCond::PERIODIC;
    EXPECT_EQ("PERIODIC", ss.str());

    ss.str("");
    EXPECT_THROW((ss << static_cast<ddc::BoundCond>(-1)), std::runtime_error);
}
