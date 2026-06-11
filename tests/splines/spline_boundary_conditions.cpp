// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <stdexcept>

#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

TEST(SplineBoundaryConditions, StreamOperator)
{
    std::stringstream ss;
    ss << ddc::SplineBuilderClosure::GREVILLE;
    EXPECT_EQ("GREVILLE", ss.str());

    ss.str("");
    ss << ddc::SplineBuilderClosure::HERMITE;
    EXPECT_EQ("HERMITE", ss.str());

    ss.str("");
    ss << ddc::SplineBuilderClosure::HOMOGENEOUS_HERMITE;
    EXPECT_EQ("HOMOGENEOUS_HERMITE", ss.str());

    ss.str("");
    ss << ddc::SplineBuilderClosure::PERIODIC;
    EXPECT_EQ("PERIODIC", ss.str());

    ss.str("");
    EXPECT_THROW((ss << static_cast<ddc::SplineBuilderClosure>(-1)), std::runtime_error);
}
