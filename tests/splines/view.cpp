// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>

#include <ddc/kernels/splines.hpp>

#include <gtest/gtest.h>

TEST(View1DTest, Constructor)
{
    std::array<double, 10> x = {0};
    std::array<double const, 10> cx = {0};
    std::array<double, 10> const ccx = {0};
    std::array<double const, 10> const ccx2 = {0};
    ddc::Span1D<double> const xv(x.data(), x.size());
    [[maybe_unused]] ddc::Span1D<double> const xv_(xv);
    [[maybe_unused]] ddc::Span1D<double const> const xcv(xv);
    [[maybe_unused]] ddc::Span1D<double const> const xcv_(xcv);
    ddc::Span1D<double const> const cxcv(cx.data(), cx.size());
    [[maybe_unused]] ddc::Span1D<double const> const cxcv_(cxcv);
    ddc::Span1D<double const> const ccxcv(ccx.data(), ccx.size());
    [[maybe_unused]] ddc::Span1D<double const> const ccxcv_(cxcv);
    ddc::Span1D<double const> const ccx2cv(ccx2.data(), ccx2.size());
    [[maybe_unused]] ddc::Span1D<double const> const ccx2cv_(ccx2cv);
}
