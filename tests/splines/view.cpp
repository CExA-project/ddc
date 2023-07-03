#include <array>
#include <memory>
#include <sstream>

#include <experimental/mdspan>

#include <sll/view.hpp>

#include <gtest/gtest.h>

using namespace std;
using namespace std::experimental;

TEST(View1DTest, Constructor)
{
    std::array<double, 10> x = {0};
    std::array<double const, 10> cx = {0};
    std::array<double, 10> const ccx = {0};
    std::array<double const, 10> const ccx2 = {0};
    Span1D<double> xv(x.data(), x.size());
    [[maybe_unused]] Span1D<double> xv_(xv);
    [[maybe_unused]] Span1D<double const> xcv(xv);
    [[maybe_unused]] Span1D<double const> xcv_(xcv);
    Span1D<double const> cxcv(cx.data(), cx.size());
    [[maybe_unused]] Span1D<double const> cxcv_(cxcv);
    Span1D<double const> ccxcv(ccx.data(), ccx.size());
    [[maybe_unused]] Span1D<double const> ccxcv_(cxcv);
    Span1D<double const> ccx2cv(ccx2.data(), ccx2.size());
    [[maybe_unused]] Span1D<double const> ccx2cv_(ccx2cv);
}

TEST(View3DTest, Stream)
{
    std::array<int, 8> data;
    SpanND<3, int> sdata(data.data(), 2, 2, 2);
    sdata(0, 0, 0) = 0;
    sdata(0, 0, 1) = 1;
    sdata(0, 1, 0) = 2;
    sdata(0, 1, 1) = 3;
    sdata(1, 0, 0) = 4;
    sdata(1, 0, 1) = 5;
    sdata(1, 1, 0) = 6;
    sdata(1, 1, 1) = 7;
    ostringstream oss;
    oss << sdata;
    EXPECT_EQ(oss.str(), "[[[0,1][2,3]][[4,5][6,7]]]");
}
