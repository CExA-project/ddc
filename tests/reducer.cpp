// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <bitset>
#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

TEST(Reducer, Sum)
{
    ddc::reducer::sum<int> const reducer;
    EXPECT_EQ(reducer(-1, 3), 2);
    EXPECT_EQ(reducer(3, -1), 2);
}

TEST(Reducer, Prod)
{
    ddc::reducer::prod<int> const reducer;
    EXPECT_EQ(reducer(-1, 3), -3);
    EXPECT_EQ(reducer(3, -1), -3);
}

TEST(Reducer, LAnd)
{
    ddc::reducer::land<bool> const reducer;
    EXPECT_EQ(reducer(true, true), true);
    EXPECT_EQ(reducer(true, false), false);
    EXPECT_EQ(reducer(false, true), false);
    EXPECT_EQ(reducer(false, false), false);
}

TEST(Reducer, LOr)
{
    ddc::reducer::lor<bool> const reducer;
    EXPECT_EQ(reducer(true, true), true);
    EXPECT_EQ(reducer(true, false), true);
    EXPECT_EQ(reducer(false, true), true);
    EXPECT_EQ(reducer(false, false), false);
}

TEST(Reducer, BAnd)
{
    ddc::reducer::band<std::bitset<4>> const reducer;
    EXPECT_EQ(reducer(std::bitset<4>("1100"), std::bitset<4>("1010")), std::bitset<4>("1000"));
    EXPECT_EQ(reducer(std::bitset<4>("1010"), std::bitset<4>("1100")), std::bitset<4>("1000"));
}

TEST(Reducer, BOr)
{
    ddc::reducer::bor<std::bitset<4>> const reducer;
    EXPECT_EQ(reducer(std::bitset<4>("1100"), std::bitset<4>("1010")), std::bitset<4>("1110"));
    EXPECT_EQ(reducer(std::bitset<4>("1010"), std::bitset<4>("1100")), std::bitset<4>("1110"));
}

TEST(Reducer, BXOr)
{
    ddc::reducer::bxor<std::bitset<4>> const reducer;
    EXPECT_EQ(reducer(std::bitset<4>("1100"), std::bitset<4>("1010")), std::bitset<4>("0110"));
    EXPECT_EQ(reducer(std::bitset<4>("1010"), std::bitset<4>("1100")), std::bitset<4>("0110"));
}

TEST(Reducer, Min)
{
    ddc::reducer::min<int> const reducer;
    EXPECT_EQ(reducer(-1, 3), -1);
    EXPECT_EQ(reducer(3, -1), -1);
}

TEST(Reducer, Max)
{
    ddc::reducer::max<int> const reducer;
    EXPECT_EQ(reducer(-1, 3), 3);
    EXPECT_EQ(reducer(3, -1), 3);
}

TEST(Reducer, Minmax)
{
    ddc::reducer::minmax<int> const reducer;
    EXPECT_EQ(reducer(std::pair(-1, 3), std::pair(3, -1)), std::pair(-1, 3));
    EXPECT_EQ(reducer(std::pair(3, -1), std::pair(-1, 3)), std::pair(-1, 3));
}
