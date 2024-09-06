// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(DISCRETE_ELEMENT_CPP) {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;


struct DDimZ
{
};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;


using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;


using DElemYX = ddc::DiscreteElement<DDimY, DDimX>;
using DVectYX = ddc::DiscreteVector<DDimY, DDimX>;


using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(DISCRETE_ELEMENT_CPP)

TEST(DiscreteElementXYZTest, ConstructorFromDiscreteElements)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    std::size_t const uid_z = 4;
    DElemYX const iyx(uid_y, uid_x);
    DElemZ const iz(uid_z);
    DElemXYZ const ixyz(iyx, iz);
    EXPECT_EQ(ixyz.uid<DDimX>(), uid_x);
    EXPECT_EQ(ixyz.uid<DDimY>(), uid_y);
    EXPECT_EQ(ixyz.uid<DDimZ>(), uid_z);
}

TEST(DiscreteElementXYZTest, CopyAssignment)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    std::size_t const uid_z = 4;
    DElemXYZ const ixyz(uid_x, uid_y, uid_z);
    DElemXYZ ixyz2(0, 0, 0);
    ixyz2 = ixyz;
    EXPECT_EQ(ixyz2.uid<DDimX>(), uid_x);
    EXPECT_EQ(ixyz2.uid<DDimY>(), uid_y);
    EXPECT_EQ(ixyz2.uid<DDimZ>(), uid_z);
}

TEST(DiscreteElementXTest, PreIncrement)
{
    DElemX const ix0(3);
    DElemX ix1(ix0);
    DElemX const ix2 = ++ix1;
    EXPECT_EQ(ix1, ix0 + 1);
    EXPECT_EQ(ix2, ix0 + 1);
}

TEST(DiscreteElementXTest, PostIncrement)
{
    DElemX const ix0(3);
    DElemX ix1(ix0);
    DElemX const ix2 = ix1++;
    EXPECT_EQ(ix1, ix0 + 1);
    EXPECT_EQ(ix2, ix0);
}

TEST(DiscreteElementXTest, PreDecrement)
{
    DElemX const ix0(3);
    DElemX ix1(ix0);
    DElemX const ix2 = --ix1;
    EXPECT_EQ(ix1, ix0 - 1);
    EXPECT_EQ(ix2, ix0 - 1);
}

TEST(DiscreteElementXTest, PostDecrement)
{
    DElemX const ix0(3);
    DElemX ix1(ix0);
    DElemX const ix2 = ix1--;
    EXPECT_EQ(ix1, ix0 - 1);
    EXPECT_EQ(ix2, ix0);
}

TEST(DiscreteElementXTest, RightExternalBinaryOperatorPlus)
{
    std::size_t const uid_x = 7;
    DElemX const ix(uid_x);
    std::ptrdiff_t const dv_x = -2;
    DElemX const ix2 = ix + DVectX(dv_x);
    EXPECT_EQ(ix2.uid<DDimX>(), uid_x + dv_x);
}

TEST(DiscreteElementXTest, RightExternalBinaryOperatorMinus)
{
    std::size_t const uid_x = 7;
    DElemX const ix(uid_x);
    std::ptrdiff_t const dv_x = -2;
    DElemX const ixy2 = ix - dv_x;
    EXPECT_EQ(ixy2.uid<DDimX>(), uid_x - dv_x);
}

TEST(DiscreteElementXTest, BinaryOperatorMinus)
{
    std::size_t const uid_x = 7;
    DElemX const ix(uid_x);
    std::ptrdiff_t const dv_x = -2;
    DElemX const ix2 = ix + dv_x;
    DVectX dv2_x = ix2 - ix;
    EXPECT_EQ(ddc::get<DDimX>(dv2_x), dv_x);
}

TEST(DiscreteElementXYTest, ValueConstructor)
{
    DElemXY const ixy {};
    EXPECT_EQ(ixy.uid<DDimX>(), std::size_t());
    EXPECT_EQ(ixy.uid<DDimY>(), std::size_t());
}

TEST(DiscreteElementXYTest, UntaggedConstructor)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    DElemXY const ixy(uid_x, uid_y);
    EXPECT_EQ(ixy.uid<DDimX>(), uid_x);
    EXPECT_EQ(ixy.uid<DDimY>(), uid_y);
}

TEST(DiscreteElementXYTest, RightExternalBinaryOperatorPlus)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    DElemXY const ixy(uid_x, uid_y);
    std::ptrdiff_t const dv_x = -2;
    std::ptrdiff_t const dv_y = +3;
    DElemXY const ixy2 = ixy + DVectXY(dv_x, dv_y);
    EXPECT_EQ(ixy2.uid<DDimX>(), uid_x + dv_x);
    EXPECT_EQ(ixy2.uid<DDimY>(), uid_y + dv_y);
}

TEST(DiscreteElementXYTest, RightExternalBinaryOperatorMinus)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    DElemXY const ixy(uid_x, uid_y);
    std::ptrdiff_t const dv_x = -2;
    std::ptrdiff_t const dv_y = +3;
    DElemXY const ixy2 = ixy - DVectXY(dv_x, dv_y);
    EXPECT_EQ(ixy2.uid<DDimX>(), uid_x - dv_x);
    EXPECT_EQ(ixy2.uid<DDimY>(), uid_y - dv_y);
}

TEST(DiscreteElementXYTest, BinaryOperatorMinus)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    DElemXY const ixy(uid_x, uid_y);
    std::ptrdiff_t const dv_x = -2;
    std::ptrdiff_t const dv_y = +3;
    DElemXY const ixy2 = ixy + DVectXY(dv_x, dv_y);
    DVectXY dv_xy = ixy2 - ixy;
    EXPECT_EQ(ddc::get<DDimX>(dv_xy), dv_x);
    EXPECT_EQ(ddc::get<DDimY>(dv_xy), dv_y);
}

TEST(DiscreteElementXYZTest, RightExternalBinaryOperatorPlus)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    std::size_t const uid_z = 4;
    DElemXYZ const ixyz(uid_x, uid_y, uid_z);
    std::ptrdiff_t const dv_x = -2;
    std::ptrdiff_t const dv_y = +3;
    DElemXYZ const ixyz2 = ixyz + DVectXY(dv_x, dv_y);
    EXPECT_EQ(ixyz2.uid<DDimX>(), uid_x + dv_x);
    EXPECT_EQ(ixyz2.uid<DDimY>(), uid_y + dv_y);
    EXPECT_EQ(ixyz2.uid<DDimZ>(), uid_z);
}

TEST(DiscreteElementXYZTest, RightExternalBinaryOperatorMinus)
{
    std::size_t const uid_x = 7;
    std::size_t const uid_y = 13;
    std::size_t const uid_z = 4;
    DElemXYZ const ixyz(uid_x, uid_y, uid_z);
    std::ptrdiff_t const dv_x = -2;
    std::ptrdiff_t const dv_y = +3;
    DElemXYZ const ixyz2 = ixyz - DVectXY(dv_x, dv_y);
    EXPECT_EQ(ixyz2.uid<DDimX>(), uid_x - dv_x);
    EXPECT_EQ(ixyz2.uid<DDimY>(), uid_y - dv_y);
    EXPECT_EQ(ixyz2.uid<DDimZ>(), uid_z);
}
