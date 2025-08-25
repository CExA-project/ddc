// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

inline namespace anonymous_namespace_workaround_sparse_discrete_domain_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::SparseDiscreteDomain<DDimX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::SparseDiscreteDomain<DDimY>;


using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::SparseDiscreteDomain<DDimX, DDimY>;


using DElemYX = ddc::DiscreteElement<DDimY, DDimX>;
using DVectYX = ddc::DiscreteVector<DDimY, DDimX>;
using DDomYX = ddc::SparseDiscreteDomain<DDimY, DDimX>;


DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();

} // namespace anonymous_namespace_workaround_sparse_discrete_domain_cpp

TEST(SparseDiscreteDomainTest, Constructor)
{
    Kokkos::View<DElemX*, Kokkos::SharedSpace> const view_x("view_x", 2);
    view_x(0) = lbound_x + 0;
    view_x(1) = lbound_x + 2;
    Kokkos::View<DElemY*, Kokkos::SharedSpace> const view_y("view_y", 3);
    view_y(0) = lbound_y + 0;
    view_y(1) = lbound_y + 2;
    view_y(2) = lbound_y + 5;
    DDomXY const dom_xy(view_x, view_y);
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 0, lbound_y + 0), DVectXY(0, 0));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 0, lbound_y + 2), DVectXY(0, 1));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 0, lbound_y + 5), DVectXY(0, 2));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 2, lbound_y + 0), DVectXY(1, 0));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 2, lbound_y + 2), DVectXY(1, 1));
    EXPECT_EQ(dom_xy.distance_from_front(lbound_x + 2, lbound_y + 5), DVectXY(1, 2));
    EXPECT_FALSE(dom_xy.contains(lbound_x + 1, lbound_y + 0));
}
