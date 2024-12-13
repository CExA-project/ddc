// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(DISCRETE_DOMAIN_CPP) {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::StorageDiscreteDomain<DDimX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::StorageDiscreteDomain<DDimY>;


struct DDimZ
{
};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;
using DDomZ = ddc::StorageDiscreteDomain<DDimZ>;


using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::StorageDiscreteDomain<DDimX, DDimY>;


using DElemYX = ddc::DiscreteElement<DDimY, DDimX>;
using DVectYX = ddc::DiscreteVector<DDimY, DDimX>;
using DDomYX = ddc::StorageDiscreteDomain<DDimY, DDimX>;

using DElemXZ = ddc::DiscreteElement<DDimX, DDimZ>;
using DVectXZ = ddc::DiscreteVector<DDimX, DDimZ>;
using DDomXZ = ddc::StorageDiscreteDomain<DDimX, DDimZ>;

using DElemZY = ddc::DiscreteElement<DDimZ, DDimY>;
using DVectZY = ddc::DiscreteVector<DDimZ, DDimY>;
using DDomZY = ddc::StorageDiscreteDomain<DDimZ, DDimY>;


using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;
using DDomXYZ = ddc::StorageDiscreteDomain<DDimX, DDimY, DDimZ>;

using DElemZYX = ddc::DiscreteElement<DDimZ, DDimY, DDimX>;
using DVectZYX = ddc::DiscreteVector<DDimZ, DDimY, DDimX>;
using DDomZYX = ddc::StorageDiscreteDomain<DDimZ, DDimY, DDimX>;

DElemX constexpr lbound_x(50);
DVectX constexpr nelems_x(3);
DElemX constexpr sentinel_x(lbound_x + nelems_x);
DElemX constexpr ubound_x(sentinel_x - 1);


DElemY constexpr lbound_y(4);
DVectY constexpr nelems_y(10);
DElemY constexpr sentinel_y(lbound_y);
DElemY constexpr ubound_y(sentinel_y - 1);

DElemZ constexpr lbound_z(7);
DVectZ constexpr nelems_z(15);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);
DElemXY constexpr ubound_x_y(ubound_x, ubound_y);

DElemXZ constexpr lbound_x_z(lbound_x, lbound_z);
DVectXZ constexpr nelems_x_z(nelems_x, nelems_z);

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(DISCRETE_DOMAIN_CPP)

TEST(StorageDiscreteDomainTest, Constructor)
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
    EXPECT_FALSE(dom_xy.is_inside(lbound_x + 1, lbound_y + 0));
}
