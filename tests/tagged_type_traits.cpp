// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_tagged_type_traits_cpp {

struct DDimX
{
};

struct DDimY
{
};

struct DDimZ
{
};

} // namespace anonymous_namespace_workaround_tagged_type_traits_cpp

TEST(Combine, DiscreteElement)
{
    EXPECT_TRUE((std::is_same_v<
                 ddc::combine_t<
                         ddc::DiscreteElement<DDimX>,
                         ddc::DiscreteElement<DDimY>,
                         ddc::DiscreteElement<DDimZ>>,
                 ddc::DiscreteElement<DDimX, DDimY, DDimZ>>));
}

TEST(Combine, DiscreteVector)
{
    EXPECT_TRUE((std::is_same_v<
                 ddc::combine_t<
                         ddc::DiscreteVector<DDimX>,
                         ddc::DiscreteVector<DDimY>,
                         ddc::DiscreteVector<DDimZ>>,
                 ddc::DiscreteVector<DDimX, DDimY, DDimZ>>));
}

TEST(Combine, DiscreteDomain)
{
    EXPECT_TRUE((std::is_same_v<
                 ddc::combine_t<
                         ddc::DiscreteDomain<DDimX>,
                         ddc::DiscreteDomain<DDimY>,
                         ddc::DiscreteDomain<DDimZ>>,
                 ddc::DiscreteDomain<DDimX, DDimY, DDimZ>>));
}

TEST(Combine, StridedDiscreteDomain)
{
    EXPECT_TRUE((std::is_same_v<
                 ddc::combine_t<
                         ddc::StridedDiscreteDomain<DDimX>,
                         ddc::StridedDiscreteDomain<DDimY>,
                         ddc::StridedDiscreteDomain<DDimZ>>,
                 ddc::StridedDiscreteDomain<DDimX, DDimY, DDimZ>>));
}
