// SPDX-License-Identifier: MIT

#include <type_traits>

#include <ddc/Coordinate>
#include <ddc/DiscreteCoordinate>
#include <ddc/RectilinearDomain>
#include <ddc/DiscreteDomain>
#include <ddc/UniformDiscretization>
#include <ddc/NonUniformDiscretization>

#include <gtest/gtest.h>

namespace {
    struct X;
    struct Y;

    using DDimX     = UniformDiscretization<X>;
    using DDimY     = UniformDiscretization<Y>;
    using NUDDimX   = NonUniformDiscretization<X>;
    using NUDDimY   = NonUniformDiscretization<Y>;

    using DDomNull  = DiscreteDomain<nullptr_t>;
    using DDomX     = DiscreteDomain<DDimX>;
    using NUDDomX   = DiscreteDomain<NUDDimX>;
    using DDomXY    = DiscreteDomain<DDimX, DDimY>;
    using NUDDomXY  = DiscreteDomain<NUDDimX, NUDDimY>;
}

TEST(RectilinearDomainTest, NonDiscreteDomainSpecialization_Value) {
    EXPECT_FALSE(is_rectilinear_domain_v<nullptr_t>);
    EXPECT_FALSE(is_rectilinear_domain_v<X>);
}

TEST(RectilinearDomainTest, DiscreteDomainSpecialization_Value) {
    EXPECT_FALSE(is_rectilinear_domain_v<DDomNull>);
    
    EXPECT_TRUE(is_rectilinear_domain_v<DDomX>);
    EXPECT_TRUE(is_rectilinear_domain_v<DDomXY>);
    EXPECT_TRUE(is_rectilinear_domain_v<NUDDomX>);
    EXPECT_TRUE(is_rectilinear_domain_v<NUDDomXY>);
}
