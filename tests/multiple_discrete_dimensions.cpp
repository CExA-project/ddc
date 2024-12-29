// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(MULTIPLE_DISCRETE_DIMENSIONS_CPP) {

class SingleValueDiscreteDimension
{
public:
    using discrete_dimension_type = SingleValueDiscreteDimension;

public:
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    private:
        int m_value;

    public:
        using discrete_dimension_type = SingleValueDiscreteDimension;

        Impl() = default;

        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl) : m_value(impl.m_value)
        {
        }

        explicit Impl(int value) : m_value(value) {}

        Impl(Impl const&) = delete;

        Impl(Impl&&) = default;

        ~Impl() = default;

        Impl& operator=(Impl const& x) = delete;

        Impl& operator=(Impl&& x) = default;

        KOKKOS_FUNCTION int value() const noexcept
        {
            return m_value;
        }
    };
};

struct SVDD1 : SingleValueDiscreteDimension
{
};

struct SVDD2 : SingleValueDiscreteDimension
{
};

template <class DDim>
void create_single_discretization(int value)
{
    ddc::init_discrete_space_from_impl<DDim>(
            SingleValueDiscreteDimension::template Impl<DDim, Kokkos::HostSpace>(value));
}

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(MULTIPLE_DISCRETE_DIMENSIONS_CPP)

TEST(MultipleDiscreteDimensions, Value)
{
    create_single_discretization<SVDD1>(239);
    create_single_discretization<SVDD2>(928);
    EXPECT_EQ(ddc::discrete_space<SVDD1>().value(), 239);
    EXPECT_EQ(ddc::discrete_space<SVDD2>().value(), 928);
}
