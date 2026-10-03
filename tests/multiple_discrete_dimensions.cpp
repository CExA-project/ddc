// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Macros.hpp>

inline namespace anonymous_namespace_workaround_multiple_discrete_dimensions_cpp {

class SingleValueDiscreteDimension : ddc::DiscreteDimension
{
public:
    using discrete_dimension_type = SingleValueDiscreteDimension;

public:
    template <class DDim, class MemorySpace>
    class Impl
    {
        friend class Impl<DDim, Kokkos::HostSpace>;

    private:
        int m_value;

    public:
        using discrete_dimension_type = SingleValueDiscreteDimension;

        Impl() = default;

        explicit Impl(int value) : m_value(value) {}

        explicit Impl(Impl<DDim, Kokkos::HostSpace> const& impl)
            requires(!std::is_same_v<MemorySpace, Kokkos::HostSpace>)
            : m_value(impl.m_value)
        {
        }

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

} // namespace anonymous_namespace_workaround_multiple_discrete_dimensions_cpp

TEST(MultipleDiscreteDimensions, Value)
{
    ddc::init_discrete_space<SVDD1>(239);
    ddc::init_discrete_space<SVDD2>(928);
    EXPECT_EQ(ddc::discrete_space<SVDD1>().value(), 239);
    EXPECT_EQ(ddc::discrete_space<SVDD2>().value(), 928);
}
