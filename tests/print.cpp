// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

inline namespace anonymous_namespace_workaround_print_cpp {

struct DDim0
{
};
struct DDim1
{
};
struct DDim2
{
};
struct DDim3
{
};
struct DDim4
{
};
struct DDim5
{
};

} // namespace anonymous_namespace_workaround_print_cpp

void PrintTestCheckOutput0d()
{
    using cell = double;

    ddc::DiscreteDomain<> const domain_full;

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<> const i) { cells_in(i) = 0.12345; });

    {
        std::stringstream ss;
		print_content(ss, cells_in);
        EXPECT_EQ("0.12345", ss.str());
    }
}

TEST(Print, CheckOutput0d)
{
    PrintTestCheckOutput0d();
}

void TestPrintCheckOutput2d()
{
    using cell = double;

    unsigned const dim0 = 2;
    unsigned const dim1 = 2;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) { cells_in(i) = 0.12345; });

	cells_in(ddc::DiscreteElement<DDim0, DDim1>(1,1)) = -0.12345;
    {
        std::stringstream ss;
		print_content(ss, cells_in);
        EXPECT_EQ(
                "[[ 0.12345  0.12345]\n"
                " [ 0.12345 -0.12345]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::setprecision(2);
		print_content(ss, cells_in);
        EXPECT_EQ(
                "[[ 0.12  0.12]\n"
                " [ 0.12 -0.12]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::hexfloat;
		print_content(ss, cells_in);
        EXPECT_EQ(
                "[[ 0x1.f9a6b50b0f27cp-4  0x1.f9a6b50b0f27cp-4]\n"
                " [ 0x1.f9a6b50b0f27cp-4 -0x1.f9a6b50b0f27cp-4]]",
                ss.str());
    }
    {
        std::stringstream ss;
        ss << std::scientific;
		print_content(ss, cells_in);
        EXPECT_EQ(
                "[[ 1.234500e-01  1.234500e-01]\n"
                " [ 1.234500e-01 -1.234500e-01]]",
                ss.str());
    }
}

TEST(Print, CheckOutput2d)
{
    TestPrintCheckOutput2d();
}

void TestPrintCheckoutOutput2d_elision()
{
    using cell = double;

    unsigned const dim0 = 100;
    unsigned const dim1 = 100;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) { cells_in(i) = 0.12345; });

	// All the following values are just outside the bound of what is visible and thus should not affect alignment
	cells_in(ddc::DiscreteElement<DDim0, DDim1>(0,3)) = -0.12345;
	cells_in(ddc::DiscreteElement<DDim0, DDim1>(0,96)) = -0.12345;
	cells_in(ddc::DiscreteElement<DDim0, DDim1>(3,0)) = -0.12345;
	cells_in(ddc::DiscreteElement<DDim0, DDim1>(96,0)) = -0.12345;
    {
        std::stringstream ss;
		print_content(ss, cells_in);
        EXPECT_EQ(
                "[[0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " ...\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]]",
                ss.str());
    }
}

TEST(Print, CheckOutput2d_elision)
{
    TestPrintCheckoutOutput2d_elision();
}

void PrintTestCheckoutOutput3d()
{
    using cell = double;

    unsigned const dim0 = 3;
    unsigned const dim1 = 3;
    unsigned const dim2 = 3;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));
    ddc::DiscreteDomain<DDim2> const domain_2
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim2>(dim2));

    ddc::DiscreteDomain<DDim0, DDim1, DDim2> const domain_3d(domain_0, domain_1, domain_2);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_3d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    ddc::parallel_for_each(
            domain_3d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1, DDim2> const i) {
                cells_in(i) = 0.12345;
            });

    {
        std::stringstream ss;
		print_content(ss, cells_in);
        EXPECT_EQ(
                "[[[0.12345 0.12345 0.12345]\n"
                "  [0.12345 0.12345 0.12345]\n"
                "  [0.12345 0.12345 0.12345]]\n"
                "\n"
                " [[0.12345 0.12345 0.12345]\n"
                "  [0.12345 0.12345 0.12345]\n"
                "  [0.12345 0.12345 0.12345]]\n"
                "\n"
                " [[0.12345 0.12345 0.12345]\n"
                "  [0.12345 0.12345 0.12345]\n"
                "  [0.12345 0.12345 0.12345]]]",
                ss.str());
    }
}

TEST(Print, CheckOutput3d)
{
    PrintTestCheckoutOutput3d();
}
