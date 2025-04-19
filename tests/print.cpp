// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <ddc/ddc.hpp>

#include <gmock/gmock.h>
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

template <typename cell>
void PrintTestCheckOutput0d()
{
    ddc::DiscreteDomain<> const domain_full;

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_full, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    ddc::parallel_for_each(
            domain_full,
            KOKKOS_LAMBDA(ddc::DiscreteElement<> const i) { cells_in(i) = 0.12345; });

    {
        std::stringstream ss;
        print_content(ss, cells_in);
        EXPECT_EQ(ss.str(), "0.12345");
    }
}

TEST(Print, CheckOutput0d)
{
    PrintTestCheckOutput0d<float>();
    PrintTestCheckOutput0d<double>();
}

template <typename cell>
void TestPrintCheckOutput2d()
{
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
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                if (i == domain_2d.front() + ddc::DiscreteVector<DDim0, DDim1>(1, 1)) {
                    cells_in(i) = -0.12345;
                } else {
                    cells_in(i) = 0.12345;
                }
            });

    {
        std::stringstream ss;
        print_content(ss, cells_in);
        EXPECT_EQ(
                ss.str(),
                "[[ 0.12345  0.12345]\n"
                " [ 0.12345 -0.12345]]");
    }
    {
        std::stringstream ss;
        ss << std::setprecision(2);
        print_content(ss, cells_in);
        EXPECT_EQ(
                ss.str(),
                "[[ 0.12  0.12]\n"
                " [ 0.12 -0.12]]");
    }
    {
        std::stringstream ss;
        ss << std::hexfloat;
        print_content(ss, cells_in);
        if constexpr (std::is_same_v<cell, double>) {
            EXPECT_EQ(
                    ss.str(),
                    "[[ 0x1.f9a6b50b0f27cp-4  0x1.f9a6b50b0f27cp-4]\n"
                    " [ 0x1.f9a6b50b0f27cp-4 -0x1.f9a6b50b0f27cp-4]]");
        } else {
#if defined(KOKKOS_COMPILER_MSVC)
            EXPECT_EQ(
                    ss.str(),
                    "[[ 0x1.f9a6b60000000p-4  0x1.f9a6b60000000p-4]\n"
                    " [ 0x1.f9a6b60000000p-4 -0x1.f9a6b60000000p-4]]");
#else
            EXPECT_EQ(
                    ss.str(),
                    "[[ 0x1.f9a6b6p-4  0x1.f9a6b6p-4]\n"
                    " [ 0x1.f9a6b6p-4 -0x1.f9a6b6p-4]]");
#endif
        }
    }
    {
        std::stringstream ss;
        ss << std::scientific;
        print_content(ss, cells_in);
        EXPECT_EQ(
                ss.str(),
                "[[ 1.234500e-01  1.234500e-01]\n"
                " [ 1.234500e-01 -1.234500e-01]]");
    }
}

TEST(Print, CheckOutput2d)
{
    TestPrintCheckOutput2d<float>();
    TestPrintCheckOutput2d<double>();
}

template <typename cell>
void TestPrintCheckoutOutput2dElision()
{
    unsigned const dim0 = 100;
    unsigned const dim1 = 100;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    // Fill the array with 0.12345 in the cells that should be visible and -0.12345 in the one that will be eluded
    // Check that the output is only aligned on 0.12345
    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim0, DDim1> const i) {
                if ((i.uid<DDim0>() >= 3 && i.uid<DDim0>() < dim0 - 3)
                    || (i.uid<DDim1>() >= 3 && i.uid<DDim1>() < dim1 - 3)) {
                    cells_in(i) = -0.12345;
                } else {
                    cells_in(i) = 0.12345;
                }
            });
    {
        std::stringstream ss;
        print_content(ss, cells_in);
        EXPECT_EQ(
                ss.str(),
                "[[0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " ...\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]\n"
                " [0.12345 0.12345 0.12345 ... 0.12345 0.12345 0.12345]]");
    }
}

TEST(Print, CheckOutput2dElision)
{
    TestPrintCheckoutOutput2dElision<float>();
    TestPrintCheckoutOutput2dElision<double>();
}

template <typename cell>
void PrintTestCheckoutOutput3d()
{
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
                ss.str(),
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
                "  [0.12345 0.12345 0.12345]]]");
    }
}

TEST(Print, CheckOutput3d)
{
    PrintTestCheckoutOutput3d<float>();
    PrintTestCheckoutOutput3d<double>();
}

#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
void PrintTestMetadata()
{
    using cell = double;

    unsigned const dim0 = 5;
    unsigned const dim1 = 5;

    ddc::DiscreteDomain<DDim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim0>(dim0));
    ddc::DiscreteDomain<DDim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDim1>(dim1));

    ddc::DiscreteDomain<DDim0, DDim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk cells_in_dev_alloc("cells_in_dev", domain_2d, ddc::DeviceAllocator<cell>());
    ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();

    {
        std::stringstream ss;
        print_type_info(ss, cells_in);
        EXPECT_THAT(
                ss.str(),
                testing::MatchesRegex(
                        "anonymous_namespace_workaround_print_cpp::DDim0(5)×"
                        "anonymous_namespace_workaround_print_cpp::DDim1(5)\\n"
                        "ddc::ChunkSpan<double, ddc::DiscreteDomain"
                        "<anonymous_namespace_workaround_print_cpp::DDim0,"
                        " anonymous_namespace_workaround_print_cpp::DDim1>"
                        ", Kokkos::layout_.+, Kokkos::.+Space>\\n"));
    }
}

TEST(Print, CheckMetadata)
{
    PrintTestMetadata();
}
#endif
