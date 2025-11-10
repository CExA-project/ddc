// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_print_cpp {

struct Dim0
{
};
struct Dim1
{
};
struct Dim2
{
};
struct Dim3
{
};
struct Dim4
{
};
struct Dim5
{
};

} // namespace anonymous_namespace_workaround_print_cpp

TEST(Print, ValidDemangledTypeName)
{
    std::stringstream ss;
    ddc::detail::print_demangled_type_name(ss, typeid(ddc::DiscreteDomain<>).name());
#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
    EXPECT_EQ(ss.str(), "ddc::DiscreteDomain<>");
#elif defined(KOKKOS_COMPILER_MSVC)
    EXPECT_EQ(ss.str(), "class ddc::DiscreteDomain<>");
#else
    GTEST_SKIP();
#endif
}

TEST(Print, InvalidDemangledTypeName)
{
    std::stringstream ss;
    ddc::detail::print_demangled_type_name(ss, "0");
#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
    EXPECT_EQ(ss.str(), "Error demangling dimension name: -2");
#else
    EXPECT_EQ(ss.str(), "0");
#endif
}

template <typename ElementType>
void PrintTestCheckOutput0d()
{
    ddc::DiscreteDomain<> const domain_full;

    ddc::Chunk chunk("chunk", domain_full, ddc::DeviceAllocator<ElementType>());
    ddc::ChunkSpan const chunk_span = chunk.span_view();

    ddc::parallel_fill(chunk_span, 0.12345);

    {
        std::stringstream ss;
        print_content(ss, chunk_span);
        EXPECT_EQ(ss.str(), "0.12345");
    }
}

TEST(Print, CheckOutput0d)
{
    PrintTestCheckOutput0d<float>();
    PrintTestCheckOutput0d<double>();
}

template <typename ElementType>
void TestPrintCheckOutput2d()
{
    unsigned const dim0 = 2;
    unsigned const dim1 = 2;

    ddc::DiscreteDomain<Dim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim0>(dim0));
    ddc::DiscreteDomain<Dim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim1>(dim1));

    ddc::DiscreteDomain<Dim0, Dim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk chunk("chunk", domain_2d, ddc::DeviceAllocator<ElementType>());
    ddc::ChunkSpan const chunk_span = chunk.span_view();

    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<Dim0, Dim1> const i) {
                if (i == domain_2d.front() + ddc::DiscreteVector<Dim0, Dim1>(1, 1)) {
                    chunk_span(i) = -0.12345;
                } else {
                    chunk_span(i) = 0.12345;
                }
            });

    {
        std::stringstream ss;
        print_content(ss, chunk_span);
        EXPECT_EQ(
                ss.str(),
                "[[ 0.12345  0.12345]\n"
                " [ 0.12345 -0.12345]]");
    }
    {
        std::stringstream ss;
        ss << std::setprecision(2);
        print_content(ss, chunk_span);
        EXPECT_EQ(
                ss.str(),
                "[[ 0.12  0.12]\n"
                " [ 0.12 -0.12]]");
    }
    {
        std::stringstream ss;
        ss << std::hexfloat;
        print_content(ss, chunk_span);
        if constexpr (std::is_same_v<ElementType, double>) {
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
        print_content(ss, chunk_span);
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

template <typename ElementType>
void TestPrintCheckoutOutput2dElision()
{
    unsigned const dim0 = 100;
    unsigned const dim1 = 100;

    ddc::DiscreteDomain<Dim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim0>(dim0));
    ddc::DiscreteDomain<Dim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim1>(dim1));

    ddc::DiscreteDomain<Dim0, Dim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk chunk("chunk", domain_2d, ddc::DeviceAllocator<ElementType>());
    ddc::ChunkSpan const chunk_span = chunk.span_view();

    // Fill the array with 0.12345 in the cells that should be visible and -0.12345 in the one that will be eluded
    // Check that the output is only aligned on 0.12345
    auto const subdom_0
            = domain_0.remove(ddc::DiscreteVector<Dim0>(3), ddc::DiscreteVector<Dim0>(3));
    auto const subdom_1
            = domain_1.remove(ddc::DiscreteVector<Dim1>(3), ddc::DiscreteVector<Dim1>(3));
    ddc::parallel_for_each(
            domain_2d,
            KOKKOS_LAMBDA(ddc::DiscreteElement<Dim0, Dim1> const i) {
                if (subdom_0.contains(ddc::DiscreteElement<Dim0>(i))
                    || subdom_1.contains(ddc::DiscreteElement<Dim1>(i))) {
                    chunk_span(i) = -0.12345;
                } else {
                    chunk_span(i) = 0.12345;
                }
            });

    {
        std::stringstream ss;
        print_content(ss, chunk_span);
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

template <typename ElementType>
void PrintTestCheckoutOutput3d()
{
    unsigned const dim0 = 3;
    unsigned const dim1 = 3;
    unsigned const dim2 = 3;

    ddc::DiscreteDomain<Dim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim0>(dim0));
    ddc::DiscreteDomain<Dim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim1>(dim1));
    ddc::DiscreteDomain<Dim2> const domain_2
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim2>(dim2));

    ddc::DiscreteDomain<Dim0, Dim1, Dim2> const domain_3d(domain_0, domain_1, domain_2);

    ddc::Chunk chunk("chunk", domain_3d, ddc::DeviceAllocator<ElementType>());
    ddc::ChunkSpan const chunk_span = chunk.span_view();

    ddc::parallel_fill(chunk_span, 0.12345);

    {
        std::stringstream ss;
        print_content(ss, chunk_span);
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
    using ElementType = double;

    unsigned const dim0 = 5;
    unsigned const dim1 = 5;

    ddc::DiscreteDomain<Dim0> const domain_0
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim0>(dim0));
    ddc::DiscreteDomain<Dim1> const domain_1
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<Dim1>(dim1));

    ddc::DiscreteDomain<Dim0, Dim1> const domain_2d(domain_0, domain_1);

    ddc::Chunk chunk("chunk", domain_2d, ddc::DeviceAllocator<ElementType>());
    ddc::ChunkSpan const chunk_span = chunk.span_view();

    {
        std::stringstream ss;
        print_type_info(ss, chunk_span);
        EXPECT_THAT(
                ss.str(),
                testing::MatchesRegex(
                        "anonymous_namespace_workaround_print_cpp::Dim0\\(5\\)×"
                        "anonymous_namespace_workaround_print_cpp::Dim1\\(5\\)\n"
                        "ddc::ChunkSpan<double, ddc::DiscreteDomain"
                        "<anonymous_namespace_workaround_print_cpp::Dim0,"
                        " anonymous_namespace_workaround_print_cpp::Dim1>"
                        ", Kokkos::layout_.+, Kokkos::.+Space>\n"));
    }
}

TEST(Print, CheckMetadata)
{
    PrintTestMetadata();
}
#endif
