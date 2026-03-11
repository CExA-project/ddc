// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

// #include <array>
#include <cmath>
// #include <complex>
#include <stdexcept>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// namespace {

// std::array constexpr ns {2, 3, 4};
// int constexpr n = ns[0] * ns[1] * ns[2];

// template <typename T>
// constexpr T make_value()
// {
//     std::complex<double> const base_value(2.3, 0.4);
//     if constexpr (
//             std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
//         return T(base_value);
//     } else if constexpr (std::is_floating_point_v<T>) {
//         return T(std::real(base_value));
//     } else {
//         return T(std::llround(std::real(base_value)));
//     }
// }

// } // namespace

// template <typename T>
// struct SaveNpyTest : public ::testing::Test
// {
//     using data_type = T;
// };

// using SaveNpyTypes = ::testing::Types<
//         float,
//         double,
//         std::complex<float>,
//         std::complex<double>,
//         char,
//         signed char,
//         signed short,
//         signed int,
//         signed long,
//         signed long long,
//         unsigned char,
//         unsigned short,
//         unsigned int,
//         unsigned long,
//         unsigned long long>;

// TYPED_TEST_SUITE(SaveNpyTest, SaveNpyTypes);

// TYPED_TEST(SaveNpyTest, SaveNpy0d)
// {
//     using data_type = typename TestFixture::data_type;
//     std::string const label(typeid(data_type).name());
//     Kokkos::View<data_type, Kokkos::HostSpace> const alloc(label);
//     Kokkos::deep_copy(alloc, make_value<data_type>());
//     Kokkos::mdspan const view(alloc.data());

//     ddc::experimental::save_npy("test.npy", view);
// }

// TYPED_TEST(SaveNpyTest, SaveNpy1d)
// {
//     using data_type = typename TestFixture::data_type;
//     std::string const label(typeid(data_type).name());
//     Kokkos::View<data_type*, Kokkos::HostSpace> const alloc(label, n);
//     Kokkos::deep_copy(alloc, make_value<data_type>());
//     Kokkos::mdspan const view(alloc.data(), n);

//     ddc::experimental::save_npy("test_" + label + "_1d.npy", view);
// }

// TYPED_TEST(SaveNpyTest, SaveNpy3d)
// {
//     using data_type = typename TestFixture::data_type;
//     std::string const label(typeid(data_type).name());
//     Kokkos::View<data_type*, Kokkos::HostSpace> const alloc(label, n);
//     Kokkos::deep_copy(alloc, make_value<data_type>());
//     Kokkos::mdspan const view(alloc.data(), ns);

//     ddc::experimental::save_npy("test_" + label + "_3d.npy", view);
// }

TEST(SaveNpy, LargeHeader)
{
    ddc::detail::NpyArrayView const view {
            .data = nullptr,
            .dtype = ddc::detail::convert_to_npy_dtype<char>(),
            .shape = std::vector<std::size_t>(25'000, 0),
            .fortran_order = true,
    };
    EXPECT_THROW(ddc::detail::save_npy("test.npy", view), std::runtime_error);
}
