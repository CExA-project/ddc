// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <filesystem>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

namespace {

std::array constexpr ns {2, 3, 4};
int constexpr n = ns[0] * ns[1] * ns[2];

template <typename T>
constexpr T make_value()
{
    std::complex<double> const base_value(2.3, 0.4);
    if constexpr (
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return T(base_value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return T(std::real(base_value));
    } else {
        return T(std::llround(std::real(base_value)));
    }
}

template <typename T>
void save_array_0d(std::filesystem::path const& path, T value)
{
    Kokkos::View<T, Kokkos::HostSpace> const alloc("");
    Kokkos::deep_copy(alloc, value);
    Kokkos::mdspan const view(alloc.data());

    ddc::experimental::save_npy(path, view);
}

template <typename T>
void save_array_1d(std::filesystem::path const& path, T value)
{
    Kokkos::View<T*, Kokkos::HostSpace> const alloc("", n);
    Kokkos::deep_copy(alloc, value);
    Kokkos::mdspan const view(alloc.data(), n);

    ddc::experimental::save_npy(path, view);
}

template <typename T>
void save_array_3d(std::filesystem::path const& path, T value)
{
    Kokkos::View<T*, Kokkos::HostSpace> const alloc("", n);
    Kokkos::deep_copy(alloc, value);
    Kokkos::mdspan const view(alloc.data(), ns);

    ddc::experimental::save_npy(path, view);
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos(argc, argv);
    ddc::ScopeGuard const ddc(argc, argv);

    save_array_0d("./float_0d.npy", make_value<float>());

    save_array_1d("./double_1d.npy", make_value<double>());

    save_array_3d("./int8_3d.npy", make_value<std::int8_t>());
    save_array_3d("./int16_3d.npy", make_value<std::int16_t>());
    save_array_3d("./int32_3d.npy", make_value<std::int32_t>());
    save_array_3d("./int64_3d.npy", make_value<std::int64_t>());

    save_array_3d("./uint8_3d.npy", make_value<std::uint8_t>());
    save_array_3d("./uint16_3d.npy", make_value<std::uint16_t>());
    save_array_3d("./uint32_3d.npy", make_value<std::uint32_t>());
    save_array_3d("./uint64_3d.npy", make_value<std::uint64_t>());

    save_array_3d("./float_3d.npy", make_value<float>());
    save_array_3d("./double_3d.npy", make_value<double>());

    save_array_3d("./complex_float_3d.npy", make_value<std::complex<float>>());
    save_array_3d("./complex_double_3d.npy", make_value<std::complex<double>>());

    return 0;
}
