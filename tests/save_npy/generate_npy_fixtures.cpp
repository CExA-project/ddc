// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <type_traits>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;


struct DDimZ
{
};
using DElemZ = ddc::DiscreteElement<DDimZ>;
using DVectZ = ddc::DiscreteVector<DDimZ>;
using DDomZ = ddc::DiscreteDomain<DDimZ>;

using DElemXYZ = ddc::DiscreteElement<DDimX, DDimY, DDimZ>;
using DVectXYZ = ddc::DiscreteVector<DDimX, DDimY, DDimZ>;
using DDomXYZ = ddc::DiscreteDomain<DDimX, DDimY, DDimZ>;

namespace {

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(2);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(3);

DElemZ constexpr lbound_z = ddc::init_trivial_half_bounded_space<DDimZ>();
DVectZ constexpr nelems_z(4);

DElemXYZ constexpr lbound_x_y_z(lbound_x, lbound_y, lbound_z);
DVectXYZ constexpr nelems_x_y_z(nelems_x, nelems_y, nelems_z);

template <typename T>
constexpr T make_value()
{
    double const real = 2.3;
    double const imag = 0.4;
    if constexpr (
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>
            || std::is_same_v<T, Kokkos::complex<float>>
            || std::is_same_v<T, Kokkos::complex<double>>) {
        return T(real, imag);
    } else if constexpr (std::is_floating_point_v<T>) {
        return T(real);
    } else {
        return T(std::llround(real));
    }
}

template <typename T>
void save_array_0d(std::filesystem::path const& path, T value)
{
    ddc::DiscreteDomain<> const dom;
    ddc::Chunk chk(dom, ddc::DeviceAllocator<T>());
    ddc::parallel_fill(chk, value);

    ddc::experimental::save_npy(path, chk.span_cview());
}

template <typename T>
void save_array_1d(std::filesystem::path const& path, T value)
{
    DDomX const dom(lbound_x, nelems_x);
    ddc::Chunk chk(dom, ddc::DeviceAllocator<T>());
    ddc::parallel_fill(chk, value);

    ddc::experimental::save_npy(path, chk.span_cview());
}

template <typename T>
void save_array_2d_slice(std::filesystem::path const& path, T value)
{
    DDomXYZ const dom(lbound_x_y_z, nelems_x_y_z);
    ddc::Chunk chk(dom, ddc::DeviceAllocator<T>());
    ddc::parallel_fill(chk, value);

    ddc::experimental::save_npy(path, chk[DVectY(2)].span_cview());
}

template <typename T>
void save_array_3d(std::filesystem::path const& path, T value)
{
    DDomXYZ const dom(lbound_x_y_z, nelems_x_y_z);
    ddc::Chunk chk(dom, ddc::DeviceAllocator<T>());
    ddc::parallel_fill(chk, value);

    ddc::experimental::save_npy(path, chk.span_cview());
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos(argc, argv);
    ddc::ScopeGuard const ddc(argc, argv);

    save_array_0d("./float_0d.npy", make_value<float>());

    save_array_1d("./double_1d.npy", make_value<double>());

    save_array_2d_slice("./double_2d.npy", make_value<double>());

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

    save_array_3d("./std_complex_float_3d.npy", make_value<std::complex<float>>());
    save_array_3d("./std_complex_double_3d.npy", make_value<std::complex<double>>());
    save_array_3d("./kokkos_complex_float_3d.npy", make_value<Kokkos::complex<float>>());
    save_array_3d("./kokkos_complex_double_3d.npy", make_value<Kokkos::complex<double>>());

    return 0;
}
