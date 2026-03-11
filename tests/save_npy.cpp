#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace {

std::array constexpr ns {2, 3, 4};
int constexpr n = ns[0] * ns[1] * ns[2];

} // namespace

TEST(SaveNpy, Float)
{
    Kokkos::View<float*, Kokkos::HostSpace> alloc("float", n);
    Kokkos::deep_copy(alloc, 2.3);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_float.npy", view);
}

TEST(SaveNpy, Double)
{
    Kokkos::View<double*, Kokkos::HostSpace> alloc("double", n);
    Kokkos::deep_copy(alloc, 2.3);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_double.npy", view);
}

TEST(SaveNpy, ComplexFloat)
{
    Kokkos::View<std::complex<float>*, Kokkos::HostSpace> alloc("complex float", n);
    Kokkos::deep_copy(alloc, std::complex<float>(2.3, 0.4));
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_complex_float.npy", view);
}

TEST(SaveNpy, ComplexDouble)
{
    Kokkos::View<std::complex<double>*, Kokkos::HostSpace> alloc("complex double", n);
    Kokkos::deep_copy(alloc, std::complex<double>(2.3, 0.4));
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_complex_double.npy", view);
}

TEST(SaveNpy, Char)
{
    Kokkos::View<char*, Kokkos::HostSpace> alloc("char", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_char.npy", view);
}

TEST(SaveNpy, Short)
{
    Kokkos::View<short*, Kokkos::HostSpace> alloc("short", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_short.npy", view);
}

TEST(SaveNpy, Int)
{
    Kokkos::View<int*, Kokkos::HostSpace> alloc("int", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_int.npy", view);
}

TEST(SaveNpy, Long)
{
    Kokkos::View<long*, Kokkos::HostSpace> alloc("long", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_long.npy", view);
}

TEST(SaveNpy, LongLong)
{
    Kokkos::View<long long*, Kokkos::HostSpace> alloc("long long", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_long_long.npy", view);
}

TEST(SaveNpy, UnsignedChar)
{
    Kokkos::View<unsigned char*, Kokkos::HostSpace> alloc("unsigned char", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_unsigned_char.npy", view);
}

TEST(SaveNpy, UnsignedShort)
{
    Kokkos::View<unsigned short*, Kokkos::HostSpace> alloc("unsigned short", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_unsigned_short.npy", view);
}

TEST(SaveNpy, UnsignedInt)
{
    Kokkos::View<unsigned int*, Kokkos::HostSpace> alloc("unsigned int", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_unsigned_int.npy", view);
}

TEST(SaveNpy, UnsignedLong)
{
    Kokkos::View<unsigned long*, Kokkos::HostSpace> alloc("unsigned long", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_unsigned_long.npy", view);
}

TEST(SaveNpy, UnsignedLongLong)
{
    Kokkos::View<unsigned long long*, Kokkos::HostSpace> alloc("unsigned long long", n);
    Kokkos::deep_copy(alloc, 2);
    Kokkos::mdspan view(alloc.data(), ns);
    ddc::experimental::save_npy("test_unsigned_long_long.npy", view);
}
