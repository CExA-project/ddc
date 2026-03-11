#pragma once

#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

namespace ddc::detail {

enum class NpyByteOrder : char { little_endian = '<', big_endian = '>', not_applicable = '|' };

NpyByteOrder get_byte_order() noexcept;

enum class NpyKind : char {
    boolean = 'b',
    signed_int = 'i',
    unsigned_int = 'u',
    floating_point = 'f',
    complex = 'c',
    other = 'V',
};

struct NpyDtype
{
    NpyByteOrder byte_order;
    NpyKind kind;
    std::size_t itemsize; // in bytes

    std::string str() const;
};

template <typename T>
NpyDtype convert_to_npy_dtype()
{
    std::size_t const itemsize = sizeof(T);
    NpyByteOrder const byte_order = itemsize == 1 ? NpyByteOrder::not_applicable : get_byte_order();
    NpyKind kind;

    // ── Single-byte / untyped ─────────────────────────────────────────
    if constexpr (std::is_same_v<T, bool>) {
        kind = NpyKind::boolean;
    }
    // std::byte → raw byte buffer, no arithmetic meaning
    else if constexpr (std::is_same_v<T, std::byte>) {
        kind = NpyKind::other;
    }
    // char is a distinct type; its signedness is implementation-defined
    else if constexpr (std::is_same_v<T, char>) {
        kind = std::is_signed_v<char> ? NpyKind::signed_int : NpyKind::unsigned_int;
    }

    // ── Complex ───────────────────────────────────────────────────────
    // NumPy 'c' dtype stores interleaved real+imag, same layout as std::complex
    else if constexpr (
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        kind = NpyKind::complex;
    }

    // ── Floating-point ────────────────────────────────────────────────
    else if constexpr (std::is_floating_point_v<T>) {
        static_assert(
                !std::is_same_v<T, long double>,
                "long double is platform-specific (80/96/128-bit); cast to double first.");
        kind = NpyKind::floating_point;
    }

    // ── Integers ──────────────────────────────────────────────────────
    else if constexpr (std::is_signed_v<T>) {
        kind = NpyKind::signed_int;
    } else if constexpr (std::is_unsigned_v<T>) {
        kind = NpyKind::unsigned_int;
    }

    else {
        static_assert(sizeof(T) == 0, "Unsupported type for NpyDtype::of<T>()");
    }

    return {.byte_order = byte_order, .kind = kind, .itemsize = itemsize};
}

struct NpyArrayView
{
    void const* data;
    NpyDtype dtype;
    std::vector<std::size_t> shape;
    bool fortran_order;
};

void save_npy(std::ostream& os, NpyArrayView const& view);

} // namespace ddc::detail

namespace ddc::experimental {

template <typename T, typename Extents, typename Layout, typename Accessor>
void save_npy(std::ostream& os, Kokkos::mdspan<T, Extents, Layout, Accessor> const& mds)
{
    static_assert(
            std::is_same_v<Layout, Kokkos::layout_left>
                    || std::is_same_v<Layout, Kokkos::layout_right>,
            "save_npy: only contiguous layouts supported.");
    static_assert(
            std::is_same_v<Accessor, Kokkos::default_accessor<T>>
                    || std::is_same_v<Accessor, Kokkos::default_accessor<T const>>,
            "save_npy: non-host-accessible accessor. Use create_mirror_view + deep_copy first.");

    std::vector<std::size_t> shape(Extents::rank());
    for (std::size_t i = 0; i < Extents::rank(); ++i) {
        shape[i] = mds.extent(i);
    }

    ddc::detail::save_npy(
            os,
            ddc::detail::NpyArrayView {
                    .data = mds.data_handle(),
                    .dtype = ddc::detail::convert_to_npy_dtype<std::remove_const_t<T>>(),
                    .shape = std::move(shape),
                    .fortran_order = std::is_same_v<Layout, Kokkos::layout_left>,
            });
}

template <typename T, typename Extents, typename Layout, typename Accessor>
void save_npy(
        std::filesystem::path const& filename,
        Kokkos::mdspan<T, Extents, Layout, Accessor> const& mds)
{
    static_assert(
            std::is_same_v<Layout, Kokkos::layout_left>
                    || std::is_same_v<Layout, Kokkos::layout_right>,
            "save_npy: only contiguous layouts supported.");
    static_assert(
            std::is_same_v<Accessor, Kokkos::default_accessor<T>>
                    || std::is_same_v<Accessor, Kokkos::default_accessor<T const>>,
            "save_npy: non-host-accessible accessor. Use create_mirror_view + deep_copy first.");

    std::ofstream file(filename, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);

    save_npy(file, mds);
}

} // namespace ddc::experimental
