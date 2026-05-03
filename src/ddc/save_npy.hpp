// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

namespace ddc::detail {

enum class NpyByteOrder : char { little_endian = '<', big_endian = '>', not_applicable = '|' };

NpyByteOrder get_byte_order(std::size_t itemsize) noexcept;

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
    NpyByteOrder const byte_order = get_byte_order(itemsize);
    NpyKind kind;

    if constexpr (std::is_same_v<T, bool>) {
        // ── Single-byte / untyped ─────────────────────────────────────────
        kind = NpyKind::boolean;
    } else if constexpr (std::is_same_v<T, std::byte>) {
        // std::byte → raw byte buffer, no arithmetic meaning
        kind = NpyKind::other;
    } else if constexpr (std::is_same_v<T, char>) {
        // char is a distinct type; its signedness is implementation-defined
        kind = std::is_signed_v<char> ? NpyKind::signed_int : NpyKind::unsigned_int;
    } else if constexpr (
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        // ── Complex ───────────────────────────────────────────────────────
        // NumPy 'c' dtype stores interleaved real+imag, same layout as std::complex
        kind = NpyKind::complex;
    } else if constexpr (std::is_floating_point_v<T>) {
        // ── Floating-point ────────────────────────────────────────────────
        static_assert(
                !std::is_same_v<T, long double>,
                "long double is platform-specific (80/96/128-bit); cast to double first.");
        kind = NpyKind::floating_point;
    } else if constexpr (std::is_signed_v<T>) {
        // ── Integers ──────────────────────────────────────────────────────
        kind = NpyKind::signed_int;
    } else if constexpr (std::is_unsigned_v<T>) {
        kind = NpyKind::unsigned_int;
    } else {
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

void save_npy(std::filesystem::path const& filename, NpyArrayView const& view);

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

    std::vector<std::size_t> shape(Extents::rank());
    if constexpr (Extents::rank() > 0) {
        for (std::size_t i = 0; i < Extents::rank(); ++i) {
            shape[i] = mds.extent(i);
        }
    }

    ddc::detail::save_npy(
            filename,
            ddc::detail::NpyArrayView {
                    .data = mds.data_handle(),
                    .dtype = ddc::detail::convert_to_npy_dtype<std::remove_const_t<T>>(),
                    .shape = std::move(shape),
                    .fortran_order = std::is_same_v<Layout, Kokkos::layout_left>,
            });
}

} // namespace ddc::experimental
