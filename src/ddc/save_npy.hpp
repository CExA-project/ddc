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

#include "chunk_span.hpp"
#include "create_mirror.hpp"

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
    } else if constexpr (
            std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>
            || std::is_same_v<T, Kokkos::complex<float>>
            || std::is_same_v<T, Kokkos::complex<double>>) {
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

template <typename T, typename Extents, typename Layout, typename Accessor>
NpyArrayView to_np_array_view(Kokkos::mdspan<T, Extents, Layout, Accessor> const& mds)
{
    std::vector<std::size_t> shape(Extents::rank());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        shape[i] = mds.extent(i);
    }

    return NpyArrayView {
            .data = mds.data_handle(),
            .dtype = convert_to_npy_dtype<std::remove_const_t<T>>(),
            .shape = std::move(shape),
            .fortran_order = std::is_same_v<Layout, Kokkos::layout_left>,
    };
}

void save_npy(std::ostream& os, NpyArrayView const& view);

void save_npy(std::filesystem::path const& filename, NpyArrayView const& view);

} // namespace ddc::detail

namespace ddc::experimental {

/**
 * @brief Save a ddc::ChunkSpan in the NumPy format in a stream.
 *
 * @param os Output stream receiving the .npy data.
 * @param chunk_span ChunkSpan to serialize.
 */
template <typename T, typename SupportType, typename LayoutStridedPolicy, typename MemorySpace>
void save_npy(
        std::ostream& os,
        ChunkSpan<T, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    auto chunk_span_right = ::ddc::detail::create_layout_right_view_and_copy(chunk_span);
    auto chunk_span_right_host
            = create_mirror_view_and_copy(Kokkos::HostSpace(), chunk_span_right.span_view());
    ddc::detail::
            save_npy(os, ddc::detail::to_np_array_view(chunk_span_right_host.allocation_mdspan()));
}

/**
 * @brief Save a ddc::ChunkSpan in the NumPy format in a file.
 *
 * @param filename Path to the output .npy file.
 * @param chunk_span ChunkSpan to serialize.
 */
template <typename T, typename SupportType, typename LayoutStridedPolicy, typename MemorySpace>
void save_npy(
        std::filesystem::path const& filename,
        ChunkSpan<T, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    auto chunk_span_right = ::ddc::detail::create_layout_right_view_and_copy(chunk_span);
    auto chunk_span_right_host
            = create_mirror_view_and_copy(Kokkos::HostSpace(), chunk_span_right.span_view());
    ddc::detail::save_npy(
            filename,
            ddc::detail::to_np_array_view(chunk_span_right_host.allocation_mdspan()));
}

} // namespace ddc::experimental
