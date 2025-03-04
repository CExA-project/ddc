// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace ddc::detail {

template <class T>
struct type_holder
{
    using type = T;
};

template <class KokkosLP>
struct kokkos_to_mdspan_layout
{
    static_assert(
            std::is_same_v<KokkosLP, KokkosLP>,
            "Usage of non-specialized kokkos_to_mdspan_layout struct is not allowed");
};

template <>
struct kokkos_to_mdspan_layout<Kokkos::LayoutLeft>
{
    using type = Kokkos::layout_left;
};

template <>
struct kokkos_to_mdspan_layout<Kokkos::LayoutRight>
{
    using type = Kokkos::layout_right;
};

template <>
struct kokkos_to_mdspan_layout<Kokkos::LayoutStride>
{
    using type = Kokkos::layout_stride;
};

/// Alias template to transform a mdspan layout type to a Kokkos layout type
template <class KokkosLP>
using kokkos_to_mdspan_layout_t = typename kokkos_to_mdspan_layout<KokkosLP>::type;


template <class mdspanLP>
struct mdspan_to_kokkos_layout
{
    static_assert(
            std::is_same_v<mdspanLP, mdspanLP>,
            "Usage of non-specialized mdspan_to_kokkos_layout struct is not allowed");
};

template <>
struct mdspan_to_kokkos_layout<Kokkos::layout_left>
{
    using type = Kokkos::LayoutLeft;
};

template <>
struct mdspan_to_kokkos_layout<Kokkos::layout_right>
{
    using type = Kokkos::LayoutRight;
};

template <>
struct mdspan_to_kokkos_layout<Kokkos::layout_stride>
{
    using type = Kokkos::LayoutStride;
};

/// Alias template to transform a Kokkos layout type to a mdspan layout type
template <class mdspanLP>
using mdspan_to_kokkos_layout_t = typename mdspan_to_kokkos_layout<mdspanLP>::type;

template <class ET, std::size_t N>
struct mdspan_to_kokkos_element
    : std::conditional_t<
              N == 0,
              type_holder<ET>,
              mdspan_to_kokkos_element<std::add_pointer_t<ET>, N - 1>>
{
};

/// Alias template to transform a mdspan element type to a Kokkos element type
/// Only dynamic dimensions is supported for now i.e. `double[4]*` is not yet covered.
template <class ET, std::size_t N>
using mdspan_to_kokkos_element_t = typename mdspan_to_kokkos_element<ET, N>::type;

template <class ET>
struct kokkos_to_mdspan_element
    : std::conditional_t<
              std::is_pointer_v<std::decay_t<ET>>,
              kokkos_to_mdspan_element<std::remove_pointer_t<std::decay_t<ET>>>,
              type_holder<ET>>
{
};

/// Alias template to transform a Kokkos element type to a mdspan element type
/// Only dynamic dimensions is supported for now i.e. `double[4]*` is not yet covered.
template <class ET>
using kokkos_to_mdspan_element_t = typename kokkos_to_mdspan_element<ET>::type;


template <std::size_t... Is>
KOKKOS_FUNCTION Kokkos::LayoutStride make_layout_stride(
        std::array<std::size_t, sizeof...(Is)> const& interleaved_extents_strides,
        std::index_sequence<Is...>)
{
    return Kokkos::LayoutStride(interleaved_extents_strides[Is]...);
}

template <class EP, class MP, std::size_t... Is>
KOKKOS_FUNCTION mdspan_to_kokkos_layout_t<typename MP::layout_type> build_kokkos_layout(
        EP const& ep,
        MP const& mapping,
        std::index_sequence<Is...>)
{
    using kokkos_layout_type = mdspan_to_kokkos_layout_t<typename MP::layout_type>;
    if constexpr (std::is_same_v<kokkos_layout_type, Kokkos::LayoutStride>) {
        std::array<std::size_t, sizeof...(Is) * 2> storage;
        Kokkos::mdspan<
                std::size_t,
                Kokkos::extents<std::size_t, sizeof...(Is), 2>,
                Kokkos::layout_right> const interleaved_extents_strides(storage.data());
        ((interleaved_extents_strides(Is, 0) = ep.extent(Is),
          interleaved_extents_strides(Is, 1) = mapping.stride(Is)),
         ...);
        return make_layout_stride(storage, std::make_index_sequence<sizeof...(Is) * 2> {});
    } else {
        return kokkos_layout_type(ep.extent(Is)...);
    }
}

// If is Kokkos::LayoutLeft/Kokkos::LayoutRight
// check it is contiguous to be compatible with Kokkos::layout_left/Kokkos::layout_right
template <class DataType, class... Properties>
KOKKOS_FUNCTION bool is_kokkos_layout_compatible(Kokkos::View<DataType, Properties...> const& view)
{
    using layout_type = typename Kokkos::View<DataType, Properties...>::array_layout;
    if (std::is_same_v<layout_type, Kokkos::LayoutLeft>
        || std::is_same_v<layout_type, Kokkos::LayoutRight>) {
        return view.span_is_contiguous();
    }
    return true;
}

template <class DataType, class... Properties, std::size_t... Is>
KOKKOS_FUNCTION auto build_mdspan(
        Kokkos::View<DataType, Properties...> const view,
        std::index_sequence<Is...>)
{
    assert(is_kokkos_layout_compatible(view));
    using element_type = kokkos_to_mdspan_element_t<DataType>;
    using extents_type = Kokkos::dextents<std::size_t, Kokkos::View<DataType, Properties...>::rank>;
    using layout_type = kokkos_to_mdspan_layout_t<
            typename Kokkos::View<DataType, Properties...>::array_layout>;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    extents_type const exts(view.extent(Is)...);
    if constexpr (std::is_same_v<layout_type, Kokkos::layout_stride>) {
        return Kokkos::mdspan(view.data(), mapping_type(exts, std::array {view.stride(Is)...}));
    } else {
        return Kokkos::
                mdspan<element_type, extents_type, layout_type>(view.data(), mapping_type(exts));
    }
}

} // namespace ddc::detail
