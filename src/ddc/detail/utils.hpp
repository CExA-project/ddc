// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Macros.hpp>

namespace ddc::detail {

template <class OutputType, class InputType, std::size_t... Idx>
KOKKOS_FUNCTION constexpr std::array<OutputType, sizeof...(Idx)> convert_array_to(
        std::array<InputType, sizeof...(Idx)> const& values,
        std::index_sequence<Idx...>) noexcept
{
    static_assert(std::is_convertible_v<InputType, OutputType>);
    if constexpr (std::is_same_v<InputType, OutputType>) {
        return values;
    } else {
        return {static_cast<OutputType>(std::get<Idx>(values))...};
    }
}

} // namespace ddc::detail
