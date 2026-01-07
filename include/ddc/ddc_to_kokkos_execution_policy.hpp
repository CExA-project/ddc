// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>

#include <Kokkos_Core.hpp>

#include "discrete_vector.hpp"

namespace ddc::detail {

template <class ExecSpace, std::size_t N>
auto ddc_to_kokkos_execution_policy(
        ExecSpace const& execution_space,
        std::array<DiscreteVectorElement, N> const& size)
{
    using work_tag = void;
    using index_type = Kokkos::IndexType<DiscreteVectorElement>;
    if constexpr (N == 0) {
        return Kokkos::RangePolicy<ExecSpace, work_tag, index_type>(execution_space, 0, 1);
    } else {
        if constexpr (N == 1) {
            return Kokkos::
                    RangePolicy<ExecSpace, work_tag, index_type>(execution_space, 0, size[0]);
        } else {
            using iteration_pattern
                    = Kokkos::Rank<N, Kokkos::Iterate::Right, Kokkos::Iterate::Right>;
            Kokkos::Array<DiscreteVectorElement, N> const begin {};
            Kokkos::Array<DiscreteVectorElement, N> end;
            for (std::size_t i = 0; i < N; ++i) {
                end[i] = size[i];
            }
            return Kokkos::MDRangePolicy<
                    ExecSpace,
                    iteration_pattern,
                    work_tag,
                    index_type>(execution_space, begin, end);
        }
    }
}

} // namespace ddc::detail
