// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "ddc/detail/kokkos.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"

namespace ddc::detail {

template <class ExecSpace, class Support>
auto ddc_to_kokkos_execution_policy(ExecSpace const& execution_space, Support const& domain)
{
    using work_tag = void;
    using index_type = Kokkos::IndexType<DiscreteElementType>;
    if constexpr (Support::rank() == 0) {
        return Kokkos::RangePolicy<ExecSpace, work_tag, index_type>(execution_space, 0, 1);
    } else {
        if constexpr (Support::rank() == 1) {
            return Kokkos::RangePolicy<
                    ExecSpace,
                    work_tag,
                    index_type>(execution_space, 0, domain.extents().value());
        } else {
            using iteration_pattern
                    = Kokkos::Rank<Support::rank(), Kokkos::Iterate::Right, Kokkos::Iterate::Right>;
            Kokkos::Array<std::size_t, Support::rank()> const begin {};
            std::array const end = detail::array(domain.extents());
            Kokkos::Array<std::size_t, Support::rank()> end2;
            for (int i = 0; i < Support::rank(); ++i) {
                end2[i] = end[i];
            }
            return Kokkos::MDRangePolicy<
                    ExecSpace,
                    iteration_pattern,
                    work_tag,
                    index_type>(execution_space, begin, end2);
        }
    }
}

} // namespace ddc::detail
