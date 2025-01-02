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

template <class ExecSpace, class... DDims>
auto ddc_to_kokkos_execution_policy(
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain)
{
    using work_tag = void;
    using index_type = Kokkos::IndexType<DiscreteElementType>;
    if constexpr (sizeof...(DDims) == 0) {
        return Kokkos::RangePolicy<ExecSpace, work_tag, index_type>(execution_space, 0, 1);
    } else {
        DiscreteElement<DDims...> const ddc_begin = domain.front();
        DiscreteElement<DDims...> const ddc_end = domain.front() + domain.extents();
        if constexpr (sizeof...(DDims) == 1) {
            std::size_t const begin = ddc_begin.uid();
            std::size_t const end = ddc_end.uid();
            return Kokkos::
                    RangePolicy<ExecSpace, work_tag, index_type>(execution_space, begin, end);
        } else {
            using iteration_pattern = Kokkos::
                    Rank<sizeof...(DDims), Kokkos::Iterate::Right, Kokkos::Iterate::Right>;
            Kokkos::Array<std::size_t, sizeof...(DDims)> const begin {
                    ddc::uid<DDims>(ddc_begin)...};
            Kokkos::Array<std::size_t, sizeof...(DDims)> const end {ddc::uid<DDims>(ddc_end)...};
            return Kokkos::MDRangePolicy<
                    ExecSpace,
                    iteration_pattern,
                    work_tag,
                    index_type>(execution_space, begin, end);
        }
    }
}

} // namespace ddc::detail
