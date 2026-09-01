// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

namespace ddc::testing {

template <class View, class T>
class CountFn
{
    View m_view;

    T m_value;

public:
    using index_type = std::size_t;

    using result_type = std::ptrdiff_t;

    CountFn(View view, T value) : m_view(std::move(view)), m_value(std::move(value)) {}

    KOKKOS_FUNCTION
    void operator()(index_type i, result_type& local_sum) const noexcept
    {
        if (m_view(i) == m_value) {
            ++local_sum;
        }
    }
};

template <class ExecutionSpace, class DataType, class... Properties, class T>
    requires(Kokkos::is_execution_space_v<ExecutionSpace>)
auto count(
        ExecutionSpace const& ex,
        Kokkos::View<DataType*, Properties...> const& view,
        T const& value) -> std::ptrdiff_t
{
    using result_type = std::ptrdiff_t;
    using index_type = std::size_t;

    result_type sum = 0;
    Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<index_type>> const
            policy(ex, 0, view.extent(0));
    Kokkos::parallel_reduce(policy, CountFn(view, value), sum);
    return sum;
}

} // namespace ddc::testing
