// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace ddc::experimental {

struct DiscreteDimension
{
};

struct DiscreteSet
{
};

template <class, class = void>
struct has_discrete_set_type_member : std::false_type
{
};

// specialization recognizes types that do have a nested ::discrete_set_type member:
template <class T>
struct has_discrete_set_type_member<T, std::void_t<typename T::discrete_set_type>> : std::true_type
{
};

template <class T>
inline constexpr bool is_discrete_dimension_v
        = std::is_base_of_v<DiscreteDimension, T>&& has_discrete_set_type_member<T>::value;

template <class T>
inline constexpr bool is_discrete_set_v = std::is_base_of_v<DiscreteSet, T>;

template <class NamedDSet>
using continuous_dimension_t = typename NamedDSet::continuous_dimension_type;

} // namespace ddc::experimental
