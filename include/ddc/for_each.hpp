// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "discrete_element.hpp"
#include "discrete_vector.hpp"

namespace ddc {

namespace detail {

template <class Support, class Element, std::size_t N, class Functor, class... Is>
void host_for_each_serial(
        Support const& support,
        std::array<Element, N> const& size,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(support(typename Support::discrete_vector_type(is...)));
    } else {
        for (Element ii = 0; ii < size[I]; ++ii) {
            host_for_each_serial(support, size, f, is..., ii);
        }
    }
}

template <class Support, class Element, std::size_t N, class Functor, class... Is>
KOKKOS_FUNCTION void device_for_each_serial(
        Support const& support,
        std::array<Element, N> const& size,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(support(typename Support::discrete_vector_type(is...)));
    } else {
        for (Element ii = 0; ii < size[I]; ++ii) {
            device_for_each_serial(support, size, f, is..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class Support, class Functor>
[[deprecated("Use host_for_each instead")]]
void for_each(Support const& domain, Functor&& f) noexcept
{
    host_for_each(domain, std::forward<Functor>(f));
}

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class Support, class Functor>
void host_for_each(Support const& domain, Functor&& f) noexcept
{
    std::array const size = detail::array(domain.extents());
    detail::host_for_each_serial(domain, size, std::forward<Functor>(f));
}

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class Support, class Functor>
KOKKOS_FUNCTION void device_for_each(Support const& domain, Functor&& f) noexcept
{
    std::array const size = detail::array(domain.extents());
    detail::device_for_each_serial(domain, size, std::forward<Functor>(f));
}

} // namespace ddc
