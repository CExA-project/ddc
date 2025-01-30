// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

namespace detail {

template <class Support, class Element, std::size_t N, class Functor, class... Is>
void for_each_serial(
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
            for_each_serial(support, size, f, is..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class Support, class Functor>
void for_each(Support const& domain, Functor&& f) noexcept
{
    std::array const size = detail::array(domain.extents());
    detail::for_each_serial(domain, size, std::forward<Functor>(f));
}

} // namespace ddc
