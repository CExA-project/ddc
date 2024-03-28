// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/chunk_span.hpp"
#include "ddc/detail/kokkos.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

namespace detail {

template <class RetType, class Element, std::size_t N, class Functor, class... Is>
KOKKOS_FUNCTION void for_each_serial(
        std::array<Element, N> const& begin,
        std::array<Element, N> const& end,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(RetType(is...));
    } else {
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            for_each_serial<RetType>(begin, end, f, is..., ii);
        }
    }
}

template <class RetType, class Element, std::size_t N, class Functor, class... Is>
void host_for_each_serial(
        std::array<Element, N> const& begin,
        std::array<Element, N> const& end,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(RetType(is...));
    } else {
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            host_for_each_serial<RetType>(begin, end, f, is..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain in serial
 * This version must only be called with `f` being a function compiled for all execution spaces. See also \ref host_for_each when this function is only compiled for the host.
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
KOKKOS_FUNCTION void for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    DiscreteElement<DDims...> const ddc_begin = domain.front();
    DiscreteElement<DDims...> const ddc_end = domain.front() + domain.extents();
    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    detail::for_each_serial<DiscreteElement<DDims...>>(begin, end, std::forward<Functor>(f));
}

/** iterates over a nD domain in serial.
 * This version must only be called from the host.
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
void host_for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    DiscreteElement<DDims...> const ddc_begin = domain.front();
    DiscreteElement<DDims...> const ddc_end = domain.front() + domain.extents();
    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    detail::for_each_serial<DiscreteElement<DDims...>>(begin, end, std::forward<Functor>(f));
}

} // namespace ddc
