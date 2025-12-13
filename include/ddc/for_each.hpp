// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_vector.hpp"

namespace ddc {

namespace detail {

template <class Support, class Element, std::size_t N, class Functor, class... Is>
void host_for_each_serial(
        std::array<Element, N> const& begin,
        std::array<Element, N> const& end,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(Support(is...));
    } else {
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            host_for_each_serial<Support>(begin, end, f, is..., ii);
        }
    }
}

template <class Support, class Element, std::size_t N, class Functor, class... Is>
KOKKOS_FUNCTION void device_for_each_serial(
        std::array<Element, N> const& begin,
        std::array<Element, N> const& end,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(Support(is...));
    } else {
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            device_for_each_serial<Support>(begin, end, f, is..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain in serial
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
[[deprecated("Use host_for_each instead")]]
void for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    host_for_each(domain, std::forward<Functor>(f));
}

/** iterates over a nD domain in serial
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
    detail::host_for_each_serial<DiscreteElement<DDims...>>(begin, end, std::forward<Functor>(f));
}

/** iterates over a nD domain in serial. Can be called from a device kernel.
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
KOKKOS_FUNCTION void device_for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    DiscreteElement<DDims...> const ddc_begin = domain.front();
    DiscreteElement<DDims...> const ddc_end = domain.front() + domain.extents();
    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    detail::device_for_each_serial<DiscreteElement<DDims...>>(begin, end, std::forward<Functor>(f));
}

/** iterates over a nD sparse domain in serial. Can be called from a device kernel.
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an discrete vector as parameter
 */
template <class... DDims, class Functor>
KOKKOS_FUNCTION void device_for_each(
        StridedDiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept
{
    using discrete_element_type = typename StridedDiscreteDomain<DDims...>::discrete_element_type;
    using discrete_vector_type = typename StridedDiscreteDomain<DDims...>::discrete_vector_type;
    discrete_element_type const ddc_begin = domain.front();
    discrete_element_type const ddc_end = domain.front() + domain.extents();

    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    detail::device_for_each_serial<discrete_vector_type>(begin, end, std::forward<Functor>(f));
}

/** iterates over a nD sparse domain in serial. Can be called from a device kernel.
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an discrete vector as parameter
 */
template <class... DDims, class Functor>
KOKKOS_FUNCTION void device_for_each(
        SparseDiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept
{
    using discrete_element_type = typename SparseDiscreteDomain<DDims...>::discrete_element_type;
    using discrete_vector_type = typename SparseDiscreteDomain<DDims...>::discrete_vector_type;
    discrete_element_type const ddc_begin = domain.front();
    discrete_element_type const ddc_end = domain.front() + domain.extents();

    std::array const begin = detail::array(ddc_begin);
    std::array const end = detail::array(ddc_end);
    detail::device_for_each_serial<discrete_vector_type>(begin, end, std::forward<Functor>(f));
}

} // namespace ddc
