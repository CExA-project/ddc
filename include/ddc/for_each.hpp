// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "ddc/chunk_span.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"

#include "ddc/detail/kokkos.hpp"
#include "ddc/detail/uid_to_element_adapter.hpp"

namespace ddc {

namespace detail {

template <class RetType, class Element, ::std::size_t N, class Functor, class... Is>
KOKKOS_IMPL_FORCEINLINE void for_each_impl(
        ::std::array<Element, N> const& begin,
        ::std::array<Element, N> const& end,
        Functor&& f,
        Is... is) noexcept(noexcept(f(::std::declval<RetType>())))
{
    static constexpr ::std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(RetType(is...));
    } else {
        for (Element ii = begin[I]; ii < end[I]; ++ii) {
            for_each_impl<RetType>(begin, end, ::std::forward<Functor>(f), is..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking a discrete element as parameter
 */
template <class... DDims, class Functor>
KOKKOS_IMPL_FORCEINLINE void for_each(
        ::ddc::DiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept(noexcept(f(::std::declval<::ddc::DiscreteElement<DDims...>>())))
{
    auto&& begin = ::ddc::detail::array(domain.front());
    auto&& end = ::ddc::detail::array(domain.front() + domain.extents());
    ::ddc::detail::for_each_impl<
            ::ddc::DiscreteElement<DDims...>>(begin, end, ::std::forward<Functor>(f));
}

} // namespace ddc
