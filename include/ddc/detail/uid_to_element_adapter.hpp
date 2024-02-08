// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"

#include "ddc/detail/kokkos.hpp"

namespace ddc::detail {

/** A class that adapts a functor taking a discrete element as parameters to take integers (UID) instead
 *
 * - If IsDevice is true, the wrapped functor will be annotated so as to be compiled for device too,
 * hence it should contain valid device code
 * - In the 0D case, the wrapping functor will accept a single integer uid that it discards
 *
 * @tparam Functor the type of the functor
 * @tparam DDom the type of the ::ddc::DiscreteDomain that the functor expects as first parameter
 * @tparam IsDevice whether the Functor should be annotated to run on device
 */
template <class Functor, class DDom, bool IsDevice>
struct UidToElementAdapter;

// Implementation of UidToElementAdapter when the execution space does not require device annotation
template <class Functor, class... DDims>
struct UidToElementAdapter<Functor, ::ddc::DiscreteDomain<DDims...>, false>
{
    // hack to expand `ids` below
    template <class T>
    using index_type = ::std::size_t;

    Functor m_f;

    template <class... T>
    KOKKOS_IMPL_FORCEINLINE void operator()(index_type<DDims>... ids, T&&... t) const
            noexcept(noexcept(m_f(::std::declval<::ddc::DiscreteElement<DDims...>>())))
    {
        m_f(::ddc::DiscreteElement<DDims...>(ids...), ::std::forward<T>(t)...);
    }
};

// 0D Implementation of UidToElementAdapter when the execution space does not require device annotation
template <class Functor>
struct UidToElementAdapter<Functor, ::ddc::DiscreteDomain<>, false>
{
    Functor m_f;

    // in the 0D case, we throw away a single integer uid since kokkos always provides one
    template <class... T>
    KOKKOS_IMPL_FORCEINLINE void operator()(::std::size_t, T&&... t) const
            noexcept(noexcept(m_f(::std::declval<::ddc::DiscreteElement<>>())))
    {
        m_f(::ddc::DiscreteElement<>(), ::std::forward<T>(t)...);
    }
};

// Implementation of UidToElementAdapter when the execution space requires device annotation
template <class Functor, class... DDims>
struct UidToElementAdapter<Functor, ::ddc::DiscreteDomain<DDims...>, true>
{
    // hack to expand `ids` below
    template <class T>
    using index_type = ::std::size_t;

    Functor m_f;

    template <class... T>
    KOKKOS_FORCEINLINE_FUNCTION void operator()(index_type<DDims>... ids, T&&... t) const
            noexcept(noexcept(m_f(::std::declval<::ddc::DiscreteElement<DDims...>>())))
    {
        m_f(::ddc::DiscreteElement<DDims...>(ids...), ::std::forward<T>(t)...);
    }
};

// 0D Implementation of UidToElementAdapter when the execution space requires device annotation
template <class Functor>
struct UidToElementAdapter<Functor, ::ddc::DiscreteDomain<>, true>
{
    Functor m_f;

    // in the 0D case, we throw away a single integer uid since kokkos always provides one
    template <class... T>
    KOKKOS_FORCEINLINE_FUNCTION void operator()(::std::size_t, T&&... t) const
            noexcept(noexcept(m_f(::std::declval<::ddc::DiscreteElement<>>())))
    {
        m_f(::ddc::DiscreteElement<>(), ::std::forward<T>(t)...);
    }
};

} // namespace ddc::detail
