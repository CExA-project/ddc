// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"

#include "ddc/detail/kokkos.hpp"
#include "ddc/detail/uid_to_element_adapter.hpp"

namespace ddc {

/** iterates in parallel over a 0D domain using Kokkos
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking a discrete element as parameter
 */
template <class ExecSpace, class Functor>
KOKKOS_IMPL_FORCEINLINE void parallel_each(
        ExecSpace&& execution_space,
        [[maybe_unused]] ::ddc::DiscreteDomain<> const& domain,
        Functor&& f) noexcept(noexcept(f(::std::declval<::ddc::DiscreteElement<>>())))
{
    using AdaptedFunctor = ::ddc::detail::UidToElementAdapter<
            Functor,
            ::ddc::DiscreteDomain<>,
            ::ddc::detail::need_annotated_operator<ExecSpace>()>;
    Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>(::std::forward<ExecSpace>(execution_space), 0, 1),
            AdaptedFunctor(::std::forward<Functor>(f)));
}

/** iterates in parallel over a 1D domain using Kokkos
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking a discrete element as parameter
 */
template <class ExecSpace, class Functor, class DDim0>
KOKKOS_IMPL_FORCEINLINE void parallel_each(
        ExecSpace&& execution_space,
        ::ddc::DiscreteDomain<DDim0> const& domain,
        Functor&& f) noexcept(noexcept(f(::std::declval<::ddc::DiscreteElement<DDim0>>())))
{
    using AdaptedFunctor = ::ddc::detail::UidToElementAdapter<
            Functor,
            ::ddc::DiscreteDomain<DDim0>,
            ::ddc::detail::need_annotated_operator<ExecSpace>()>;
    ::ddc::DiscreteElement<DDim0> const ddc_begin = domain.front();
    ::ddc::DiscreteElement<DDim0> const ddc_end = domain.front() + domain.extents();
    ::std::size_t const begin = ddc::uid<DDim0>(ddc_begin);
    ::std::size_t const end = ddc::uid<DDim0>(ddc_end);
    Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>(::std::forward<ExecSpace>(execution_space), begin, end),
            AdaptedFunctor(::std::forward<Functor>(f)));
}

/** iterates in parallel over a nD domain using Kokkos
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking a discrete element as parameter
 */
template <class ExecSpace, class Functor, class DDim0, class DDim1, class... DDims>
KOKKOS_IMPL_FORCEINLINE void parallel_each(
        ExecSpace&& execution_space,
        ::ddc::DiscreteDomain<DDim0, DDim1, DDims...> const& domain,
        Functor&& f) noexcept(noexcept(f(::std::
                                                 declval<::ddc::DiscreteElement<
                                                         DDim0,
                                                         DDim1,
                                                         DDims...>>())))
{
    using AdaptedFunctor = ::ddc::detail::UidToElementAdapter<
            Functor,
            ::ddc::DiscreteDomain<DDim0, DDim1, DDims...>,
            ::ddc::detail::need_annotated_operator<ExecSpace>()>;
    using ExecutionPolicy = Kokkos::MDRangePolicy<
            ExecSpace,
            Kokkos::Rank<2 + sizeof...(DDims), Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
    ::ddc::DiscreteElement<DDim0, DDim1, DDims...> const ddc_begin = domain.front();
    ::ddc::DiscreteElement<DDim0, DDim1, DDims...> const ddc_end
            = domain.front() + domain.extents();
    Kokkos::Array<::std::size_t, 2 + sizeof...(DDims)> const
            begin {ddc::uid<DDim0>(ddc_begin),
                   ddc::uid<DDim1>(ddc_begin),
                   ddc::uid<DDims>(ddc_begin)...};
    Kokkos::Array<::std::size_t, 2 + sizeof...(DDims)> const
            end {ddc::uid<DDim0>(ddc_end), ddc::uid<DDim1>(ddc_end), ddc::uid<DDims>(ddc_end)...};
    Kokkos::parallel_for(
            ExecutionPolicy(::std::forward<ExecSpace>(execution_space), begin, end),
            AdaptedFunctor(::std::forward<Functor>(f)));
}

/** iterates in parallel over a nD domain using Kokkos
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking a discrete element as parameter
 */
template <class ExecSpace, class Functor, class... DDims>
KOKKOS_IMPL_FORCEINLINE void parallel_each(
        ::ddc::DiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept(noexcept(f(::std::declval<::ddc::DiscreteElement<DDims...>>())))
{
    parallel_each(ExecSpace(), domain, ::std::forward<Functor>(f));
}

/** iterates in parallel over a nD domain using Kokkos
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking a discrete element as parameter
 */
template <class Functor, class... DDims>
KOKKOS_IMPL_FORCEINLINE void parallel_each(
        ::ddc::DiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept(noexcept(f(::std::declval<::ddc::DiscreteElement<DDims...>>())))
{
    parallel_each<Kokkos::DefaultExecutionSpace>(domain, ::std::forward<Functor>(f));
}

} // namespace ddc
