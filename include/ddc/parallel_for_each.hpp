// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/ddc_to_kokkos_execution_policy.hpp"
#include "ddc/detail/kokkos.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"

namespace ddc {

namespace detail {

template <class F, class... DDims>
class ForEachKokkosLambdaAdapter
{
    template <class T>
    using index_type = DiscreteElementType;

    F m_f;

public:
    explicit ForEachKokkosLambdaAdapter(F const& f) : m_f(f) {}

    template <std::size_t N = sizeof...(DDims), std::enable_if_t<(N == 0), bool> = true>
    KOKKOS_FUNCTION void operator()([[maybe_unused]] index_type<void> unused_id) const
    {
        m_f(DiscreteElement<>());
    }

    template <std::size_t N = sizeof...(DDims), std::enable_if_t<(N > 0), bool> = true>
    KOKKOS_FUNCTION void operator()(index_type<DDims>... ids) const
    {
        m_f(DiscreteElement<DDims...>(ids...));
    }
};

template <class ExecSpace, class Functor, class... DDims>
void for_each_kokkos(
        std::string const& label,
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain,
        Functor const& f) noexcept
{
    Kokkos::parallel_for(
            label,
            ddc_to_kokkos_execution_policy(execution_space, domain),
            ForEachKokkosLambdaAdapter<Functor, DDims...>(f));
}

} // namespace detail

/** iterates over a nD domain using a given `Kokkos` execution space
 * @param[in] label  name for easy identification of the parallel_for_each algorithm
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class ExecSpace, class... DDims, class Functor>
void parallel_for_each(
        std::string const& label,
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept
{
    detail::for_each_kokkos(label, execution_space, domain, std::forward<Functor>(f));
}

/** iterates over a nD domain using a given `Kokkos` execution space
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class ExecSpace, class... DDims, class Functor>
std::enable_if_t<Kokkos::is_execution_space_v<ExecSpace>> parallel_for_each(
        ExecSpace const& execution_space,
        DiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept
{
    detail::for_each_kokkos(
            "ddc_for_each_default",
            execution_space,
            domain,
            std::forward<Functor>(f));
}

/** iterates over a nD domain using the `Kokkos` default execution space
 * @param[in] label  name for easy identification of the parallel_for_each algorithm
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
void parallel_for_each(
        std::string const& label,
        DiscreteDomain<DDims...> const& domain,
        Functor&& f) noexcept
{
    parallel_for_each(label, Kokkos::DefaultExecutionSpace(), domain, std::forward<Functor>(f));
}

/** iterates over a nD domain using the `Kokkos` default execution space
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
void parallel_for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    parallel_for_each(
            "ddc_for_each_default",
            Kokkos::DefaultExecutionSpace(),
            domain,
            std::forward<Functor>(f));
}

} // namespace ddc
