// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/kokkos.hpp"

#include "ddc_to_kokkos_execution_policy.hpp"
#include "discrete_domain.hpp"
#include "discrete_element.hpp"

namespace ddc {

namespace detail {

template <class F, class Support, class IndexSequence>
class ForEachKokkosLambdaAdapter;

template <class F, class Support, std::size_t... Idx>
class ForEachKokkosLambdaAdapter<F, Support, std::index_sequence<Idx...>>
{
    template <std::size_t I>
    using index_type = DiscreteElementType;

    F m_f;

    Support m_support;

public:
    explicit ForEachKokkosLambdaAdapter(F const& f, Support const& support)
        : m_f(f)
        , m_support(support)
    {
    }

    template <std::size_t N = sizeof...(Idx), std::enable_if_t<(N == 0), bool> = true>
    KOKKOS_FUNCTION void operator()([[maybe_unused]] index_type<0> unused_id) const
    {
        m_f(DiscreteElement<>());
    }

    template <std::size_t N = sizeof...(Idx), std::enable_if_t<(N > 0), bool> = true>
    KOKKOS_FUNCTION void operator()(index_type<Idx>... ids) const
    {
        m_f(m_support(typename Support::discrete_vector_type(ids...)));
    }
};

template <class ExecSpace, class Support, class Functor>
void for_each_kokkos(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        Functor const& f) noexcept
{
    Kokkos::parallel_for(
            label,
            ddc_to_kokkos_execution_policy(execution_space, domain),
            ForEachKokkosLambdaAdapter<
                    Functor,
                    Support,
                    std::make_index_sequence<Support::rank()>>(f, domain));
}

} // namespace detail

/** iterates over a nD domain using a given `Kokkos` execution space
 * @param[in] label  name for easy identification of the parallel_for_each algorithm
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class ExecSpace, class Support, class Functor>
void parallel_for_each(
        std::string const& label,
        ExecSpace const& execution_space,
        Support const& domain,
        Functor&& f) noexcept
{
    detail::for_each_kokkos(label, execution_space, domain, std::forward<Functor>(f));
}

/** iterates over a nD domain using a given `Kokkos` execution space
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class ExecSpace, class Support, class Functor>
std::enable_if_t<Kokkos::is_execution_space_v<ExecSpace>> parallel_for_each(
        ExecSpace const& execution_space,
        Support const& domain,
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
template <class Support, class Functor>
void parallel_for_each(std::string const& label, Support const& domain, Functor&& f) noexcept
{
    parallel_for_each(label, Kokkos::DefaultExecutionSpace(), domain, std::forward<Functor>(f));
}

/** iterates over a nD domain using the `Kokkos` default execution space
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class Support, class Functor>
void parallel_for_each(Support const& domain, Functor&& f) noexcept
{
    parallel_for_each(
            "ddc_for_each_default",
            Kokkos::DefaultExecutionSpace(),
            domain,
            std::forward<Functor>(f));
}

} // namespace ddc
