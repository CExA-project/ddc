// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/chunk_span.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/detail/tagged_vector.hpp"
#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_vector.hpp"


namespace detail {

template <class F, class... DDims>
class ForEachKokkosLambdaAdapter
{
    template <class T>
    using index_type = std::size_t;

    F m_f;

public:
    ForEachKokkosLambdaAdapter(F const& f) : m_f(f) {}

    DDC_FORCEINLINE_FUNCTION void operator()(index_type<DDims>... ids) const
    {
        m_f(DiscreteCoordinate<DDims...>(ids...));
    }
};

template <class Functor, class DDim0>
inline void for_each_kokkos(DiscreteDomain<DDim0> const& domain, Functor const& f) noexcept
{
    Kokkos::parallel_for(
            Kokkos::RangePolicy<>(
                    select<DDim0>(domain).front().uid(),
                    select<DDim0>(domain).back().uid() + 1),
            ForEachKokkosLambdaAdapter<Functor, DDim0>(f));
}

template <class Functor, class DDim0, class DDim1, class... DDims>
inline void for_each_kokkos(
        DiscreteDomain<DDim0, DDim1, DDims...> const& domain,
        Functor&& f) noexcept
{
    Kokkos::Array<std::size_t, 2 + sizeof...(DDims)> const
            begin {select<DDim0>(domain).front().uid(),
                   select<DDim1>(domain).front().uid(),
                   select<DDims>(domain).front().uid()...};
    Kokkos::Array<std::size_t, 2 + sizeof...(DDims)> const
            end {(select<DDim0>(domain).back().uid() + 1),
                 (select<DDim1>(domain).back().uid() + 1),
                 (select<DDims>(domain).back().uid() + 1)...};
    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2 + sizeof...(DDims)>>(begin, end),
            ForEachKokkosLambdaAdapter<Functor, DDim0, DDim1, DDims...>(f));
}

template <class RetType, class Element, std::size_t N, class Functor, class... Is>
inline void for_each_serial(
        std::array<Element, N> const& start,
        std::array<Element, N> const& end,
        Functor const& f,
        Is const&... is) noexcept
{
    static constexpr std::size_t I = sizeof...(Is);
    if constexpr (I == N) {
        f(RetType(is...));
    } else {
        for (Element ii = start[I]; ii <= end[I]; ++ii) {
            for_each_serial<RetType>(start, end, f, is..., ii);
        }
    }
}

template <class RetType, class Element, std::size_t N, class Functor>
inline void for_each_omp(
        std::array<Element, N> const& start,
        std::array<Element, N> const& end,
        Functor&& f) noexcept
{
    Element const ib = start[0];
    Element const ie = end[0];
#pragma omp parallel for default(none) shared(ib, ie, start, end, f)
    for (Element ii = ib; ii <= ie; ++ii) {
        if constexpr (N == 1) {
            f(RetType(ii));
        } else {
            detail::for_each_serial<RetType>(start, end, f, ii);
        }
    }
}

} // namespace detail

/// Serial execution
struct serial_policy
{
};

/** iterates over a nD domain using the serial execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(serial_policy, DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    detail::for_each_serial<DiscreteCoordinate<DDims...>>(
            detail::array(domain.front()),
            detail::array(domain.back()),
            std::forward<Functor>(f));
}

/** iterates over a nD extent using the serial execution policy
 * @param[in] extent the extent over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each_n(serial_policy, DiscreteVector<DDims...> const& extent, Functor&& f) noexcept
{
    detail::for_each_serial<DiscreteVector<DDims...>>(
            std::array<DiscreteVectorElement, sizeof...(DDims)> {},
            std::array<DiscreteVectorElement, sizeof...(DDims)> {get<DDims>(extent) - 1 ...},
            std::forward<Functor>(f));
}

/// OpenMP parallel execution on the outer loop with default scheduling
struct omp_policy
{
};

/** iterates over a nD domain using the OpenMP execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(omp_policy, DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    detail::for_each_omp<DiscreteCoordinate<DDims...>>(
            detail::array(domain.front()),
            detail::array(domain.back()),
            std::forward<Functor>(f));
}

/** iterates over a nD extent using the OpenMP execution policy
 * @param[in] extent the extent over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each_n(omp_policy, DiscreteVector<DDims...> const& extent, Functor&& f) noexcept
{
    detail::for_each_omp<DiscreteVector<DDims...>>(
            std::array<DiscreteVectorElement, sizeof...(DDims)> {},
            std::array<DiscreteVectorElement, sizeof...(DDims)> {get<DDims>(extent) - 1 ...},
            std::forward<Functor>(f));
}

/// Kokkos parallel execution uisng MDRange policy
struct kokkos_policy
{
};

/** iterates over a nD domain using the serial execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(kokkos_policy, DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    detail::for_each_kokkos(domain, std::forward<Functor>(f));
}

using default_policy = serial_policy;

namespace policies {

inline constexpr omp_policy omp;
inline constexpr serial_policy serial;
inline constexpr kokkos_policy kokkos;

}; // namespace policies

/** iterates over a nD domain using the default execution policy
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    for_each(default_policy(), domain, std::forward<Functor>(f));
}

/** iterates over a nD extent using the default execution policy
 * @param[in] extent the extent over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each_n(DiscreteVector<DDims...> const& extent, Functor&& f) noexcept
{
    for_each_n(default_policy(), extent, std::forward<Functor>(f));
}

template <
        class ExecutionPolicy,
        class ElementType,
        class... DDims,
        class LayoutPolicy,
        class Functor>
inline void for_each_elem(
        ExecutionPolicy&& policy,
        ChunkSpan<ElementType, DiscreteDomain<DDims...>, LayoutPolicy> chunk_span,
        Functor&& f) noexcept
{
    for_each(std::forward<ExecutionPolicy>(policy), chunk_span.domain(), std::forward<Functor>(f));
}

template <class ElementType, class... DDims, class LayoutPolicy, class Functor>
inline void for_each_elem(
        ChunkSpan<ElementType, DiscreteDomain<DDims...>, LayoutPolicy> chunk_span,
        Functor&& f) noexcept
{
    for_each(chunk_span.domain(), std::forward<Functor>(f));
}
