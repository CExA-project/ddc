// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "ddc/chunk_span.hpp"
#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"

/// Serial execution
struct serial_policy
{
};

namespace detail {

template <class... DDims, class Functor, class... DCoords>
inline void for_each_serial(
        DiscreteDomain<DDims...> const& domain,
        Functor const& f,
        DCoords const&... dcoords) noexcept
{
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        f(DiscreteCoordinate<DDims...>(dcoords...));
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        for (DiscreteCoordinate<CurrentDDim> const ii : select<CurrentDDim>(domain)) {
            for_each_serial(domain, f, dcoords..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nD domain
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(serial_policy, DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    detail::for_each_serial(domain, std::forward<Functor>(f));
}

/** OpenMP parallel execution with default scheduling
 */
struct omp_policy
{
};

namespace detail {

template <class... DDims, class Functor, class... DCoords>
inline void for_each_omp(
        DiscreteDomain<DDims...> const& domain,
        Functor const& f,
        DCoords const&... dcoords) noexcept
{
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        f(DiscreteCoordinate<DDims...>(dcoords...));
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        for (DiscreteCoordinate<CurrentDDim> const ii : select<CurrentDDim>(domain)) {
            for_each_omp(domain, f, dcoords..., ii);
        }
    }
}

} // namespace detail

/** iterates over a nd domain
 * @param[in] domain the nd domain to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(omp_policy, DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    using FirstDDim = type_seq_element_t<0, detail::TypeSeq<DDims...>>;
    DiscreteDomainIterator<FirstDDim> const it_b = select<FirstDDim>(domain).begin();
    DiscreteDomainIterator<FirstDDim> const it_e = select<FirstDDim>(domain).end();
#pragma omp parallel for default(none) shared(it_b, it_e, domain, f)
    for (DiscreteDomainIterator<FirstDDim> it = it_b; it != it_e; ++it) {
        if constexpr (sizeof...(DDims) == 1) {
            f(*it);
        } else {
            detail::for_each_omp(domain, f, *it);
        }
    }
}

using default_policy = serial_policy;

namespace policies {

inline constexpr omp_policy omp;
inline constexpr serial_policy serial;

}; // namespace policies

template <class... DDims, class Functor>
inline void for_each(DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    for_each(default_policy(), domain, std::forward<Functor>(f));
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
    for_each(default_policy(), chunk_span.domain(), std::forward<Functor>(f));
}
