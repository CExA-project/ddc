// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "ddc/chunk_span.hpp"
#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"

namespace detail {

template <class... DDims, class Functor, class... DCoords>
inline void for_each_impl(
        DiscreteDomain<DDims...> const& domain,
        Functor const& f,
        DCoords const&... dcs) noexcept
{
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        f(DiscreteCoordinate<DDims...> {dcs...});
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        for (auto&& ii : select<CurrentDDim>(domain)) {
            for_each_impl(domain, f, dcs..., ii);
        }
    }
}

template <class... DDims, class Functor, class... DCoords>
inline void for_each_impl(
        DiscreteVector<DDims...> const& domain,
        Functor const& f,
        DCoords const&... dcs) noexcept
{
    if constexpr (sizeof...(DCoords) == sizeof...(DDims)) {
        f(DiscreteVector<DDims...> {dcs...});
    } else {
        using CurrentDDim = type_seq_element_t<sizeof...(DCoords), detail::TypeSeq<DDims...>>;
        for (size_t ii = 0; ii < select<CurrentDDim>(domain); ++ii) {
            for_each_impl(domain, f, dcs..., ii);
        }
    }
}
} // namespace detail

/// Serial execution
struct serial_policy
{
};

/** iterates over a nD domain
 * @param[in] domain the domain over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(serial_policy, DiscreteDomain<DDims...> const& domain, Functor&& f) noexcept
{
    detail::for_each_impl(domain, std::forward<Functor>(f));
}

/** iterates over a nD DiscreteVector
 * @param[in] domain the vector over which to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class... DDims, class Functor>
inline void for_each(serial_policy, DiscreteVector<DDims...> const& domain, Functor&& f) noexcept
{
    detail::for_each_impl(domain, std::forward<Functor>(f));
}

/** OpenMP parallel execution
 * - default scheduling
 * - collapsing in case of tightly nested loops
 */
struct omp_policy
{
};

/** iterates over a 1d domain
 * @param[in] domain the 1d domain to iterate
 * @param[in] f      a functor taking an index as parameter
 */
template <class DDim, class Functor>
inline void for_each(omp_policy, DiscreteDomain<DDim> const& domain, Functor&& f) noexcept
{
    DiscreteDomainIterator<DDim> const it_b = domain.begin();
    DiscreteDomainIterator<DDim> const it_e = domain.end();
#pragma omp parallel for default(none) shared(it_b, it_e, f)
    for (DiscreteDomainIterator<DDim> it = it_b; it != it_e; ++it) {
        f(*it);
    }
}

/** iterates over a 2d domain
 * @param[in] domain the 2d domain to iterate
 * @param[in] f      a functor taking 2 indices as parameter
 */
template <class DDim1, class DDim2, class Functor>
inline void for_each(omp_policy, DiscreteDomain<DDim1, DDim2> const& domain, Functor&& f) noexcept
{
    DiscreteDomainIterator<DDim1> const it1_b = select<DDim1>(domain).begin();
    DiscreteDomainIterator<DDim1> const it1_e = select<DDim1>(domain).end();
    DiscreteDomainIterator<DDim2> const it2_b = select<DDim2>(domain).begin();
    DiscreteDomainIterator<DDim2> const it2_e = select<DDim2>(domain).end();
#pragma omp parallel for collapse(2) default(none) shared(it1_b, it1_e, it2_b, it2_e, f)
    for (DiscreteDomainIterator<DDim1> it1 = it1_b; it1 != it1_e; ++it1) {
        for (DiscreteDomainIterator<DDim2> it2 = it2_b; it2 != it2_e; ++it2) {
            f(DiscreteCoordinate<DDim1, DDim2>(*it1, *it2));
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
