#pragma once

#include <utility>

#include <Kokkos_Core.hpp>

#include "ddc/chunk_span.hpp"
#include "ddc/discrete_coordinate.hpp"
#include "ddc/discrete_domain.hpp"

template <class DDim1, class DDim2, class Functor>
inline void for_each(DiscreteDomain<DDim1, DDim2> const& domain, Functor&& f) noexcept
{
    Kokkos::Array<std::size_t, 2> const
            begin {select<DDim1>(domain).front().value(), select<DDim2>(domain).front().value()};
    Kokkos::Array<std::size_t, 2> const
            end {select<DDim1>(domain).back().value() + 1,
                 select<DDim2>(domain).back().value() + 1};
    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(begin, end),
            KOKKOS_LAMBDA(std::size_t i, std::size_t j) {
                f(DiscreteCoordinate<DDim1, DDim2>(i, j));
            });
}

template <class ElementType, class... DDims, class LayoutPolicy, class Functor>
inline void for_each_elem(
        ChunkSpan<ElementType, DiscreteDomain<DDims...>, LayoutPolicy> chunk_span,
        Functor&& f) noexcept
{
    for_each(chunk_span.domain(), std::forward<Functor>(f));
}
