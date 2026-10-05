// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <Kokkos_Macros.hpp>

#include "detail/tagged_vector.hpp"

#include "discrete_dimension.hpp"
#include "discrete_element.hpp"
#include "real_type.hpp"

namespace ddc {

/** A CoordinateElement the type of the scalar used to represent elements of coordinates in the
 * continuous space.
 */
using CoordinateElement = Real;

/** A Coordinate represents a coordinate in the continuous space
 *
 * It is tagged by its dimensions.
 */

template <class... CDims>
using Coordinate = detail::TaggedVector<CoordinateElement, CDims...>;

template <concepts::discrete_dimension... DDims>
KOKKOS_FUNCTION Coordinate<typename DDims::continuous_dimension_type...> coordinate(
        DiscreteElement<DDims...> const& c)
    requires(sizeof...(DDims) > 1)
{
    return Coordinate<typename DDims::continuous_dimension_type...>(
            coordinate(DiscreteElement<DDims>(c))...);
}

// Gives access to the type of the coordinates of a discrete element
// Example usage : "using Coords = coordinate_of_t<DElem>;"
template <concepts::discrete_element T>
struct coordinate_of
{
    using type = decltype(coordinate(std::declval<T>()));
};

/// Helper type of \ref ddc::coordinate_of
template <concepts::discrete_element T>
using coordinate_of_t = coordinate_of<T>::type;

} // namespace ddc
