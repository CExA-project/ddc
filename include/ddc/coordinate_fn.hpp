// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/coordinate.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"

namespace ddc {

template <class... PointSampling>
DDC_INLINE_FUNCTION Coordinate<typename PointSampling::continuous_dimension_type...> coordinate(
        DiscreteElement<PointSampling...> const& c)
{
    return Coordinate<typename PointSampling::continuous_dimension_type...>(
            discrete_space<PointSampling>().coordinate(select<PointSampling>(c))...);
}

} // namespace ddc
