// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/coordinate.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"

namespace ddc {

template <class DDim0, class DDim1, class... DDims>
DDC_INLINE_FUNCTION Coordinate<
        typename DDim0::continuous_dimension_type,
        typename DDim1::continuous_dimension_type,
        typename DDims::continuous_dimension_type...>
coordinate(DiscreteElement<DDim0, DDim1, DDims...> const& c)
{
    return Coordinate<
            typename DDim0::continuous_dimension_type,
            typename DDim1::continuous_dimension_type,
            typename DDims::continuous_dimension_type...>(
            coordinate(select<DDim0>(c)),
            coordinate(select<DDim1>(c)),
            coordinate(select<DDims>(c))...);
}

} // namespace ddc
