// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/experimental/concepts.hpp"

namespace ddc::experimental {

/// New discrete dimension: Node
// Design choices:
// - make it external to ease template deduction (for example see `distance_at_left`)
// - assign them a `discrete_set_type` (mandatory, shall be checked by `is_discrete_dimension_v`) so that they cannot be reused in an other DiscreteSet, can be `void`
template <class NamedDDim>
struct Node : DiscreteDimension
{
    using named_discrete_set_type = NamedDDim;
    using discrete_set_type = typename NamedDDim::discrete_set_type;
};

/// New discrete dimension: Cell
template <class NamedDDim>
struct Cell : DiscreteDimension
{
    using named_discrete_set_type = NamedDDim;
    using discrete_set_type = typename NamedDDim::discrete_set_type;
};

} // namespace ddc::experimental
