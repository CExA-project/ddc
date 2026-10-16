// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <span>
#include <stdexcept>
#include <utility>

#include <Kokkos_Core.hpp>

#include "discrete_vector.hpp"
#include "parallel_for_each_team.hpp"

namespace ddc::detail {

int team_league_size(std::span<DiscreteVectorElement const> extents)
{
    DiscreteVectorElement league_size = 1;
    for (DiscreteVectorElement const extent : extents) {
        league_size *= extent;
    }

    if (std::in_range<int>(league_size)) {
        return static_cast<int>(league_size);
    }

    throw std::runtime_error(
            "ddc::detail::team_league_size: the domain is too large for a Kokkos team policy");
}

} // namespace ddc::detail
