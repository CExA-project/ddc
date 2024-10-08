// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <ddc/ddc.hpp>

#include "bsplines_uniform.hpp"
#include "knot_discrete_dimension_type.hpp"
#include "spline_boundary_conditions.hpp"

namespace ddc {

/**
 * @brief Helper class for the initialisation of the mesh of interpolation points.
 *
 * A helper class for the initialisation of the mesh of interpolation points. This
 * class should be used when the interpolation points should be located at the
 * knots of the spline. This is possible with any kind of boundary condition except
 * Greville boundary conditions (as there will not be enough interpolation points).
 * In the case of strongly non-uniform splines this choice may result in a less
 * well conditioned problem, however most mathematical stability results are proven
 * with this choice of interpolation points.
 *
 * @tparam BSplines The type of the uniform or non-uniform spline basis whose knots are used as interpolation points.
 * @tparam BcLower The lower boundary condition.
 * @tparam BcLower The upper boundary condition.
 */
template <class BSplines, ddc::BoundCond BcLower, ddc::BoundCond BcUpper>
class KnotsAsInterpolationPoints
{
    static_assert(BcLower != ddc::BoundCond::GREVILLE);
    static_assert(BcUpper != ddc::BoundCond::GREVILLE);

    using continuous_dimension_type = typename BSplines::continuous_dimension_type;

public:
    /**
     * Get the sampling of interpolation points.
     *
     * @return sampling The DDC point sampling of the interpolation points.
     */
    template <typename Sampling, typename U = BSplines>
    static auto get_sampling()
    {
        if constexpr (U::is_uniform()) {
            return std::get<0>(Sampling::
                                       init(ddc::discrete_space<BSplines>().rmin(),
                                            ddc::discrete_space<BSplines>().rmax(),
                                            ddc::DiscreteVector<Sampling>(
                                                    ddc::discrete_space<BSplines>().ncells() + 1)));
        } else {
            using SamplingImpl = typename Sampling::template Impl<Sampling, Kokkos::HostSpace>;
            std::vector<double> knots(ddc::discrete_space<BSplines>().npoints());
            ddc::DiscreteDomain<knot_discrete_dimension_t<BSplines>> break_point_domain(
                    ddc::discrete_space<BSplines>().break_point_domain());
            ddc::for_each(
                    break_point_domain,
                    [&](ddc::DiscreteElement<knot_discrete_dimension_t<BSplines>> ik) {
                        knots[ik - break_point_domain.front()] = ddc::coordinate(ik);
                    });
            return SamplingImpl(knots);
        }
    }

    /// The DDC type of the sampling for the interpolation points.
    using interpolation_discrete_dimension_type = std::conditional_t<
            is_uniform_bsplines_v<BSplines>,
            ddc::UniformPointSampling<continuous_dimension_type>,
            ddc::NonUniformPointSampling<continuous_dimension_type>>;

    /**
     * Get the domain which can be used to access the interpolation points in the sampling.
     *
     * @return domain The discrete domain which maps to the sampling of interpolation points.
     */
    template <typename Sampling>
    static ddc::DiscreteDomain<Sampling> get_domain()
    {
        int const npoints = ddc::discrete_space<BSplines>().ncells() + !BSplines::is_periodic();
        return ddc::DiscreteDomain<Sampling>(
                ddc::DiscreteElement<Sampling>(0),
                ddc::DiscreteVector<Sampling>(npoints));
    }
};

} // namespace ddc
