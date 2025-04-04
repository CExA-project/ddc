// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include <ddc/ddc.hpp>

#include "bsplines_non_uniform.hpp"
#include "bsplines_uniform.hpp"
#include "spline_boundary_conditions.hpp"

namespace ddc {

/**
 * A class which provides helper functions to initialise the Greville points from a B-Spline definition.
 *
 * @tparam BSplines The bspline class relative to which the Greville points will be calculated.
 * @tparam BcLower The lower boundary condition that will be used to build the splines.
 * @tparam BcUpper The upper boundary condition that will be used to build the splines.
 */
template <class BSplines, ddc::BoundCond BcLower, ddc::BoundCond BcUpper>
class GrevilleInterpolationPoints
{
    using continuous_dimension_type = typename BSplines::continuous_dimension_type;

    template <class Sampling>
    struct IntermediateUniformSampling
        : UniformPointSampling<typename Sampling::continuous_dimension_type>
    {
    };

    template <class Sampling>
    struct IntermediateNonUniformSampling
        : NonUniformPointSampling<typename Sampling::continuous_dimension_type>
    {
    };

    template <class Sampling, typename U = BSplines, class = std::enable_if_t<U::is_uniform()>>
    static auto uniform_greville_points()
    {
        using SamplingImpl = typename Sampling::template Impl<Sampling, Kokkos::HostSpace>;

        double constexpr shift = (BSplines::degree() % 2 == 0) ? 0.5 : 0.0;
        double const dx
                = (ddc::discrete_space<BSplines>().rmax() - ddc::discrete_space<BSplines>().rmin())
                  / ddc::discrete_space<BSplines>().ncells();
        return SamplingImpl(
                ddc::Coordinate<continuous_dimension_type>(
                        ddc::discrete_space<BSplines>().rmin() + shift * dx),
                ddc::Coordinate<continuous_dimension_type>(dx));
    }

    template <class Sampling, typename U = BSplines, class = std::enable_if_t<!U::is_uniform()>>
    static auto non_uniform_greville_points()
    {
        using SamplingImpl = typename Sampling::template Impl<Sampling, Kokkos::HostSpace>;

        int n_greville_points = 0;
        if constexpr (U::is_periodic()) {
            n_greville_points = ddc::discrete_space<BSplines>().nbasis() + 1;
        } else {
            n_greville_points = ddc::discrete_space<BSplines>().nbasis();
        }

        std::vector<double> greville_points(n_greville_points);
        ddc::DiscreteDomain<BSplines> const bspline_domain
                = ddc::discrete_space<BSplines>().full_domain().take_first(
                        ddc::DiscreteVector<BSplines>(ddc::discrete_space<BSplines>().nbasis()));

        ddc::DiscreteVector<NonUniformBsplinesKnots<BSplines>> n_points_in_average(
                BSplines::degree());

        ddc::DiscreteElement<BSplines> ib0(bspline_domain.front());

        ddc::for_each(bspline_domain, [&](ddc::DiscreteElement<BSplines> ib) {
            // Define the Greville points from the bspline knots
            greville_points[ib - ib0] = 0.0;
            ddc::DiscreteDomain<NonUniformBsplinesKnots<BSplines>> const sub_domain(
                    ddc::discrete_space<BSplines>().get_first_support_knot(ib) + 1,
                    n_points_in_average);
            ddc::for_each(sub_domain, [&](auto ik) {
                greville_points[ib - ib0] += ddc::coordinate(ik);
            });
            greville_points[ib - ib0] /= n_points_in_average.value();
        });

        // Use periodicity to ensure all points are in the domain
        if constexpr (U::is_periodic()) {
            std::vector<double> temp_knots(BSplines::degree());
            int npoints(0);
            // Count the number of interpolation points that need shifting to preserve the ordering
            while (greville_points[npoints] < ddc::discrete_space<BSplines>().rmin()) {
                temp_knots[npoints]
                        = greville_points[npoints] + ddc::discrete_space<BSplines>().length();
                ++npoints;
            }
            // Shift the points
            for (std::size_t i = 0; i < ddc::discrete_space<BSplines>().nbasis() - npoints; ++i) {
                greville_points[i] = greville_points[i + npoints];
            }
            for (int i = 0; i < npoints; ++i) {
                greville_points[ddc::discrete_space<BSplines>().nbasis() - npoints + i]
                        = temp_knots[i];
            }

            // Save a periodic point to initialise the domain size
            greville_points[n_greville_points - 1]
                    = greville_points[0] + ddc::discrete_space<BSplines>().length();
        }

        return SamplingImpl(greville_points);
    }

    static constexpr int N_BE_MIN = n_boundary_equations(BcLower, BSplines::degree());
    static constexpr int N_BE_MAX = n_boundary_equations(BcUpper, BSplines::degree());
    template <class U>
    static constexpr bool is_uniform_discrete_dimension_v
            = U::is_uniform() && ((N_BE_MIN != 0 && N_BE_MAX != 0) || U::is_periodic());

public:
    /**
     * Get the UniformPointSampling defining the Greville points.
     *
     * This function is called when the result is a UniformPointSampling. This is the case
     * when uniform splines are used with an odd degree and with boundary conditions which
     * do not introduce additional interpolation points.
     *
     * @tparam Sampling The discrete dimension supporting the Greville points.
     *
     * @returns The mesh of uniform Greville points.
     */
    template <
            class Sampling,
            typename U = BSplines,
            std::enable_if_t<
                    is_uniform_discrete_dimension_v<U>,
                    bool>
            = true> // U must be in condition for SFINAE
    static auto get_sampling()
    {
        return uniform_greville_points<Sampling>();
    }

    /**
     * Get the NonUniformPointSampling defining the Greville points.
     *
     * @tparam Sampling The discrete dimension supporting the Greville points.
     *
     * @returns The mesh of non-uniform Greville points.
     */
    template <
            class Sampling,
            typename U = BSplines,
            std::enable_if_t<
                    !is_uniform_discrete_dimension_v<U>,
                    bool>
            = true> // U must be in condition for SFINAE
    static auto get_sampling()
    {
        using SamplingImpl = typename Sampling::template Impl<Sampling, Kokkos::HostSpace>;
        if constexpr (U::is_uniform()) {
            using IntermediateSampling = IntermediateUniformSampling<Sampling>;
            auto points_wo_bcs = uniform_greville_points<IntermediateSampling>();
            int const n_break_points = ddc::discrete_space<BSplines>().ncells() + 1;
            int const npoints = ddc::discrete_space<BSplines>().nbasis() - N_BE_MIN - N_BE_MAX;
            std::vector<double> points_with_bcs(npoints);

            // Construct Greville-like points at the edge
            if constexpr (BcLower == ddc::BoundCond::GREVILLE) {
                for (std::size_t i(0); i < BSplines::degree() / 2 + 1; ++i) {
                    points_with_bcs[i]
                            = (BSplines::degree() - i) * ddc::discrete_space<BSplines>().rmin();
                    ddc::DiscreteElement<BSplines> const spline_idx(i);
                    ddc::DiscreteVector<UniformBsplinesKnots<BSplines>> const n_knots_in_domain(i);
                    ddc::DiscreteDomain<UniformBsplinesKnots<BSplines>> const sub_domain(
                            ddc::discrete_space<BSplines>().get_last_support_knot(spline_idx)
                                    - n_knots_in_domain,
                            n_knots_in_domain);
                    ddc::for_each(
                            sub_domain,
                            [&](DiscreteElement<UniformBsplinesKnots<BSplines>> ik) {
                                points_with_bcs[i] += ddc::coordinate(ik);
                            });
                    points_with_bcs[i] /= BSplines::degree();
                }
            } else {
                points_with_bcs[0]
                        = points_wo_bcs.coordinate(ddc::DiscreteElement<IntermediateSampling>(0));
            }

            int const n_start
                    = (BcLower == ddc::BoundCond::GREVILLE) ? BSplines::degree() / 2 + 1 : 1;
            int const domain_size = n_break_points - 2;
            ddc::DiscreteElement<IntermediateSampling> domain_start(1);
            ddc::DiscreteDomain<IntermediateSampling> const
                    domain(domain_start, ddc::DiscreteVector<IntermediateSampling>(domain_size));

            // Copy central points
            ddc::for_each(domain, [&](auto ip) {
                points_with_bcs[ip - domain_start + n_start] = points_wo_bcs.coordinate(ip);
            });

            // Construct Greville-like points at the edge
            if constexpr (BcUpper == ddc::BoundCond::GREVILLE) {
                for (std::size_t i(0); i < BSplines::degree() / 2 + 1; ++i) {
                    points_with_bcs[npoints - 1 - i]
                            = (BSplines::degree() - i) * ddc::discrete_space<BSplines>().rmax();
                    ddc::DiscreteElement<BSplines> const spline_idx(
                            ddc::discrete_space<BSplines>().nbasis() - 1 - i);
                    ddc::DiscreteVector<UniformBsplinesKnots<BSplines>> const n_knots_in_domain(i);
                    ddc::DiscreteDomain<UniformBsplinesKnots<BSplines>> const sub_domain(
                            ddc::discrete_space<BSplines>().get_first_support_knot(spline_idx) + 1,
                            n_knots_in_domain);
                    ddc::for_each(
                            sub_domain,
                            [&](DiscreteElement<UniformBsplinesKnots<BSplines>> ik) {
                                points_with_bcs[npoints - 1 - i] += ddc::coordinate(ik);
                            });
                    points_with_bcs[npoints - 1 - i] /= BSplines::degree();
                }
            } else {
                points_with_bcs[npoints - 1] = points_wo_bcs.coordinate(
                        ddc::DiscreteElement<IntermediateSampling>(
                                ddc::discrete_space<BSplines>().ncells() - 1
                                + BSplines::degree() % 2));
            }
            return SamplingImpl(points_with_bcs);
        } else {
            using IntermediateSampling = IntermediateNonUniformSampling<Sampling>;
            if constexpr (N_BE_MIN == 0 && N_BE_MAX == 0) {
                return non_uniform_greville_points<Sampling>();
            } else {
                auto points_wo_bcs = non_uniform_greville_points<IntermediateSampling>();
                // All points are Greville points. Extract unnecessary points near the boundary
                std::vector<double> points_with_bcs(points_wo_bcs.size() - N_BE_MIN - N_BE_MAX);
                int constexpr n_start = N_BE_MIN;

                using length = ddc::DiscreteVector<IntermediateSampling>;

                ddc::DiscreteElement<IntermediateSampling> domain_start(n_start);
                ddc::DiscreteDomain<IntermediateSampling> const
                        domain(domain_start, length(points_with_bcs.size()));

                points_with_bcs[0] = points_wo_bcs.coordinate(domain.front());
                ddc::for_each(domain.remove(length(1), length(1)), [&](auto ip) {
                    points_with_bcs[ip - domain_start] = points_wo_bcs.coordinate(ip);
                });
                points_with_bcs[points_with_bcs.size() - 1]
                        = points_wo_bcs.coordinate(domain.back());

                return SamplingImpl(points_with_bcs);
            }
        }
    }

    /**
     * The type of the mesh.
     *
     * This is either NonUniformPointSampling or UniformPointSampling.
     */
    using interpolation_discrete_dimension_type = std::conditional_t<
            is_uniform_discrete_dimension_v<BSplines>,
            ddc::UniformPointSampling<continuous_dimension_type>,
            ddc::NonUniformPointSampling<continuous_dimension_type>>;

    /**
     * Get the domain which gives us access to all of the Greville points.
     *
     * @tparam Sampling The discrete dimension supporting the Greville points.
     *
     * @returns The domain of the Greville points.
     */
    template <class Sampling>
    static ddc::DiscreteDomain<Sampling> get_domain()
    {
        int const npoints = ddc::discrete_space<BSplines>().nbasis() - N_BE_MIN - N_BE_MAX;
        return ddc::DiscreteDomain<Sampling>(
                ddc::DiscreteElement<Sampling>(0),
                ddc::DiscreteVector<Sampling>(npoints));
    }
};

} // namespace ddc
