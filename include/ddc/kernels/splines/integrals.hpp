// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ddc/ddc.hpp>

#include "bsplines_non_uniform.hpp"
#include "bsplines_uniform.hpp"

namespace ddc {

namespace detail {

/** @brief Compute the integrals of the uniform B-splines.
 *
 * The integral of each of the B-splines over their support within the domain on which this basis was defined.
 *
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] int_vals The values of the integrals. It has to be a 1D Chunkspan of size (nbasis).
 */
template <class ExecSpace, class DDim, class Layout, class MemorySpace>
void uniform_bsplines_integrals(
        ExecSpace const& execution_space,
        ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim>, Layout, MemorySpace> int_vals)
{
    static_assert(is_uniform_bsplines_v<DDim>);
    static_assert(
            Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible,
            "MemorySpace has to be accessible for ExecutionSpace.");

    assert([&]() -> bool {
        if constexpr (DDim::is_periodic()) {
            return int_vals.size() == ddc::discrete_space<DDim>().nbasis()
                   || int_vals.size() == ddc::discrete_space<DDim>().size();
        } else {
            return int_vals.size() == ddc::discrete_space<DDim>().nbasis();
        }
    }());

    ddc::DiscreteDomain<DDim> const full_dom_splines(ddc::discrete_space<DDim>().full_domain());

    if constexpr (DDim::is_periodic()) {
        ddc::DiscreteDomain<DDim> const dom_bsplines(full_dom_splines.take_first(
                ddc::DiscreteVector<DDim> {ddc::discrete_space<DDim>().nbasis()}));
        ddc::parallel_fill(
                execution_space,
                int_vals[dom_bsplines],
                ddc::step<UniformBsplinesKnots<DDim>>());
        if (int_vals.size() == ddc::discrete_space<DDim>().size()) {
            ddc::DiscreteDomain<DDim> const dom_bsplines_repeated(
                    full_dom_splines.take_last(ddc::DiscreteVector<DDim> {DDim::degree()}));
            ddc::parallel_fill(execution_space, int_vals[dom_bsplines_repeated], 0);
        }
    } else {
        ddc::DiscreteDomain<DDim> const dom_bspline_entirely_in_domain
                = full_dom_splines
                          .remove(ddc::DiscreteVector<DDim>(DDim::degree()),
                                  ddc::DiscreteVector<DDim>(DDim::degree()));
        ddc::parallel_fill(
                execution_space,
                int_vals[dom_bspline_entirely_in_domain],
                ddc::step<UniformBsplinesKnots<DDim>>());

        ddc::DiscreteElement<DDim> const first_bspline = full_dom_splines.front();
        ddc::DiscreteElement<DDim> const last_bspline = full_dom_splines.back();

        Kokkos::parallel_for(
                Kokkos::RangePolicy<
                        ExecSpace,
                        Kokkos::IndexType<std::size_t>>(execution_space, 0, DDim::degree()),
                KOKKOS_LAMBDA(std::size_t i) {
                    std::array<double, DDim::degree() + 2> edge_vals_ptr;
                    std::experimental::mdspan<
                            double,
                            std::experimental::extents<std::size_t, DDim::degree() + 2>> const
                            edge_vals(edge_vals_ptr.data());

                    ddc::discrete_space<DDim>().eval_basis(
                            edge_vals,
                            ddc::discrete_space<DDim>().rmin(),
                            DDim::degree() + 1);

                    double const d_eval = ddc::detail::sum(edge_vals);

                    double const c_eval = ddc::detail::sum(edge_vals, 0, DDim::degree() - i);

                    double const edge_value
                            = ddc::step<UniformBsplinesKnots<DDim>>() * (d_eval - c_eval);

                    int_vals(first_bspline + i) = edge_value;
                    int_vals(last_bspline - i) = edge_value;
                });
    }
}

/** @brief Compute the integrals of the non uniform B-splines.
 *
 * The integral of each of the B-splines over their support within the domain on which this basis was defined.
 *
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] int_vals The values of the integrals. It has to be a 1D Chunkspan of size (nbasis).
 */
template <class ExecSpace, class DDim, class Layout, class MemorySpace>
void non_uniform_bsplines_integrals(
        ExecSpace const& execution_space,
        ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim>, Layout, MemorySpace> int_vals)
{
    static_assert(is_non_uniform_bsplines_v<DDim>);
    static_assert(
            Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible,
            "MemorySpace has to be accessible for ExecutionSpace.");

    assert([&]() -> bool {
        if constexpr (DDim::is_periodic()) {
            return int_vals.size() == ddc::discrete_space<DDim>().nbasis()
                   || int_vals.size() == ddc::discrete_space<DDim>().size();
        } else {
            return int_vals.size() == ddc::discrete_space<DDim>().nbasis();
        }
    }());

    ddc::DiscreteDomain<DDim> const full_dom_splines(ddc::discrete_space<DDim>().full_domain());

    double const inv_deg = 1.0 / (DDim::degree() + 1);

    ddc::DiscreteDomain<DDim> const dom_bsplines(full_dom_splines.take_first(
            ddc::DiscreteVector<DDim> {ddc::discrete_space<DDim>().nbasis()}));
    ddc::parallel_for_each(
            execution_space,
            dom_bsplines,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDim> ix) {
                int_vals(ix)
                        = (ddc::coordinate(ddc::discrete_space<DDim>().get_last_support_knot(ix))
                           - ddc::coordinate(
                                   ddc::discrete_space<DDim>().get_first_support_knot(ix)))
                          * inv_deg;
            });

    if constexpr (DDim::is_periodic()) {
        if (int_vals.size() == ddc::discrete_space<DDim>().size()) {
            ddc::DiscreteDomain<DDim> const dom_bsplines_wrap(
                    full_dom_splines.take_last(ddc::DiscreteVector<DDim> {DDim::degree()}));
            ddc::parallel_fill(execution_space, int_vals[dom_bsplines_wrap], 0);
        }
    }
}

} // namespace detail

/** @brief Compute the integrals of the B-splines.
 *
 * The integral of each of the B-splines over their support within the domain on which this basis was defined.
 *
 * @param[in] execution_space a Kokkos execution space where the loop will be executed on
 * @param[out] int_vals The values of the integrals. It has to be a 1D Chunkspan of size (nbasis).
 * @return The values of the integrals.
 */
template <class ExecSpace, class DDim, class Layout, class MemorySpace>
ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim>, Layout, MemorySpace> integrals(
        ExecSpace const& execution_space,
        ddc::ChunkSpan<double, ddc::DiscreteDomain<DDim>, Layout, MemorySpace> int_vals)
{
    static_assert(is_uniform_bsplines_v<DDim> || is_non_uniform_bsplines_v<DDim>);
    if constexpr (is_uniform_bsplines_v<DDim>) {
        uniform_bsplines_integrals(execution_space, int_vals);
    } else if constexpr (is_non_uniform_bsplines_v<DDim>) {
        non_uniform_bsplines_integrals(execution_space, int_vals);
    }
    return int_vals;
}

} // namespace ddc
