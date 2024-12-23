// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ddc/coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

namespace detail {

struct NonUniformPointSamplingBase
{
};

} // namespace detail

/// `NonUniformPointSampling` models a non-uniform discretization of the `CDim` segment \f$[a, b]\f$.
template <class CDim>
class NonUniformPointSampling : detail::NonUniformPointSamplingBase
{
public:
    using continuous_dimension_type = CDim;

    using continuous_element_type = Coordinate<CDim>;


    using discrete_dimension_type = NonUniformPointSampling;

public:
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

        Kokkos::View<continuous_element_type*, MemorySpace> m_points;

    public:
        using discrete_dimension_type = NonUniformPointSampling;

        using discrete_domain_type = DiscreteDomain<DDim>;

        using discrete_element_type = DiscreteElement<DDim>;

        using discrete_vector_type = DiscreteVector<DDim>;

        Impl() = default;

        /// @brief Construct a `NonUniformPointSampling` using a brace-list, i.e. `NonUniformPointSampling mesh({0., 1.})`
        Impl(std::initializer_list<continuous_element_type> const points)
            : Impl(points.begin(), points.end())
        {
        }

        /// @brief Construct a `NonUniformPointSampling` using a C++20 "common range".
        template <class InputRange>
        explicit Impl(InputRange const& points) : Impl(points.begin(), points.end())
        {
        }

        /// @brief Construct a `NonUniformPointSampling` using a pair of iterators.
        template <class InputIt>
        Impl(InputIt const points_begin, InputIt const points_end)
        {
            using view_type = Kokkos::View<continuous_element_type*, MemorySpace>;
            if (!std::is_sorted(points_begin, points_end)) {
                throw std::runtime_error("Input points must be sorted");
            }
            // Make a contiguous copy of [points_begin, points_end[
            std::vector<continuous_element_type> host_points(points_begin, points_end);
            m_points = view_type("NonUniformPointSampling::points", host_points.size());
            Kokkos::deep_copy(m_points, view_type(host_points.data(), host_points.size()));
        }

        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl)
            : m_points(Kokkos::create_mirror_view_and_copy(MemorySpace(), impl.m_points))
        {
        }

        Impl(Impl const& x) = delete;

        Impl(Impl&& x) = default;

        ~Impl() = default;

        Impl& operator=(Impl const& x) = delete;

        Impl& operator=(Impl&& x) = default;

        KOKKOS_FUNCTION std::size_t size() const
        {
            return m_points.size();
        }

        /// @brief Lower bound index of the mesh
        KOKKOS_FUNCTION discrete_element_type front() const noexcept
        {
            return discrete_element_type(0);
        }

        /// @brief Convert a mesh index into a position in `CDim`
        KOKKOS_FUNCTION continuous_element_type
        coordinate(discrete_element_type const& icoord) const noexcept
        {
            return m_points(icoord.uid());
        }
    };

    /** Construct an Impl<Kokkos::HostSpace> and associated discrete_domain_type from a range
     * containing the points coordinates along the `DDim` dimension.
     *
     * @param non_uniform_points a range (std::vector, std::array, ...) containing the coordinates of the points of the domain.
     */
    template <class DDim, class InputRange>
    static std::tuple<typename DDim::template Impl<DDim, Kokkos::HostSpace>, DiscreteDomain<DDim>>
    init(InputRange const& non_uniform_points)
    {
        assert(!non_uniform_points.empty());
        DiscreteVector<DDim> const n(non_uniform_points.size());
        typename DDim::template Impl<DDim, Kokkos::HostSpace> disc(non_uniform_points);
        DiscreteDomain<DDim> domain(disc.front(), n);
        return std::make_tuple(std::move(disc), std::move(domain));
    }

    /** Construct 4 non-uniform `DiscreteDomain` and an Impl<Kokkos::HostSpace> from 3 ranges containing the points coordinates along the `DDim` dimension.
     *
     * @param domain_r a range containing the coordinates of the points of the main domain along the DDim position
     * @param pre_ghost_r a range containing the positions of the ghost points before the main domain the DDim position
     * @param post_ghost_r a range containing the positions of the ghost points after the main domain the DDim position
     */
    template <class DDim, class InputRange>
    static std::tuple<
            typename DDim::template Impl<DDim, Kokkos::HostSpace>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>>
    init_ghosted(
            InputRange const& domain_r,
            InputRange const& pre_ghost_r,
            InputRange const& post_ghost_r)
    {
        assert(!domain_r.empty());

        DiscreteVector<DDim> const n(domain_r.size());
        DiscreteVector<DDim> const n_ghosts_before(pre_ghost_r.size());
        DiscreteVector<DDim> const n_ghosts_after(post_ghost_r.size());

        std::vector<typename InputRange::value_type> full_domain;

        std::copy(pre_ghost_r.begin(), pre_ghost_r.end(), std::back_inserter(full_domain));
        std::copy(domain_r.begin(), domain_r.end(), std::back_inserter(full_domain));
        std::copy(post_ghost_r.begin(), post_ghost_r.end(), std::back_inserter(full_domain));

        typename DDim::template Impl<DDim, Kokkos::HostSpace> disc(full_domain);

        DiscreteDomain<DDim> ghosted_domain(disc.front(), n + n_ghosts_before + n_ghosts_after);
        DiscreteDomain<DDim> pre_ghost = ghosted_domain.take_first(n_ghosts_before);
        DiscreteDomain<DDim> main_domain = ghosted_domain.remove(n_ghosts_before, n_ghosts_after);
        DiscreteDomain<DDim> post_ghost = ghosted_domain.take_last(n_ghosts_after);
        return std::make_tuple(
                std::move(disc),
                std::move(main_domain),
                std::move(ghosted_domain),
                std::move(pre_ghost),
                std::move(post_ghost));
    }
};

template <class DDim>
struct is_non_uniform_point_sampling
    : public std::is_base_of<detail::NonUniformPointSamplingBase, DDim>::type
{
};

template <class DDim>
constexpr bool is_non_uniform_point_sampling_v = is_non_uniform_point_sampling<DDim>::value;

template <
        class DDimImpl,
        std::enable_if_t<
                is_non_uniform_point_sampling_v<typename DDimImpl::discrete_dimension_type>,
                int>
        = 0>
std::ostream& operator<<(std::ostream& out, DDimImpl const& mesh)
{
    return out << "NonUniformPointSampling(" << mesh.size() << ")";
}

template <class DDim, std::enable_if_t<is_non_uniform_point_sampling_v<DDim>, int> = 0>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> coordinate(
        DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().coordinate(c);
}

template <class DDim, std::enable_if_t<is_non_uniform_point_sampling_v<DDim>, int> = 0>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> distance_at_left(
        DiscreteElement<DDim> i)
{
    return coordinate(i) - coordinate(i - 1);
}

template <class DDim, std::enable_if_t<is_non_uniform_point_sampling_v<DDim>, int> = 0>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> distance_at_right(
        DiscreteElement<DDim> i)
{
    return coordinate(i + 1) - coordinate(i);
}

template <class DDim, std::enable_if_t<is_non_uniform_point_sampling_v<DDim>, int> = 0>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmin(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.front());
}

template <class DDim, std::enable_if_t<is_non_uniform_point_sampling_v<DDim>, int> = 0>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmax(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.back());
}

template <class DDim, std::enable_if_t<is_non_uniform_point_sampling_v<DDim>, int> = 0>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rlength(
        DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

} // namespace ddc
