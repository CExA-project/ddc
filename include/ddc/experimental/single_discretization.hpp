// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <ostream>
#include <type_traits>
#include <utility>

#include <Kokkos_Macros.hpp>

#include "ddc/coordinate.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"

namespace ddc::experimental {

struct SingleDiscretizationBase
{
};

/** Experimental concept representing a discretization with a single point.
 *
 * The main purpose is when taking a subdomain in a multidimensional discrete space in order to
 * loose track of the real position.
 */
template <class CDim>
class SingleDiscretization : SingleDiscretizationBase
{
public:
    using continuous_dimension_type = CDim;

#if defined(DDC_BUILD_DEPRECATED_CODE)
    using continuous_element_type
            [[deprecated("Use ddc::Coordinate<continuous_dimension_type> instead.")]]
            = Coordinate<CDim>;
#endif

    using discrete_dimension_type = SingleDiscretization;

public:
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    private:
        /// origin
        Coordinate<CDim> m_point;

    public:
        using discrete_dimension_type = SingleDiscretization;

        using discrete_domain_type = DiscreteDomain<DDim>;

        using discrete_element_type = DiscreteElement<DDim>;

        using discrete_vector_type = DiscreteVector<DDim>;

        explicit Impl(Coordinate<CDim> origin) noexcept : m_point(std::move(origin)) {}

        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl) : m_point(impl.m_point)
        {
        }

        Impl(Impl const&) = delete;

        Impl(Impl&&) = default;

        ~Impl() = default;

        Impl& operator=(Impl const& x) = delete;

        Impl& operator=(Impl&& x) = default;

        KOKKOS_FUNCTION Coordinate<CDim> origin() const noexcept
        {
            return m_point;
        }

        KOKKOS_FUNCTION Coordinate<CDim> coordinate(
                [[maybe_unused]] discrete_element_type icoord) const noexcept
        {
            assert(icoord == discrete_element_type(0));
            return m_point;
        }
    };
};

template <class DDim>
struct is_single_discretization : public std::is_base_of<SingleDiscretizationBase, DDim>::type
{
};

template <class DDim>
constexpr bool is_single_discretization_v = is_single_discretization<DDim>::value;

template <class Tag>
std::ostream& operator<<(std::ostream& out, SingleDiscretization<Tag> const& dom)
{
    return out << "SingleDiscretization( at=" << dom.origin() << " )";
}

/// @brief coordinate of the mesh
template <class DDim>
KOKKOS_FUNCTION std::enable_if_t<
        is_single_discretization_v<DDim>,
        Coordinate<typename DDim::continuous_dimension_type>>
origin() noexcept
{
    return discrete_space<DDim>().origin();
}

template <class DDim, std::enable_if_t<is_single_discretization_v<DDim>, int> = 0>
KOKKOS_FUNCTION ddc::Coordinate<typename DDim::continuous_dimension_type> coordinate(
        DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().coordinate(c);
}

template <class DDim, std::enable_if_t<is_single_discretization_v<DDim>, int> = 0>
KOKKOS_FUNCTION ddc::Coordinate<typename DDim::continuous_dimension_type> rmin(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.front());
}

template <class DDim, std::enable_if_t<is_single_discretization_v<DDim>, int> = 0>
KOKKOS_FUNCTION ddc::Coordinate<typename DDim::continuous_dimension_type> rmax(
        DiscreteDomain<DDim> const& d)
{
    return coordinate(d.back());
}

template <class DDim, std::enable_if_t<is_single_discretization_v<DDim>, int> = 0>
KOKKOS_FUNCTION ddc::Coordinate<typename DDim::continuous_dimension_type> rlength(
        DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

} // namespace ddc::experimental
