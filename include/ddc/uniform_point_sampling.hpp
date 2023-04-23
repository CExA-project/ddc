// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "ddc/coordinate.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

struct UniformPointSamplingBase
{
};

/** UniformPointSampling models a uniform discretization of the provided continuous dimension
 */
template <class CDim>
class UniformPointSampling : UniformPointSamplingBase
{
public:
    using continuous_dimension_type = CDim;

    using continuous_element_type = Coordinate<CDim>;


    using discrete_dimension_type = UniformPointSampling;

public:
    template <class MemorySpace>
    class Impl
    {
        template <class OMemorySpace>
        friend class Impl;

    private:
        continuous_element_type m_origin {0.};

        double m_step {1.};

    public:
        using discrete_dimension_type = UniformPointSampling;

        Impl() = default;

        Impl(Impl const&) = delete;

        template <class OriginMemorySpace>
        explicit Impl(Impl<OriginMemorySpace> const& impl)
            : m_origin(impl.m_origin)
            , m_step(impl.m_step)
        {
        }

        Impl(Impl&&) = default;

        /** @brief Construct a `Impl` from a point and a spacing step.
         *
         * @param origin the real coordinate of mesh coordinate 0
         * @param step   the real distance between two points of mesh distance 1
         */
        constexpr Impl(continuous_element_type origin, double step) : m_origin(origin), m_step(step)
        {
            assert(step > 0);
        }

        ~Impl() = default;

        /// @brief Lower bound index of the mesh
        constexpr continuous_element_type origin() const noexcept
        {
            return m_origin;
        }

        /// @brief Lower bound index of the mesh
        constexpr DiscreteElementType front() const noexcept
        {
            return DiscreteElementType {0};
        }

        /// @brief Spacing step of the mesh
        constexpr double step() const
        {
            return m_step;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        constexpr continuous_element_type coordinate(
                DiscreteElementType const& icoord) const noexcept
        {
            return m_origin + continuous_element_type(icoord) * m_step;
        }
    };
};

template <class DDim>
using continuous_dimension_t = typename DDim::continuous_dimension_type;

/** Construct a Impl<Kokkos::HostSpace> and associated discrete_domain_type from a segment
 *  \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
 *
 * @param a coordinate of the first point of the domain
 * @param b coordinate of the last point of the domain
 * @param n number of points to map on the segment \f$[a, b]\f$ including a & b
 */
template <class DDim>
std::tuple<typename DDim::template Impl<Kokkos::HostSpace>, DiscreteDomain<DDim>>
uniform_point_sampling_init(
        Coordinate<continuous_dimension_t<DDim>> a,
        Coordinate<continuous_dimension_t<DDim>> b,
        DiscreteVector<DDim> n)
{
    assert(a < b);
    assert(n > 1);
    typename DDim::template Impl<Kokkos::HostSpace>
            disc(a, Coordinate<continuous_dimension_t<DDim>> {(b - a) / (n - 1)});
    DiscreteDomain<DDim> domain {DiscreteElement<DDim>(disc.front()), n};
    return std::make_tuple(std::move(disc), std::move(domain));
}

/** Construct a uniform `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
 *  number of points `n`.
 *
 * @param a coordinate of the first point of the domain
 * @param b coordinate of the last point of the domain
 * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
 * @param n_ghosts_before number of additional "ghost" points before the segment
 * @param n_ghosts_after number of additional "ghost" points after the segment
 */
template <class DDim>
std::tuple<
        typename DDim::template Impl<Kokkos::HostSpace>,
        DiscreteDomain<DDim>,
        DiscreteDomain<DDim>,
        DiscreteDomain<DDim>,
        DiscreteDomain<DDim>>
uniform_point_sampling_init_ghosted(
        Coordinate<continuous_dimension_t<DDim>> a,
        Coordinate<continuous_dimension_t<DDim>> b,
        DiscreteVector<DDim> n,
        DiscreteVector<DDim> n_ghosts_before,
        DiscreteVector<DDim> n_ghosts_after)
{
    assert(a < b);
    assert(n > 1);
    double discretization_step {(b - a) / (n - 1)};
    typename DDim::template Impl<Kokkos::HostSpace>
            disc(a - n_ghosts_before.value() * discretization_step, discretization_step);
    DiscreteDomain<DDim> ghosted_domain = DiscreteDomain<
            DDim>(DiscreteElement<DDim>(disc.front()), n + n_ghosts_before + n_ghosts_after);
    DiscreteDomain<DDim> pre_ghost
            = DiscreteDomain<DDim>(DiscreteElement<DDim>(ghosted_domain.front()), n_ghosts_before);
    DiscreteDomain<DDim> main_domain = DiscreteDomain<
            DDim>(DiscreteElement<DDim>(ghosted_domain.front() + n_ghosts_before), n);
    DiscreteDomain<DDim> post_ghost
            = DiscreteDomain<DDim>(DiscreteElement<DDim>(main_domain.back() + 1), n_ghosts_after);
    return std::make_tuple(
            std::move(disc),
            std::move(main_domain),
            std::move(ghosted_domain),
            std::move(pre_ghost),
            std::move(post_ghost));
}

/** Construct a uniform `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
 *  number of points `n`.
 *
 * @param a coordinate of the first point of the domain
 * @param b coordinate of the last point of the domain
 * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
 * @param n_ghosts number of additional "ghost" points before and after the segment
 */
template <class DDim>
std::tuple<
        typename DDim::template Impl<Kokkos::HostSpace>,
        DiscreteDomain<DDim>,
        DiscreteDomain<DDim>,
        DiscreteDomain<DDim>,
        DiscreteDomain<DDim>>
uniform_point_sampling_init_ghosted(
        Coordinate<continuous_dimension_t<DDim>> a,
        Coordinate<continuous_dimension_t<DDim>> b,
        DiscreteVector<DDim> n,
        DiscreteVector<DDim> n_ghosts)
{
    return uniform_point_sampling_init_ghosted(a, b, n, n_ghosts, n_ghosts);
}

template <class DDim>
constexpr bool is_uniform_sampling_v = std::is_base_of_v<UniformPointSamplingBase, DDim>;

template <
        class DDimImpl,
        std::enable_if_t<
                is_uniform_sampling_v<typename DDimImpl::discrete_dimension_type>,
                int> = 0>
std::ostream& operator<<(std::ostream& out, DDimImpl const& mesh)
{
    return out << "UniformPointSampling( origin=" << mesh.origin() << ", step=" << mesh.step()
               << " )";
}

/// @brief Lower bound index of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::
        enable_if_t<is_uniform_sampling_v<DDim>, Coordinate<continuous_dimension_t<DDim>>>
        origin() noexcept
{
    return discrete_space<DDim>().origin();
}

/// @brief Lower bound index of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<is_uniform_sampling_v<DDim>, DiscreteElement<DDim>>
front() noexcept
{
    return discrete_space<DDim>().front();
}

/// @brief Spacing step of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<is_uniform_sampling_v<DDim>, double> step() noexcept
{
    return discrete_space<DDim>().step();
}

template <class DDim, std::enable_if_t<is_uniform_sampling_v<DDim>, int> = 0>
DDC_INLINE_FUNCTION constexpr Coordinate<continuous_dimension_t<DDim>> coordinate(
        DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().coordinate(c.uid());
}

template <class DDim, std::enable_if_t<is_uniform_sampling_v<DDim>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<DDim>> distance_at_left(DiscreteElement<DDim>)
{
    return Coordinate<continuous_dimension_t<DDim>>(step<DDim>());
}

template <class DDim, std::enable_if_t<is_uniform_sampling_v<DDim>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<DDim>> distance_at_right(
        DiscreteElement<DDim>)
{
    return Coordinate<continuous_dimension_t<DDim>>(step<DDim>());
}

template <class DDim, std::enable_if_t<is_uniform_sampling_v<DDim>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<DDim>> rmin(DiscreteDomain<DDim> const& d)
{
    return coordinate(d.front());
}

template <class DDim, std::enable_if_t<is_uniform_sampling_v<DDim>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<DDim>> rmax(DiscreteDomain<DDim> const& d)
{
    return coordinate(d.back());
}

template <class DDim, std::enable_if_t<is_uniform_sampling_v<DDim>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<DDim>> rlength(DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

template <class T>
struct is_uniform_domain : std::false_type
{
};

template <class... DDims>
struct is_uniform_domain<DiscreteDomain<DDims...>>
    : std::conditional_t<(is_uniform_sampling_v<DDims> && ...), std::true_type, std::false_type>
{
};

template <class T>
constexpr bool is_uniform_domain_v = is_uniform_domain<T>::value;

} // namespace ddc
