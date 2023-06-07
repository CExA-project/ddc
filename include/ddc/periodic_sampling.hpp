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

/** PeriodicSampling models a periodic discretization of the provided continuous dimension
 */
template <class CDim>
class PeriodicSampling
{
public:
    using continuous_dimension_type = CDim;

    using continuous_element_type = Coordinate<CDim>;


    using discrete_dimension_type = PeriodicSampling;

    using discrete_element_type = DiscreteElement<PeriodicSampling>;

    using discrete_domain_type = DiscreteDomain<PeriodicSampling>;

    using discrete_vector_type = DiscreteVector<PeriodicSampling>;

public:
    template <class MemorySpace>
    class Impl
    {
        template <class OMemorySpace>
        friend class Impl;

    private:
        continuous_element_type m_origin {0.};

        double m_step {1.};

        std::size_t m_n_period {2};

    public:
        using discrete_dimension_type = PeriodicSampling;

        Impl() = default;

        Impl(Impl const&) = delete;

        template <class OriginMemorySpace>
        explicit Impl(Impl<OriginMemorySpace> const& impl)
            : m_origin(impl.m_origin)
            , m_step(impl.m_step)
            , m_n_period(impl.m_n_period)
        {
        }

        Impl(Impl&&) = default;

        /** @brief Construct a `Impl` from a point and a spacing step.
         *
         * @param origin the real coordinate of mesh coordinate 0
         * @param step   the real distance between two points of mesh distance 1
         * @param n_period   the number of steps in a period
         */
        constexpr Impl(continuous_element_type origin, double step, std::size_t n_period)
            : m_origin(origin)
            , m_step(step)
            , m_n_period(n_period)
        {
            assert(step > 0);
            assert(n_period > 0);
        }

        ~Impl() = default;

        /// @brief Lower bound index of the mesh
        constexpr continuous_element_type origin() const noexcept
        {
            return m_origin;
        }

        /// @brief Lower bound index of the mesh
        constexpr discrete_element_type front() const noexcept
        {
            return discrete_element_type {0};
        }

        /// @brief Spacing step of the mesh
        constexpr double step() const
        {
            return m_step;
        }

        /// @brief Number of steps in a period
        constexpr std::size_t n_period() const
        {
            return m_n_period;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        constexpr continuous_element_type coordinate(
                discrete_element_type const& icoord) const noexcept
        {
            return m_origin
                   + continuous_element_type(
                             static_cast<int>((icoord.uid() + m_n_period / 2 - 1) % m_n_period)
                             - static_cast<int>(m_n_period / 2 - 1))
                             * m_step;
        }
    };

    /** Construct a Impl<Kokkos::HostSpace> and associated discrete_domain_type from a segment
     *  \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n number of points to map on the segment \f$[a, b]\f$ including a & b
     * @param n_period   the number of steps in a period
     */
    static std::tuple<Impl<Kokkos::HostSpace>, discrete_domain_type> init(
            continuous_element_type a,
            continuous_element_type b,
            discrete_vector_type n,
            discrete_vector_type n_period)
    {
        assert(a < b);
        assert(n > 1);
        assert(n_period > 1);
        Impl<Kokkos::HostSpace> disc(a, continuous_element_type {(b - a) / (n - 1)}, n_period);
        discrete_domain_type domain {disc.front(), n};
        return std::make_tuple(std::move(disc), std::move(domain));
    }

    /** Construct a periodic `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
     *  number of points `n`.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * @param n_period   the number of steps in a period
     * @param n_ghosts_before number of additional "ghost" points before the segment
     * @param n_ghosts_after number of additional "ghost" points after the segment
     */
    static std::tuple<
            Impl<Kokkos::HostSpace>,
            discrete_domain_type,
            discrete_domain_type,
            discrete_domain_type,
            discrete_domain_type>
    init_ghosted(
            continuous_element_type a,
            continuous_element_type b,
            discrete_vector_type n,
            discrete_vector_type n_period,
            discrete_vector_type n_ghosts_before,
            discrete_vector_type n_ghosts_after)
    {
        using discrete_domain_type = discrete_domain_type;
        assert(a < b);
        assert(n > 1);
        assert(n_period > 1);
        double discretization_step {(b - a) / (n - 1)};
        Impl<Kokkos::HostSpace>
                disc(a - n_ghosts_before.value() * discretization_step,
                     discretization_step,
                     n_period);
        discrete_domain_type ghosted_domain
                = discrete_domain_type(disc.front(), n + n_ghosts_before + n_ghosts_after);
        discrete_domain_type pre_ghost
                = discrete_domain_type(ghosted_domain.front(), n_ghosts_before);
        discrete_domain_type main_domain
                = discrete_domain_type(ghosted_domain.front() + n_ghosts_before, n);
        discrete_domain_type post_ghost
                = discrete_domain_type(main_domain.back() + 1, n_ghosts_after);
        return std::make_tuple(
                std::move(disc),
                std::move(main_domain),
                std::move(ghosted_domain),
                std::move(pre_ghost),
                std::move(post_ghost));
    }

    /** Construct a periodic `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
     *  number of points `n`.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * @param n_period   the number of steps in a period
     * @param n_ghosts number of additional "ghost" points before and after the segment
     */
    static std::tuple<
            Impl<Kokkos::HostSpace>,
            discrete_domain_type,
            discrete_domain_type,
            discrete_domain_type,
            discrete_domain_type>
    init_ghosted(
            continuous_element_type a,
            continuous_element_type b,
            discrete_vector_type n,
            discrete_vector_type n_period,
            discrete_vector_type n_ghosts)
    {
        return init_ghosted(a, b, n, n_period, n_ghosts, n_ghosts);
    }
};

template <class>
struct is_periodic_sampling : public std::false_type
{
};

template <class CDim>
struct is_periodic_sampling<PeriodicSampling<CDim>> : public std::true_type
{
};

template <class DDim>
constexpr bool is_periodic_sampling_v = is_periodic_sampling<DDim>::value;


template <
        class DDimImpl,
        std::enable_if_t<
                is_periodic_sampling_v<typename DDimImpl::discrete_dimension_type>,
                int> = 0>
std::ostream& operator<<(std::ostream& out, DDimImpl const& mesh)
{
    return out << "PeriodicSampling( origin=" << mesh.origin() << ", step=" << mesh.step() << " )";
}

/// @brief Lower bound index of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::
        enable_if_t<is_periodic_sampling_v<DDim>, typename DDim::continuous_element_type>
        origin() noexcept
{
    return discrete_space<DDim>().origin();
}

/// @brief Lower bound index of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::
        enable_if_t<is_periodic_sampling_v<DDim>, typename DDim::discrete_element_type>
        front() noexcept
{
    return discrete_space<DDim>().front();
}

/// @brief Spacing step of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<is_periodic_sampling_v<DDim>, double> step() noexcept
{
    return discrete_space<DDim>().step();
}

template <class CDim>
DDC_INLINE_FUNCTION constexpr Coordinate<CDim> coordinate(
        DiscreteElement<PeriodicSampling<CDim>> const& c)
{
    return discrete_space<PeriodicSampling<CDim>>().coordinate(c);
}

template <class CDim>
DDC_INLINE_FUNCTION Coordinate<CDim> distance_at_left(DiscreteElement<PeriodicSampling<CDim>>)
{
    return Coordinate<CDim>(step<PeriodicSampling<CDim>>());
}

template <class CDim>
DDC_INLINE_FUNCTION Coordinate<CDim> distance_at_right(DiscreteElement<PeriodicSampling<CDim>>)
{
    return Coordinate<CDim>(step<PeriodicSampling<CDim>>());
}

template <class CDim>
DDC_INLINE_FUNCTION Coordinate<CDim> rmin(DiscreteDomain<PeriodicSampling<CDim>> const& d)
{
    return coordinate(d.front());
}

template <class CDim>
DDC_INLINE_FUNCTION Coordinate<CDim> rmax(DiscreteDomain<PeriodicSampling<CDim>> const& d)
{
    return coordinate(d.back());
}

template <class CDim>
DDC_INLINE_FUNCTION Coordinate<CDim> rlength(DiscreteDomain<PeriodicSampling<CDim>> const& d)
{
    return rmax(d) - rmin(d);
}

template <class T>
struct is_periodic_domain : std::false_type
{
};

template <class... DDims>
struct is_periodic_domain<DiscreteDomain<DDims...>>
    : std::conditional_t<(is_periodic_sampling_v<DDims> && ...), std::true_type, std::false_type>
{
};

template <class T>
constexpr bool is_periodic_domain_v = is_periodic_domain<T>::value;

} // namespace ddc
