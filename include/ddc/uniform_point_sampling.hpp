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

/** UniformPointSampling models a uniform discretization of the provided continuous dimension
 */
template <class CDim>
class UniformPointSampling
{
public:
    using continuous_dimension_type = CDim;

    using continuous_element_type = CoordinateElement;


    using discretization_type = UniformPointSampling;

    using discrete_element_type = DiscreteElementType;

    using discrete_vector_type = DiscreteVectorElement;

public:
    static constexpr std::size_t rank()
    {
        return 1;
    }

    template <class MemorySpace>
    class Impl
    {
        template <class OMemorySpace>
        friend class Impl;

    private:
        continuous_element_type m_origin {0.};

        continuous_element_type m_step {1.};

    public:
        using discrete_dimension_type = UniformPointSampling;

        Impl() = default;

        Impl(Impl const&) = delete;

        template <class OriginMemorySpace>
        explicit Impl(Impl<OriginMemorySpace> const& impl)
        {
            m_origin = impl.m_origin;
            m_step = impl.m_step;
        }

        Impl(Impl&&) = default;

        /** @brief Construct a `Impl` from a point and a spacing step.
     *
     * @param origin the real coordinate of mesh coordinate 0
     * @param step   the real distance between two points of mesh distance 1
     */
        constexpr Impl(continuous_element_type origin, continuous_element_type step)
            : m_origin(origin)
            , m_step(step)
        {
            assert(step > 0);
        }

        /** @brief Construct a `Impl` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
     *
     * @param a the coordinate of a first real point (will have mesh coordinate 0)
     * @param b the coordinate of the second real point (will have mesh coordinate `n-1`)
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * 
     * @deprecated use the version accepting a vector for n instead
     */
        [[deprecated(
                "Use the version accepting a vector for n "
                "instead.")]] constexpr Impl(continuous_element_type a, continuous_element_type b, std::size_t n)
            : m_origin(a)
            , m_step((b - a) / (n - 1))
        {
            assert(a < b);
            assert(n > 1);
        }

        /** @brief Construct a `Impl` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
     * 
     * @param a the coordinate of a first real point (will have mesh coordinate 0)
     * @param b the coordinate of the second real point (will have mesh coordinate `n-1`)
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     */
        constexpr Impl(continuous_element_type a, continuous_element_type b, discrete_vector_type n)
            : m_origin(a)
            , m_step((b - a) / (n - 1))
        {
            assert(a < b);
            assert(n > 1);
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
        constexpr continuous_element_type step() const
        {
            return m_step;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        constexpr continuous_element_type coordinate(
                discrete_element_type const& icoord) const noexcept
        {
            return m_origin + continuous_element_type(icoord) * m_step;
        }
    };
};

template <class>
struct is_uniform_sampling : public std::false_type
{
};

template <class CDim>
struct is_uniform_sampling<UniformPointSampling<CDim>> : public std::true_type
{
};

template <class DDim>
constexpr bool is_uniform_sampling_v = is_uniform_sampling<DDim>::value;

template <class DDim, bool = is_discrete_dimension_v<DDim>>
struct is_uniform_sampling_dimension {
    static constexpr bool value = false;
};

template <class DDim>
struct is_uniform_sampling_dimension<DDim, true> {
    static constexpr bool value = is_uniform_sampling_v<typename DDim::discretization_type>;
};

template <class DDim>
constexpr bool is_uniform_sampling_dimension_v = is_uniform_sampling_dimension<DDim>::value;

/** Construct a Impl<Kokkos::HostSpace> and associated discrete_domain_type from a segment
 *  \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
 *
 * @param a coordinate of the first point of the domain
 * @param b coordinate of the last point of the domain
 * @param n number of points to map on the segment \f$[a, b]\f$ including a & b
 */
template <class DDim>
std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        std::tuple<
                typename DDim::discretization_type::template Impl<Kokkos::HostSpace>,
                DiscreteDomain<DDim>>>
init(typename DDim::discretization_type::continuous_element_type a,
     typename DDim::discretization_type::continuous_element_type b,
     DiscreteVector<DDim> n)
{
    using discretization_type = typename DDim::discretization_type;
    using discretization_impl_type = typename discretization_type::template Impl<Kokkos::HostSpace>;
    using continuous_element_type = typename discretization_type::continuous_element_type;
    using discrete_domain_type = DiscreteDomain<DDim>;
    assert(a < b);
    assert(n > 1);
    discretization_impl_type disc(a, continuous_element_type {(b - a) / (n - 1)});
    discrete_domain_type domain {DiscreteElement<DDim>(disc.front()), n};
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
std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        std::tuple<
                typename DDim::discretization_type::template Impl<Kokkos::HostSpace>,
                typename DDim::discrete_domain_type,
                typename DDim::discrete_domain_type,
                typename DDim::discrete_domain_type,
                typename DDim::discrete_domain_type>>
init_ghosted(
        typename DDim::discretization_type::continuous_element_type a,
        typename DDim::discretization_type::continuous_element_type b,
        DiscreteVector<DDim> n,
        DiscreteVector<DDim> n_ghosts_before,
        DiscreteVector<DDim> n_ghosts_after)
{
    using discretization_type = typename DDim::discretization_type;
    using discretization_impl_type = typename discretization_type::template Impl<Kokkos::HostSpace>;
    using continuous_element_type = typename discretization_type::continuous_element_type;
    using discrete_domain_type = DiscreteDomain<DDim>;
    assert(a < b);
    assert(n > 1);
    continuous_element_type discretization_step {(b - a) / (n - 1)};
    discretization_impl_type
            disc(a - n_ghosts_before.value() * discretization_step, discretization_step);
    discrete_domain_type ghosted_domain = discrete_domain_type(
            DiscreteElement<DDim>(disc.front()),
            n + n_ghosts_before + n_ghosts_after);
    discrete_domain_type pre_ghost = discrete_domain_type(ghosted_domain.front(), n_ghosts_before);
    discrete_domain_type main_domain
            = discrete_domain_type(ghosted_domain.front() + n_ghosts_before, n);
    discrete_domain_type post_ghost = discrete_domain_type(main_domain.back() + 1, n_ghosts_after);
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
std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        std::tuple<
                typename DDim::discretization_type::template Impl<Kokkos::HostSpace>,
                typename DDim::discrete_domain_type,
                typename DDim::discrete_domain_type,
                typename DDim::discrete_domain_type,
                typename DDim::discrete_domain_type>>
init_ghosted(
        typename DDim::discretization_type::continuous_element_type a,
        typename DDim::discretization_type::continuous_element_type b,
        DiscreteVector<DDim> n,
        DiscreteVector<DDim> n_ghosts)
{
    return init_ghosted<DDim>(a, b, n, n_ghosts, n_ghosts);
}


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
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
origin() noexcept
{
    return discrete_space<DDim>().origin();
}

/// @brief Lower bound index of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<is_uniform_sampling_dimension_v<DDim>, DiscreteElement<DDim>>
front() noexcept
{
    return discrete_space<DDim>().front();
}

/// @brief Spacing step of the mesh
template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
step() noexcept
{
    return discrete_space<DDim>().step();
}

template <class DDim>
DDC_INLINE_FUNCTION constexpr std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
coordinate(DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().coordinate(c.uid());
}

template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
distance_at_left(DiscreteElement<DDim> i)
{
    return step<DDim>();
}

template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
distance_at_right(DiscreteElement<DDim> i)
{
    return step<DDim>();
}

template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
rmin(DiscreteDomain<DDim> const& d)
{
    return coordinate(d.front());
}

template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
rmax(DiscreteDomain<DDim> const& d)
{
    return coordinate(d.back());
}

template <class DDim>
DDC_INLINE_FUNCTION std::enable_if_t<
        is_uniform_sampling_dimension_v<DDim>,
        typename DDim::discretization_type::continuous_element_type>
rlength(DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

template <class T>
struct is_uniform_domain : std::false_type
{
};

template <class... DDims>
struct is_uniform_domain<DiscreteDomain<DDims...>>
    : std::conditional_t<
              (is_uniform_sampling_dimension_v<DDims> && ...),
              std::true_type,
              std::false_type>
{
};

template <class T>
constexpr bool is_uniform_domain_v = is_uniform_domain<T>::value;
