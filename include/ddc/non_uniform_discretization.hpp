#pragma once

#include <cassert>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ddc/coordinate.hpp"
#include "ddc/discrete_coordinate.hpp"
#include "ddc/discretization.hpp"

/// `NonUniformDiscretization` models a non-uniform discretization of the `CDim` segment \f$[a, b]\f$.
template <class CDim>
class NonUniformDiscretization
{
public:
    using rcoord_type = Coordinate<CDim>;

    using mcoord_type = DiscreteCoordinate<NonUniformDiscretization>;

    using rdim_type = CDim;

public:
    static constexpr std::size_t rank()
    {
        return 1;
    }

private:
    Kokkos::View<rcoord_type*> m_points;

public:
    NonUniformDiscretization() = default;

    /// @brief Construct a `NonUniformDiscretization` using a brace-list, i.e. `NonUniformDiscretization mesh({0., 1.})`
    explicit NonUniformDiscretization(std::initializer_list<rcoord_type> points)
    {
        std::vector<rcoord_type> host_points(points.begin(), points.end());
        Kokkos::View<rcoord_type*, Kokkos::HostSpace> host(host_points.data(), host_points.size());
        Kokkos::resize(m_points, host.extent(0));
        Kokkos::deep_copy(m_points, host);
    }

    /// @brief Construct a `NonUniformDiscretization` using a C++20 "common range".
    template <class InputRange>
    explicit inline constexpr NonUniformDiscretization(InputRange const& points)
    {
        if constexpr (Kokkos::is_view<InputRange>::value) {
            Kokkos::deep_copy(m_points, points);
        } else {
            std::vector<rcoord_type> host_points(points.begin(), points.end());
            Kokkos::View<rcoord_type*, Kokkos::HostSpace>
                    host(host_points.data(), host_points.size());
            Kokkos::resize(m_points, host.extent(0));
            Kokkos::deep_copy(m_points, host);
        }
    }

    /// @brief Construct a `NonUniformDiscretization` using a pair of iterators.
    template <class InputIt>
    inline constexpr NonUniformDiscretization(InputIt points_begin, InputIt points_end)
    {
        std::vector<rcoord_type> host_points(points_begin, points_end);
        Kokkos::View<rcoord_type*, Kokkos::HostSpace> host(host_points.data(), host_points.size());
        Kokkos::resize(m_points, host.extent(0));
        Kokkos::deep_copy(m_points, host);
    }

    NonUniformDiscretization(NonUniformDiscretization const& x) = default;

    NonUniformDiscretization(NonUniformDiscretization&& x) = default;

    NonUniformDiscretization& operator=(NonUniformDiscretization const& x) = default;

    NonUniformDiscretization& operator=(NonUniformDiscretization&& x) = default;

    ~NonUniformDiscretization() = default;

    constexpr std::size_t size() const
    {
        return m_points.size();
    }

    /// @brief Convert a mesh index into a position in `CDim`
    constexpr rcoord_type to_real(mcoord_type const& icoord) const noexcept
    {
        return m_points[icoord.value()];
    }
};

template <class>
struct is_non_uniform_disretization : public std::false_type
{
};

template <class CDim>
struct is_non_uniform_disretization<NonUniformDiscretization<CDim>> : public std::true_type
{
};

template <class DDim>
constexpr bool is_non_uniform_disretization_v = is_non_uniform_disretization<DDim>::value;

template <class CDim>
std::ostream& operator<<(std::ostream& out, NonUniformDiscretization<CDim> const& mesh)
{
    return out << "NonUniformDiscretization(" << mesh.size() << ")";
}

template <class CDim>
Coordinate<CDim> to_real(DiscreteCoordinate<NonUniformDiscretization<CDim>> const& c)
{
    return discretization<NonUniformDiscretization<CDim>>().to_real(c);
}

template <class CDim>
Coordinate<CDim> rmin(DiscreteDomain<NonUniformDiscretization<CDim>> const& d)
{
    return to_real(d.front());
}

template <class CDim>
Coordinate<CDim> rmax(DiscreteDomain<NonUniformDiscretization<CDim>> const& d)
{
    return to_real(d.back());
}

template <class CDim>
Coordinate<CDim> rlength(DiscreteDomain<NonUniformDiscretization<CDim>> const& d)
{
    return rmax(d) - rmin(d);
}
