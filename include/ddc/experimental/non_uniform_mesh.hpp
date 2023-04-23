// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "ddc/coordinate.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"
#include "ddc/discrete_vector.hpp"
#include "ddc/experimental/concepts.hpp"
#include "ddc/experimental/discrete_set.hpp"
#include "ddc/experimental/mesh_elements.hpp"

namespace ddc::experimental {

struct NonUniformMeshBase : public DiscreteSet
{
};

template <class NamedDSet>
constexpr bool is_non_uniform_mesh_v = std::is_base_of_v<NonUniformMeshBase, NamedDSet>;

template <class CDim>
class NonUniformMesh;

/// `NonUniformMesh` models a non-uniform discretization of the `CDim` segment \f$[a, b]\f$.
template <class CDim>
class NonUniformMesh : public NonUniformMeshBase
{
public:
    using continuous_dimension_type = CDim;

    using coordinate_node_type = Coordinate<CDim>;


    using discrete_set_type = NonUniformMesh;

public:
    template <class MemorySpace>
    class Impl
    {
        template <class OMemorySpace>
        friend class Impl;

        Kokkos::View<coordinate_node_type*, MemorySpace> m_points;

    public:
        using discrete_set_type = NonUniformMesh<CDim>;

        Impl() = default;

        /// @brief Construct a `NonUniformMesh` using a brace-list, i.e. `NonUniformMesh mesh({0., 1.})`
        explicit Impl(std::initializer_list<coordinate_node_type> points)
        {
            std::vector<coordinate_node_type> host_points(points.begin(), points.end());
            Kokkos::View<coordinate_node_type*, Kokkos::HostSpace>
                    host(host_points.data(), host_points.size());
            Kokkos::resize(m_points, host.extent(0));
            Kokkos::deep_copy(m_points, host);
        }

        /// @brief Construct a `NonUniformMesh` using a C++20 "common range".
        template <class InputRange>
        explicit inline constexpr Impl(InputRange const& points)
        {
            if constexpr (Kokkos::is_view<InputRange>::value) {
                Kokkos::deep_copy(m_points, points);
            } else {
                std::vector<coordinate_node_type> host_points(points.begin(), points.end());
                Kokkos::View<coordinate_node_type*, Kokkos::HostSpace>
                        host(host_points.data(), host_points.size());
                Kokkos::resize(m_points, host.extent(0));
                Kokkos::deep_copy(m_points, host);
            }
        }

        /// @brief Construct a `NonUniformMesh` using a pair of iterators.
        template <class InputIt>
        inline constexpr Impl(InputIt points_begin, InputIt points_end)
        {
            std::vector<coordinate_node_type> host_points(points_begin, points_end);
            Kokkos::View<coordinate_node_type*, Kokkos::HostSpace>
                    host(host_points.data(), host_points.size());
            Kokkos::resize(m_points, host.extent(0));
            Kokkos::deep_copy(m_points, host);
        }

        template <class OriginMemorySpace>
        explicit Impl(Impl<OriginMemorySpace> const& impl)
            : m_points(Kokkos::create_mirror_view_and_copy(MemorySpace(), impl.m_points))
        {
        }

        Impl(Impl const& x) = delete;

        Impl(Impl&& x) = default;

        ~Impl() = default;

        constexpr std::size_t size() const
        {
            return m_points.size();
        }

        /// @brief Convert a mesh index into a position in `CDim`
        constexpr coordinate_node_type coordinate(DiscreteElementType const& icoord) const noexcept
        {
            return m_points(icoord);
        }
    };
};

template <
        class NamedDSet,
        class InputRange,
        std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
std::tuple<typename NamedDSet::template Impl<Kokkos::HostSpace>, DiscreteDomain<Node<NamedDSet>>>
non_uniform_mesh_init(InputRange const& input_rng)
{
    typename NamedDSet::template Impl<Kokkos::HostSpace> disc(std::forward<InputRange>(input_rng));
    DiscreteDomain<Node<NamedDSet>>
            domain {DiscreteElement<Node<NamedDSet>>(0),
                    DiscreteVector<Node<NamedDSet>>(disc.size())};
    return std::make_tuple(std::move(disc), std::move(domain));
}

template <
        class DSetImpl,
        std::enable_if_t<is_non_uniform_mesh_v<typename DSetImpl::discrete_set_type>, int> = 0>
std::ostream& operator<<(std::ostream& out, DSetImpl const& mesh)
{
    return out << "NonUniformMesh(" << mesh.size() << ")";
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Cell<NamedDSet>> cell_left(
        DiscreteElement<Node<NamedDSet>> const& inode)
{
    return DiscreteElement<Node<NamedDSet>>(inode.uid() - 1);
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Cell<NamedDSet>> cell_right(
        DiscreteElement<Node<NamedDSet>> const& inode)
{
    return DiscreteElement<Node<NamedDSet>>(inode.uid());
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Node<NamedDSet>> node_left(
        DiscreteElement<Cell<NamedDSet>> const& icell)
{
    return DiscreteElement<Node<NamedDSet>>(icell.uid());
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Node<NamedDSet>> node_right(
        DiscreteElement<Cell<NamedDSet>> const& icell)
{
    return DiscreteElement<Node<NamedDSet>>(icell.uid() + 1);
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> coordinate(DiscreteElement<Node<NamedDSet>> const& inode)
{
    return discrete_space<NonUniformMesh<NamedDSet>>().coordinate(inode);
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> distance_at_left(DiscreteElement<Node<NamedDSet>> i)
{
    return coordinate(i) - coordinate(i - 1);
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> distance_at_right(DiscreteElement<Node<NamedDSet>> i)
{
    return coordinate(i + 1) - coordinate(i);
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> rlength(DiscreteElement<Cell<NamedDSet>> const& icell)
{
    return coordinate(node_right(icell)) - coordinate(node_left(icell));
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> rmin(DiscreteDomain<Node<NamedDSet>> const& d)
{
    return coordinate(d.front());
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> rmax(DiscreteDomain<Node<NamedDSet>> const& d)
{
    return coordinate(d.back());
}

template <class NamedDSet, std::enable_if_t<is_non_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<NamedDSet> rlength(DiscreteDomain<Node<NamedDSet>> const& d)
{
    return rmax(d) - rmin(d);
}

} // namespace ddc::experimental
