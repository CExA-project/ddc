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
#include "ddc/experimental/concepts.hpp"
#include "ddc/experimental/discrete_set.hpp"
#include "ddc/experimental/mesh_elements.hpp"

namespace ddc::experimental {

struct UniformMeshBase : public DiscreteSet
{
};

template <class NamedDSet>
constexpr bool is_uniform_mesh_v = std::is_base_of_v<UniformMeshBase, NamedDSet>;

template <class CDim>
class UniformMesh;

/** UniformMesh models a uniform discretization of the provided continuous dimension
 *
 * UniformMesh is a discrete set meaning that it contains a coherent set of discrete dimensions: nodes and cells.
 */
template <class CDim>
class UniformMesh : UniformMeshBase
{
public:
    using continuous_dimension_type = CDim;

    using continuous_element_type = Coordinate<CDim>;


    using discrete_set_type = UniformMesh;

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
        using discrete_set_type = UniformMesh<CDim>;

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

        /** @brief Construct a `Impl` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
         *
         * @param a the coordinate of a first real point (will have mesh coordinate 0)
         * @param b the coordinate of the second real point (will have mesh coordinate `n-1`)
         * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
         */
        constexpr Impl(
                continuous_element_type a,
                continuous_element_type b,
                DiscreteVectorElement n)
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

/** Construct a Impl<Kokkos::HostSpace> and associated discrete_domain_type from a segment
 *  \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
 *
 * @param a coordinate of the first point of the domain
 * @param b coordinate of the last point of the domain
 * @param n number of points to map on the segment \f$[a, b]\f$ including a & b
 */
template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
std::tuple<typename NamedDSet::template Impl<Kokkos::HostSpace>, DiscreteDomain<Node<NamedDSet>>>
uniform_mesh_init(
        Coordinate<continuous_dimension_t<NamedDSet>> a,
        Coordinate<continuous_dimension_t<NamedDSet>> b,
        DiscreteVector<Node<NamedDSet>> n)
{
    assert(a < b);
    assert(n > 1);
    typename NamedDSet::template Impl<Kokkos::HostSpace>
            disc(a, Coordinate<continuous_dimension_t<NamedDSet>> {(b - a) / (n - 1)});
    DiscreteDomain<Node<NamedDSet>> domain {DiscreteElement<Node<NamedDSet>>(disc.front()), n};
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
template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
std::tuple<
        typename NamedDSet::template Impl<Kokkos::HostSpace>,
        DiscreteDomain<Node<NamedDSet>>,
        DiscreteDomain<Node<NamedDSet>>,
        DiscreteDomain<Node<NamedDSet>>,
        DiscreteDomain<Node<NamedDSet>>>
uniform_mesh_init_ghosted(
        Coordinate<continuous_dimension_t<NamedDSet>> a,
        Coordinate<continuous_dimension_t<NamedDSet>> b,
        DiscreteVector<Node<NamedDSet>> n,
        DiscreteVector<Node<NamedDSet>> n_ghosts_before,
        DiscreteVector<Node<NamedDSet>> n_ghosts_after)
{
    assert(a < b);
    assert(n > 1);
    double discretization_step {(b - a) / (n - 1)};
    typename NamedDSet::template Impl<Kokkos::HostSpace>
            disc(a - n_ghosts_before.value() * discretization_step, discretization_step);
    DiscreteDomain<Node<NamedDSet>> ghosted_domain = DiscreteDomain<Node<NamedDSet>>(
            DiscreteElement<Node<NamedDSet>>(disc.front()),
            n + n_ghosts_before + n_ghosts_after);
    DiscreteDomain<Node<NamedDSet>> pre_ghost = DiscreteDomain<Node<
            NamedDSet>>(DiscreteElement<Node<NamedDSet>>(ghosted_domain.front()), n_ghosts_before);
    DiscreteDomain<Node<NamedDSet>> main_domain = DiscreteDomain<Node<NamedDSet>>(
            DiscreteElement<Node<NamedDSet>>(ghosted_domain.front() + n_ghosts_before),
            n);
    DiscreteDomain<Node<NamedDSet>> post_ghost = DiscreteDomain<Node<
            NamedDSet>>(DiscreteElement<Node<NamedDSet>>(main_domain.back() + 1), n_ghosts_after);
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
template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
std::tuple<
        typename NamedDSet::template Impl<Kokkos::HostSpace>,
        DiscreteDomain<Node<NamedDSet>>,
        DiscreteDomain<Node<NamedDSet>>,
        DiscreteDomain<Node<NamedDSet>>,
        DiscreteDomain<Node<NamedDSet>>>
uniform_mesh_init_ghosted(
        Coordinate<continuous_dimension_t<NamedDSet>> a,
        Coordinate<continuous_dimension_t<NamedDSet>> b,
        DiscreteVector<Node<NamedDSet>> n,
        DiscreteVector<Node<NamedDSet>> n_ghosts)
{
    return uniform_mesh_init_ghosted(a, b, n, n_ghosts, n_ghosts);
}

template <
        class DSetImpl,
        std::enable_if_t<is_uniform_mesh_v<typename DSetImpl::discrete_set_type>, int> = 0>
std::ostream& operator<<(std::ostream& out, DSetImpl const& mesh)
{
    return out << "UniformMesh( origin=" << mesh.origin() << ", step=" << mesh.step() << " )";
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Cell<NamedDSet>> cell_left(
        DiscreteElement<Node<NamedDSet>> const& inode)
{
    return DiscreteElement<Cell<NamedDSet>>(inode.uid() - 1);
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Cell<NamedDSet>> cell_right(
        DiscreteElement<Node<NamedDSet>> const& inode)
{
    return DiscreteElement<Cell<NamedDSet>>(inode.uid());
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Node<NamedDSet>> node_left(
        DiscreteElement<Cell<NamedDSet>> const& icell)
{
    return DiscreteElement<Node<NamedDSet>>(icell.uid());
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr DiscreteElement<Node<NamedDSet>> node_right(
        DiscreteElement<Cell<NamedDSet>> const& icell)
{
    return DiscreteElement<Node<NamedDSet>>(icell.uid() + 1);
}

/// @brief Lower bound index of the mesh
template <class NamedDSet>
DDC_INLINE_FUNCTION std::
        enable_if_t<is_uniform_mesh_v<NamedDSet>, Coordinate<continuous_dimension_t<NamedDSet>>>
        origin() noexcept
{
    return Coordinate<continuous_dimension_t<NamedDSet>>(discrete_set<NamedDSet>().origin());
}

/// @brief Lower bound index of the mesh
template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION std::enable_if_t<is_uniform_mesh_v<NamedDSet>, DiscreteElement<Node<NamedDSet>>>
front() noexcept
{
    return DiscreteElement<Node<NamedDSet>>(discrete_set<NamedDSet>().front());
}

/// @brief Spacing step of the mesh
template <class NamedDSet>
DDC_INLINE_FUNCTION std::
        enable_if_t<is_uniform_mesh_v<NamedDSet>, Coordinate<continuous_dimension_t<NamedDSet>>>
        step() noexcept
{
    return Coordinate<continuous_dimension_t<NamedDSet>>(discrete_set<NamedDSet>().step());
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION constexpr Coordinate<continuous_dimension_t<NamedDSet>> coordinate(
        DiscreteElement<Node<NamedDSet>> const& inode)
{
    return discrete_set<NamedDSet>().coordinate(inode.uid());
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> distance_at_left(
        DiscreteElement<Node<NamedDSet>>)
{
    return step<NamedDSet>();
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> distance_at_right(
        DiscreteElement<Node<NamedDSet>>)
{
    return step<NamedDSet>();
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> rlength(
        DiscreteElement<Cell<NamedDSet>> const& icell)
{
    return step<NamedDSet>();
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> rmin(
        DiscreteDomain<Node<NamedDSet>> const& d)
{
    return coordinate(d.front());
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> rmax(
        DiscreteDomain<Node<NamedDSet>> const& d)
{
    return coordinate(d.back());
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> rlength(
        DiscreteDomain<Node<NamedDSet>> const& d)
{
    return rmax(d) - rmin(d);
}

template <class NamedDSet, std::enable_if_t<is_uniform_mesh_v<NamedDSet>, int> = 0>
DDC_INLINE_FUNCTION Coordinate<continuous_dimension_t<NamedDSet>> rlength(
        DiscreteDomain<Cell<NamedDSet>> const& d)
{
    return d.size() * step<NamedDSet>();
}

} // namespace ddc::experimental
