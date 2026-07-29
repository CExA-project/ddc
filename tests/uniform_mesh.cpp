// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <iosfwd>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace ddc {

namespace detail {

struct UniformMeshCellBase
{
};

void print_uniform_mesh_cell(std::ostream& os, CoordinateElement origin, Real step)
{
    os << "UniformMeshCell(origin=" << origin << ", step=" << step << ')';
}

} // namespace detail

/** UniformMeshCell models a uniform discretization of the provided continuous dimension
 */
template <class CDim>
class UniformMeshCell : detail::UniformMeshCellBase
{
public:
    using continuous_dimension_type = CDim;

    using discrete_dimension_type = UniformMeshCell;

public:
    template <class DDim, class MemorySpace>
    class Impl
    {
        template <class ODDim, class OMemorySpace>
        friend class Impl;

    private:
        Coordinate<CDim> m_origin;

        Real m_step;

        DiscreteElement<DDim> m_reference;

    public:
        using discrete_dimension_type = UniformMeshCell;

        using discrete_domain_type = DiscreteDomain<DDim>;

        using discrete_element_type = DiscreteElement<DDim>;

        using discrete_vector_type = DiscreteVector<DDim>;

        Impl() noexcept
            : m_origin(0)
            , m_step(1)
            , m_reference(create_reference_discrete_element<DDim>())
        {
        }

        Impl(Impl const&) = delete;

        template <class OriginMemorySpace>
        explicit Impl(Impl<DDim, OriginMemorySpace> const& impl)
            : m_origin(impl.m_origin)
            , m_step(impl.m_step)
            , m_reference(impl.m_reference)
        {
        }

        Impl(Impl&&) = default;

        /** @brief Construct a `Impl` from a point and a spacing step.
         *
         * @param origin the real coordinate of mesh coordinate 0
         * @param step   the real distance between two points of mesh distance 1
         */
        Impl(Coordinate<CDim> origin, Real step)
            : m_origin(origin)
            , m_step(step)
            , m_reference(create_reference_discrete_element<DDim>())
        {
            assert(step > 0);
        }

        ~Impl() = default;

        Impl& operator=(Impl const& x) = delete;

        Impl& operator=(Impl&& x) = default;

        /// @brief Lower bound index of the mesh
        KOKKOS_FUNCTION Coordinate<CDim> origin() const noexcept
        {
            return m_origin;
        }

        /// @brief Lower bound index of the mesh
        KOKKOS_FUNCTION discrete_element_type front() const noexcept
        {
            return m_reference;
        }

        /// @brief Spacing step of the mesh
        KOKKOS_FUNCTION Real step() const
        {
            return m_step;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        KOKKOS_FUNCTION Real volume(discrete_element_type const& icoord) const noexcept
        {
            return m_step;
        }

        /// @brief Convert a mesh index into a position in `CDim`
        KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> center(
                discrete_element_type const& icoord) const noexcept
        {
            return m_origin + Coordinate<CDim>((icoord - front()) * m_step) + m_step / 2;
        }
    };

    /** Construct a Impl<Kokkos::HostSpace> and associated discrete_domain_type from a segment
     *  \f$[a, b] \subset [a, +\infty[\f$ and a number of points `n`.
     *  Note that there is no guarantee that either the boundaries a or b will be exactly represented in the sampling.
     *  One should expect usual floating point rounding errors.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n number of points to map on the segment \f$[a, b]\f$ including a & b
     */
    template <class DDim>
    static std::tuple<typename DDim::template Impl<DDim, Kokkos::HostSpace>, DiscreteDomain<DDim>>
    init(Coordinate<CDim> a, Coordinate<CDim> b, DiscreteVector<DDim> n)
    {
        assert(a < b);
        assert(n > 0);
        typename DDim::template Impl<DDim, Kokkos::HostSpace>
                disc(a, Coordinate<CDim>((b - a) / n));
        DiscreteDomain<DDim> domain(disc.front(), n);
        return std::make_tuple(std::move(disc), std::move(domain));
    }

    /** Construct a uniform `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
     *  number of points `n`.
     *  Note that there is no guarantee that either the boundaries a or b will be exactly represented in the sampling.
     *  One should expect usual floating point rounding errors.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * @param n_ghosts_before number of additional "ghost" points before the segment
     * @param n_ghosts_after number of additional "ghost" points after the segment
     */
    template <class DDim>
    static std::tuple<
            typename DDim::template Impl<DDim, Kokkos::HostSpace>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>>
    init_ghosted(
            Coordinate<CDim> a,
            Coordinate<CDim> b,
            DiscreteVector<DDim> n,
            DiscreteVector<DDim> n_ghosts_before,
            DiscreteVector<DDim> n_ghosts_after)
    {
        assert(a < b);
        assert(n > 0);
        Real const discretization_step = (b - a) / n;
        typename DDim::template Impl<DDim, Kokkos::HostSpace>
                disc(a - n_ghosts_before.value() * discretization_step, discretization_step);
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

    /** Construct a uniform `DiscreteDomain` from a segment \f$[a, b] \subset [a, +\infty[\f$ and a
     *  number of points `n`.
     *  Note that there is no guarantee that either the boundaries a or b will be exactly represented in the sampling.
     *  One should expect usual floating point rounding errors.
     *
     * @param a coordinate of the first point of the domain
     * @param b coordinate of the last point of the domain
     * @param n the number of points to map the segment \f$[a, b]\f$ including a & b
     * @param n_ghosts number of additional "ghost" points before and after the segment
     */
    template <class DDim>
    static std::tuple<
            typename DDim::template Impl<DDim, Kokkos::HostSpace>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>,
            DiscreteDomain<DDim>>
    init_ghosted(
            Coordinate<CDim> a,
            Coordinate<CDim> b,
            DiscreteVector<DDim> n,
            DiscreteVector<DDim> n_ghosts)
    {
        return init_ghosted(a, b, n, n_ghosts, n_ghosts);
    }
};

template <class DDim>
struct is_uniform_mesh_cell : public std::is_base_of<detail::UniformMeshCellBase, DDim>::type
{
};

template <class DDim>
constexpr bool is_uniform_mesh_cell_v = is_uniform_mesh_cell<DDim>::value;

namespace concepts {

template <class DDim>
concept uniform_mesh_cell = is_uniform_mesh_cell_v<DDim>;

}

template <class DDimImpl>
std::ostream& operator<<(std::ostream& os, DDimImpl const& mesh)
    requires(concepts::uniform_mesh_cell<typename DDimImpl::discrete_dimension_type>)
{
    print_uniform_mesh_cell(os, mesh.origin(), mesh.step());
    return os;
}

template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION constexpr Real volume(DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().volume(c);
}

template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION constexpr Coordinate<typename DDim::continuous_dimension_type> center(
        DiscreteElement<DDim> const& c)
{
    return discrete_space<DDim>().center(c);
}

/// @brief Lower bound index of the mesh
template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> origin() noexcept
{
    return discrete_space<DDim>().origin();
}

/// @brief Lower bound index of the mesh
template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION DiscreteElement<DDim> front() noexcept
{
    return discrete_space<DDim>().front();
}

/// @brief Spacing step of the mesh
template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION Real step() noexcept
{
    return discrete_space<DDim>().step();
}

template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmin(
        DiscreteDomain<DDim> const& d)
{
    return center(d.front()) - step<DDim>() / 2;
}

template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rmax(
        DiscreteDomain<DDim> const& d)
{
    return center(d.back()) + step<DDim>() / 2;
}

template <concepts::uniform_mesh_cell DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> rlength(
        DiscreteDomain<DDim> const& d)
{
    return rmax(d) - rmin(d);
}

template <class... DDims>
KOKKOS_FUNCTION Coordinate<typename DDims::continuous_dimension_type...> center(
        DiscreteElement<DDims...> const& c)
    requires(sizeof...(DDims) > 1)
{
    return Coordinate<typename DDims::continuous_dimension_type...>(
            center(DiscreteElement<DDims>(c))...);
}

template <class... DDims>
KOKKOS_FUNCTION Real volume(DiscreteElement<DDims...> const& c)
    requires(sizeof...(DDims) > 1)
{
    return (1 * ... * volume(DiscreteElement<DDims>(c)));
}

} // namespace ddc

inline namespace anonymous_namespace_workaround_uniform_mesh_cell_cpp {

struct DimX
{
};
struct DimY
{
};

struct DDimX : ddc::UniformMeshCell<DimX>
{
};

struct DDimY : ddc::UniformMeshCell<DimY>
{
};

ddc::Coordinate<DimX> constexpr origin(-1.);
ddc::Real constexpr step = 0.5;
ddc::DiscreteElement<DDimX> constexpr cell_ix(2);
ddc::Coordinate<DimX> constexpr cell_center_x(0.25);

} // namespace anonymous_namespace_workaround_uniform_mesh_cell_cpp

TEST(UniformMeshCellTest, Constructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(origin, step);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.center(cell_ix), cell_center_x);
}

TEST(UniformMeshCell, Formatting)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(origin, step);
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "UniformMeshCell(origin=-1, step=0.5)");
}

TEST(UniformMeshCellTest, Coordinate)
{
    ddc::DiscreteElement<DDimY> const cell_iy(4);
    ddc::Coordinate<DimY> const cell_ry(-1);
    ddc::Real constexpr step_y = 2.;

    ddc::DiscreteElement<DDimX, DDimY> const cell_ixy(cell_ix, cell_iy);
    ddc::Coordinate<DimX, DimY> const cell_center_xy(cell_center_x, cell_ry);

    ddc::init_discrete_space<DDimX>(origin, step);
    ddc::init_discrete_space<DDimY>(ddc::Coordinate<DimY>(-10.), step_y);
    EXPECT_EQ(ddc::center(cell_ix), cell_center_x);
    EXPECT_EQ(ddc::volume(cell_ix), step);
    EXPECT_EQ(ddc::center(cell_iy), cell_ry);
    EXPECT_EQ(ddc::volume(cell_iy), step_y);
    EXPECT_EQ(ddc::center(cell_ixy), cell_center_xy);
    EXPECT_EQ(ddc::volume(cell_ixy), step * step_y);
}

TEST(UniformMeshCellTest, Attributes)
{
    ddc::DiscreteDomain<DDimX> const ddom_x(cell_ix, ddc::DiscreteVector<DDimX>(1));
    ddc::init_discrete_space<DDimX>(origin, step);
    EXPECT_EQ(ddc::origin<DDimX>(), origin);
    EXPECT_EQ(ddc::step<DDimX>(), step);
    EXPECT_EQ(ddc::rmin(ddom_x), cell_center_x - step / 2);
    EXPECT_EQ(ddc::rmax(ddom_x), cell_center_x + step / 2);
    EXPECT_EQ(ddc::rlength(ddom_x), step);
}
