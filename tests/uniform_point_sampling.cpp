// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <string>
#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace ddc::experimental {

template <class Tag>
struct Attribute0
{
};

template <class Tag>
struct Attribute1
{
};

struct CoordinateTag
{
};

namespace attr {

inline constexpr Attribute1<CoordinateTag> coordinate;

} // namespace attr

template <concepts::uniform_point_sampling DDim>
KOKKOS_FUNCTION Coordinate<typename DDim::continuous_dimension_type> dim_attr(
        Attribute1<CoordinateTag>,
        DiscreteElement<DDim> const& c)
{
    return coordinate(c);
}

template <class Tag, class DDim>
KOKKOS_FUNCTION auto attribute(Attribute1<Tag> attr, DiscreteElement<DDim> const& c)
{
    return dim_attr(attr, c);
}

template <class Tag>
struct AttributeFn
{
    explicit AttributeFn(Attribute1<Tag> /*attr*/) {}

    template <class DDim>
    KOKKOS_FUNCTION auto operator()(DiscreteElement<DDim> const& c) const
    {
        return dim_attr(Attribute1<Tag>(), c);
    }
};

template <class Tag>
AttributeFn<Tag> as_fn(Attribute1<Tag> attr)
{
    return AttributeFn(attr);
}

template <class Tag, class DDim>
using attribute_t = decltype(attribute(
        std::declval<Attribute1<Tag>>(),
        std::declval<ddc::DiscreteElement<DDim>>()));

template <class ExecSpace, class Tag, class DDim>
auto as_chunk(ExecSpace const& exec_space, Attribute1<Tag> attr, DiscreteDomain<DDim> const& domain)
{
    using memory_space_t = typename ExecSpace::memory_space;
    AttributeFn const attr_fn(attr);
    ddc::Chunk chunk(domain, KokkosAllocator<attribute_t<Tag, DDim>, memory_space_t>());
    ddc::ChunkSpan const chunk_span = chunk.span_view();
    ddc::parallel_for_each(
            exec_space,
            domain,
            KOKKOS_LAMBDA(DiscreteElement<DDim> i) { chunk_span(i) = attr_fn(i); });
    return chunk;
}

// Makes only sense for bounded dimension
template <class ExecSpace, class Tag, class DDim>
auto as_chunk(ExecSpace const& exec_space, Attribute1<Tag> attr)
{
    return as_chunk(exec_space, attr, ddc::discrete_space<DDim>().domain());
}

} // namespace ddc::experimental

inline namespace anonymous_namespace_workaround_uniform_point_sampling_cpp {

struct DimX;
struct DimY;

struct DDimX : ddc::UniformPointSampling<DimX>
{
};

struct DDimY : ddc::UniformPointSampling<DimY>
{
};

ddc::Coordinate<DimX> constexpr origin(-1.);
ddc::Real constexpr step = 0.5;
ddc::DiscreteElement<DDimX> constexpr point_ix(2);
ddc::Coordinate<DimX> constexpr point_rx(0.);

} // namespace anonymous_namespace_workaround_uniform_point_sampling_cpp

TEST(UniformPointSamplingTest, Constructor)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(origin, step);
    EXPECT_EQ(ddim_x.origin(), origin);
    EXPECT_EQ(ddim_x.step(), step);
    EXPECT_EQ(ddim_x.coordinate(point_ix), point_rx);
}

TEST(UniformPointSampling, Formatting)
{
    DDimX::Impl<DDimX, Kokkos::HostSpace> const ddim_x(origin, step);
    std::stringstream oss;
    oss << ddim_x;
    EXPECT_EQ(oss.str(), "UniformPointSampling( origin=(-1), step=0.5 )");
}

TEST(UniformPointSamplingTest, Coordinate)
{
    ddc::DiscreteElement<DDimY> const point_iy(4);
    ddc::Coordinate<DimY> const point_ry(-6);

    ddc::DiscreteElement<DDimX, DDimY> const point_ixy(point_ix, point_iy);
    ddc::Coordinate<DimX, DimY> const point_rxy(point_rx, point_ry);

    ddc::init_discrete_space<DDimX>(origin, step);
    ddc::init_discrete_space<DDimY>(ddc::Coordinate<DimY>(-10.), 1.);
    EXPECT_EQ(ddc::coordinate(point_ix), point_rx);
    EXPECT_EQ(ddc::coordinate(point_iy), point_ry);
    EXPECT_EQ(ddc::coordinate(point_ixy), point_rxy);
}

TEST(UniformPointSamplingTest, Attributes)
{
    ddc::DiscreteDomain<DDimX> const ddom_x(point_ix, ddc::DiscreteVector<DDimX>(2));
    ddc::init_discrete_space<DDimX>(origin, step);
    EXPECT_EQ(ddc::origin<DDimX>(), origin);
    EXPECT_EQ(ddc::step<DDimX>(), step);
    EXPECT_EQ(ddc::rmin(ddom_x), point_rx);
    EXPECT_EQ(ddc::rmax(ddom_x), point_rx + step);
    EXPECT_EQ(ddc::rlength(ddom_x), step);
    EXPECT_EQ(ddc::distance_at_left(point_ix), step);
    EXPECT_EQ(ddc::distance_at_right(point_ix), step);
}

TEST(UniformPointSamplingTest, ExperimentalAttributes)
{
    namespace ddcexp = ddc::experimental;
    ddc::DiscreteDomain<DDimX> const ddom_x(point_ix, ddc::DiscreteVector<DDimX>(2));
    ddc::init_discrete_space<DDimX>(origin, step);
    Kokkos::DefaultHostExecutionSpace const exec_space;
    ddc::Chunk const chk = ddcexp::as_chunk(exec_space, ddcexp::attr::coordinate, ddom_x);
    exec_space.fence();
    for (ddc::DiscreteElement<DDimX> const ix : ddom_x) {
        EXPECT_DOUBLE_EQ(chk(ix), ddc::coordinate(ix));
    }
}
