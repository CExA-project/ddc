// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

inline namespace anonymous_namespace_workaround_parallel_for_each_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;

using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;
using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;

DElemX constexpr lbound_x = ddc::init_trivial_half_bounded_space<DDimX>();
DVectX constexpr nelems_x(10);

DElemY constexpr lbound_y = ddc::init_trivial_half_bounded_space<DDimY>();
DVectY constexpr nelems_y(12);

DElemXY constexpr lbound_x_y(lbound_x, lbound_y);
DVectXY constexpr nelems_x_y(nelems_x, nelems_y);

template <typename Support, typename Layout, typename MemorySpace>
class IncrementFn
{
    ddc::ChunkSpan<int, Support, Layout, MemorySpace> m_chunk_span;

public:
    explicit IncrementFn(ddc::ChunkSpan<int, Support, Layout, MemorySpace> chunk_span) noexcept
        : m_chunk_span(std::move(chunk_span))
    {
    }

    KOKKOS_FUNCTION void operator()(
            typename Support::discrete_element_type const& delem) const noexcept
    {
        m_chunk_span(delem) += 1;
    }
};

} // namespace anonymous_namespace_workaround_parallel_for_each_cpp

TEST(ParallelForEachParallelHost, ZeroDimension)
{
    DDom0D const dom;
    Kokkos::View<int, Kokkos::HostSpace> const storage("storage");
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each(Kokkos::DefaultHostExecutionSpace(), dom, IncrementFn(view));
    EXPECT_EQ(
            Kokkos::Experimental::
                    count(Kokkos::DefaultHostExecutionSpace(),
                          Kokkos::View<int*, Kokkos::HostSpace>(storage.data(), 1),
                          1),
            DDom0D::size());
}

TEST(ParallelForEachParallelHost, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each(Kokkos::DefaultHostExecutionSpace(), dom, IncrementFn(view));
    EXPECT_EQ(
            Kokkos::Experimental::count(Kokkos::DefaultHostExecutionSpace(), storage, 1),
            dom.size());
}

TEST(ParallelForEachParallelHost, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const
            view(Kokkos::View<int**, Kokkos::HostSpace>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each(Kokkos::DefaultHostExecutionSpace(), dom, IncrementFn(view));
    EXPECT_EQ(
            Kokkos::Experimental::count(Kokkos::DefaultHostExecutionSpace(), storage, 1),
            dom.size());
}

TEST(ParallelForEachParallelDevice, ZeroDimension)
{
    DDom0D const dom;
    Kokkos::View<int> const storage("storage");
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each(dom, IncrementFn(view));
    EXPECT_EQ(
            Kokkos::Experimental::
                    count(Kokkos::DefaultExecutionSpace(),
                          Kokkos::View<int*>(storage.data(), 1),
                          1),
            DDom0D::size());
}

TEST(ParallelForEachParallelDevice, OneDimension)
{
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each(dom, IncrementFn(view));
    EXPECT_EQ(Kokkos::Experimental::count(Kokkos::DefaultExecutionSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachParallelDevice, TwoDimensions)
{
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each(dom, IncrementFn(view));
    EXPECT_EQ(Kokkos::Experimental::count(Kokkos::DefaultExecutionSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachParallelDevice, TwoDimensionsStrided)
{
    using DDomXY = ddc::StridedDiscreteDomain<DDimX, DDimY>;
    DDomXY const dom(lbound_x_y, nelems_x_y, DVectXY(3, 3));
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each(dom, IncrementFn(view));
    EXPECT_EQ(Kokkos::Experimental::count(Kokkos::DefaultExecutionSpace(), storage, 1), dom.size());
}
