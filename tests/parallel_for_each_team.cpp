// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <utility>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "helper_count.hpp"

inline namespace anonymous_namespace_workaround_parallel_for_each_team_cpp {

using DElem0D = ddc::DiscreteElement<>;
using DVect0D = ddc::DiscreteVector<>;
using DDom0D = ddc::DiscreteDomain<>;

struct DDimX : ddc::DiscreteDimension
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

struct DDimY : ddc::DiscreteDimension
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

// Increments once, from the first thread of the team, the element assigned to the team.
template <typename ExecSpace, typename ChunkSpanType>
class IncrementOncePerTeamFn
{
    ChunkSpanType m_chunk_span;

public:
    explicit IncrementOncePerTeamFn(ChunkSpanType chunk_span) noexcept
        : m_chunk_span(std::move(chunk_span))
    {
    }

    KOKKOS_FUNCTION void operator()(
            ddc::TeamHandle<ExecSpace> const& team,
            ChunkSpanType::discrete_element_type const& delem) const noexcept
    {
        Kokkos::single(Kokkos::PerTeam(team), [&]() { m_chunk_span(delem) += 1; });
    }
};

// Stores the rank of the team in the element assigned to the team.
template <typename ExecSpace, typename ChunkSpanType>
class StoreLeagueRankFn
{
    ChunkSpanType m_chunk_span;

public:
    explicit StoreLeagueRankFn(ChunkSpanType chunk_span) noexcept
        : m_chunk_span(std::move(chunk_span))
    {
    }

    KOKKOS_FUNCTION void operator()(
            ddc::TeamHandle<ExecSpace> const& team,
            typename ChunkSpanType::discrete_element_type const& delem) const noexcept
    {
        Kokkos::single(Kokkos::PerTeam(team), [&]() { m_chunk_span(delem) = team.league_rank(); });
    }
};

// Uses the team handle for a nested parallel loop: each element of the domain is incremented by
// the sum of the values of a team-level reduction.
template <typename ExecSpace, typename ChunkSpanType>
class NestedReduceFn
{
    ChunkSpanType m_chunk_span;

    int m_n;

public:
    NestedReduceFn(ChunkSpanType chunk_span, int n) noexcept
        : m_chunk_span(std::move(chunk_span))
        , m_n(n)
    {
    }

    KOKKOS_FUNCTION void operator()(
            ddc::TeamHandle<ExecSpace> const& team,
            typename ChunkSpanType::discrete_element_type const& delem) const noexcept
    {
        int sum = 0;
        Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, m_n),
                [&](int const i, int& local_sum) { local_sum += i; },
                sum);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { m_chunk_span(delem) = sum; });
    }
};

template <typename ExecSpace, typename ChunkSpanType>
auto increment_once_per_team(ChunkSpanType const& chunk_span)
{
    return IncrementOncePerTeamFn<ExecSpace, ChunkSpanType>(chunk_span);
}

template <typename ExecSpace, typename ChunkSpanType>
auto store_league_rank(ChunkSpanType const& chunk_span)
{
    return StoreLeagueRankFn<ExecSpace, ChunkSpanType>(chunk_span);
}

template <typename ExecSpace, typename ChunkSpanType>
auto nested_reduce(ChunkSpanType const& chunk_span, int n)
{
    return NestedReduceFn<ExecSpace, ChunkSpanType>(chunk_span, n);
}

} // namespace anonymous_namespace_workaround_parallel_for_each_team_cpp

TEST(ParallelForEachTeamHost, ZeroDimension)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDom0D const dom;
    Kokkos::View<int, Kokkos::HostSpace> const storage("storage");
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each_team(ExecSpace(), dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(
            ddc::testing::
                    count(ExecSpace(), Kokkos::View<int*, Kokkos::HostSpace>(storage.data(), 1), 1),
            DDom0D::size());
}

TEST(ParallelForEachTeamHost, OneDimension)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each_team(ExecSpace(), dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachTeamHost, TwoDimensions)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const
            view(Kokkos::View<int**, Kokkos::HostSpace>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each_team(ExecSpace(), dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachTeamHost, TwoDimensionsWithLabel)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const
            view(Kokkos::View<int**, Kokkos::HostSpace>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each_team(
            "labeled_parallel_for_each_team",
            ExecSpace(),
            dom,
            increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachTeamHost, TwoDimensionsWithTeamSize)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const
            view(Kokkos::View<int**, Kokkos::HostSpace>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each_team(
            "labeled_parallel_for_each_team",
            ExecSpace(),
            dom,
            1,
            1,
            increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}

// Checks that the element assigned to a team is consistent with the rank of the team: the last
// dimension is the fastest varying one.
TEST(ParallelForEachTeamHost, TeamRankMatchesElement)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int**, Kokkos::HostSpace> const storage("storage", nelems_x, nelems_y);
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each_team(ExecSpace(), dom, store_league_rank<ExecSpace>(view));

    for (int ix = 0; ix < nelems_x.value(); ++ix) {
        for (int iy = 0; iy < nelems_y.value(); ++iy) {
            EXPECT_EQ(storage(ix, iy), ix * nelems_y.value() + iy);
        }
    }
}

TEST(ParallelForEachTeamHost, TeamHandleNestedParallelism)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    int constexpr n = 7;
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each_team(ExecSpace(), dom, nested_reduce<ExecSpace>(view, n));
    // 0 + 1 + ... + (n - 1)
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, n * (n - 1) / 2), dom.size());
}

TEST(ParallelForEachTeamHost, EmptyDomain)
{
    using ExecSpace = Kokkos::DefaultHostExecutionSpace;
    DDomX const dom(lbound_x, DVectX(0));
    Kokkos::View<int*, Kokkos::HostSpace> const storage("storage", 1);
    ddc::ChunkSpan const view(Kokkos::View<int*, Kokkos::HostSpace>(storage.data(), 0), dom);

    ddc::parallel_for_each_team(ExecSpace(), dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(storage(0), 0);
}

TEST(ParallelForEachTeamDevice, ZeroDimension)
{
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    DDom0D const dom;
    Kokkos::View<int> const storage("storage");
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each_team(dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(
            ddc::testing::count(ExecSpace(), Kokkos::View<int*>(storage.data(), 1), 1),
            DDom0D::size());
}

TEST(ParallelForEachTeamDevice, OneDimension)
{
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    DDomX const dom(lbound_x, nelems_x);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(storage, dom);

    ddc::parallel_for_each_team(dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachTeamDevice, TwoDimensions)
{
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    DDomXY const dom(lbound_x_y, nelems_x_y);
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each_team(dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}

TEST(ParallelForEachTeamDevice, TwoDimensionsStrided)
{
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using DDomXY = ddc::StridedDiscreteDomain<DDimX, DDimY>;
    DDomXY const dom(lbound_x_y, nelems_x_y, DVectXY(3, 3));
    Kokkos::View<int*> const storage("storage", dom.size());
    ddc::ChunkSpan const view(Kokkos::View<int**>(storage.data(), nelems_x, nelems_y), dom);

    ddc::parallel_for_each_team(dom, increment_once_per_team<ExecSpace>(view));
    EXPECT_EQ(ddc::testing::count(ExecSpace(), storage, 1), dom.size());
}
