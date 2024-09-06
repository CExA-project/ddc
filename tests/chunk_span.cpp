// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <type_traits>

#include <ddc/ddc.hpp>

#include <gtest/gtest.h>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(CHUNK_SPAN_CPP) {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;

template <class Datatype>
using ChunkX = ddc::Chunk<Datatype, DDomX>;

template <class Datatype>
using ChunkSpanX = ddc::ChunkSpan<Datatype, DDomX>;

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(CHUNK_SPAN_CPP)

TEST(ChunkSpan1DTest, ConstructionFromChunk)
{
    EXPECT_FALSE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double>>));
    EXPECT_FALSE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double> const>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<double>, ChunkX<double>&>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<const double>, ChunkX<double>&>));
    EXPECT_TRUE((std::is_constructible_v<ChunkSpanX<const double>, ChunkX<double> const&>));
}
