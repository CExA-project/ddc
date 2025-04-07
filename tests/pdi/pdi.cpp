// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <string>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include <gtest/gtest.h>

#include <paraconf.h>
#include <pdi.h>

namespace {

struct DDimX
{
};

struct DDimY
{
};

} // namespace

extern "C" {

void test_ddc_expose()
{
    // pdi_chunk_label_rank
    void* pdi_chunk_label_rank_ptr;
    ASSERT_EQ(PDI_access("pdi_chunk_label_rank", &pdi_chunk_label_rank_ptr, PDI_IN), PDI_OK);
    std::size_t const* const pdi_chunk_label_rank
            = static_cast<std::size_t*>(pdi_chunk_label_rank_ptr);
    EXPECT_EQ(*pdi_chunk_label_rank, 2);

    // pdi_chunk_label_extents
    void* pdi_chunk_label_extents_ptr;
    ASSERT_EQ(PDI_access("pdi_chunk_label_extents", &pdi_chunk_label_extents_ptr, PDI_IN), PDI_OK);
    std::size_t const* const pdi_chunk_label_extents
            = static_cast<std::size_t*>(pdi_chunk_label_extents_ptr);
    ASSERT_EQ(pdi_chunk_label_extents[0], 3);
    ASSERT_EQ(pdi_chunk_label_extents[1], 5);

    // pdi_chunk_label
    void* pdi_chunk_label_ptr;
    ASSERT_EQ(PDI_access("pdi_chunk_label", &pdi_chunk_label_ptr, PDI_IN), PDI_OK);
    int const* const pdi_chunk_label = static_cast<int*>(pdi_chunk_label_ptr);
    for (std::size_t i = 0; i < pdi_chunk_label_extents[0]; ++i) {
        for (std::size_t j = 0; j < pdi_chunk_label_extents[1]; ++j) {
            EXPECT_EQ(pdi_chunk_label[pdi_chunk_label_extents[1] * i + j], 3);
        }
    }

    EXPECT_EQ(PDI_reclaim("pdi_chunk_label_rank"), PDI_OK);
    EXPECT_EQ(PDI_reclaim("pdi_chunk_label_extents"), PDI_OK);
    EXPECT_EQ(PDI_reclaim("pdi_chunk_label"), PDI_OK);

    // nb_event_called
    void* nb_event_called_ptr;
    ASSERT_EQ(PDI_access("nb_event_called", &nb_event_called_ptr, PDI_INOUT), PDI_OK);
    int* const nb_event_called = static_cast<int*>(nb_event_called_ptr);
    *nb_event_called += 1;

    EXPECT_EQ(PDI_reclaim("nb_event_called"), PDI_OK);
}
}

TEST(Pdi, ChunkAndChunkSpan)
{
    std::string const pdi_cfg = R"PDI_CFG(
metadata:
  pdi_chunk_label_rank: size_t
  pdi_chunk_label_extents:
    type: array
    subtype: size_t
    size: $pdi_chunk_label_rank

data:
  pdi_chunk_label:
    type: array
    subtype: int
    size: [ '$pdi_chunk_label_extents[0]', '$pdi_chunk_label_extents[1]' ]
  nb_event_called: int

plugins:
  user_code:
    on_event:
      some_event:
        test_ddc_expose: {}
)PDI_CFG";

    PC_tree_t pdi_conf = PC_parse_string(pdi_cfg.c_str());
    PDI_init(pdi_conf);

    PDI_errhandler(PDI_NULL_HANDLER);

    {
        ddc::DiscreteDomain<DDimX> const ddom_x
                = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(3));
        ddc::DiscreteDomain<DDimY> const ddom_y
                = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimY>(5));
        ddc::DiscreteDomain<DDimX, DDimY> const ddom_xy(ddom_x, ddom_y);

        ddc::Chunk chunk("ddc_chunk_label", ddom_xy, ddc::HostAllocator<int>());
        ddc::parallel_fill(chunk, 3);

        int nb_event_called = 0;

        ddc::PdiEvent("some_event")
                .with("pdi_chunk_label", chunk)
                .with("nb_event_called", nb_event_called);

        ddc::PdiEvent("some_event")
                .with("pdi_chunk_label", chunk.span_view())
                .with("nb_event_called", nb_event_called);

        ddc::PdiEvent("some_event")
                .with("pdi_chunk_label", chunk.span_cview())
                .with("nb_event_called", nb_event_called);

        EXPECT_EQ(nb_event_called, 3);
    }

    PDI_finalize();
    PC_tree_destroy(&pdi_conf);
}
