// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>

#include <ddc/ddc.hpp>
#include <ddc/pdi.hpp>

#include <Kokkos_Core.hpp>
#include <paraconf.h>
#include <pdi.h>

struct DDimX
{
};

struct DDimY
{
};

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

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

plugins:
  decl_hdf5:
  - file: 'output.h5'
    on_event: some_event
    collision_policy: replace_and_warn
    write:
    - pdi_chunk_label_rank
    - pdi_chunk_label_extents
    - pdi_chunk_label
)PDI_CFG";

    PC_tree_t pdi_conf = PC_parse_string(pdi_cfg.c_str());
    PDI_init(pdi_conf);

    // Create a new scope to ensure the anonymous `ddc::PdiEvent` object
    // will be destroyed before calling `PDI_finalize`
    {
        ddc::DiscreteDomain<DDimX, DDimY> const
                ddom_xy(ddc::DiscreteElement<DDimX, DDimY>(7, 11),
                        ddc::DiscreteVector<DDimX, DDimY>(3, 5));

        ddc::Chunk chunk("ddc_chunk_label", ddom_xy, ddc::HostAllocator<int>());

        ddc::parallel_fill(chunk, 3);

        // Use the DDC API to expose a read-only 2D `ddc::ChunkSpan`.
        // It exposes three related variables:
        // - `pdi_chunk_label_rank`, the number of dimensions
        // - `pdi_chunk_label_extents`, the size of each dimension
        // - `pdi_chunk_label`, the pointer to the raw data
        // see `pdi_cfg` for the corresponding PDI types
        ddc::PdiEvent("some_event").with("pdi_chunk_label", chunk.span_cview());
    }

    PDI_finalize();
    PC_tree_destroy(&pdi_conf);
}
