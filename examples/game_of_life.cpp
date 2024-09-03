// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cmath>
#include <cstddef>
#include <iostream>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

using cell = bool;

// Name of the axis
struct DDimX
{
};
struct DDimY
{
};

void blinker_init(
        ddc::DiscreteDomain<DDimX, DDimY> const& domain,
        ddc::ChunkSpan<
                cell,
                ddc::DiscreteDomain<DDimX, DDimY>,
                std::experimental::layout_right,
                Kokkos::DefaultExecutionSpace::memory_space> cells)
{
    ddc::parallel_for_each(
            domain,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> const ixy) {
                ddc::DiscreteElement<DDimX> const ix(ixy);
                ddc::DiscreteElement<DDimY> const iy(ixy);
                if (iy == ddc::DiscreteElement<DDimY>(2)
                    && (ix >= ddc::DiscreteElement<DDimX>(1)
                        && ix <= ddc::DiscreteElement<DDimX>(3))) {
                    cells(ixy) = true;
                } else {
                    cells(ixy) = false;
                }
            });
}

std::ostream& display(
        std::ostream& os,
        ddc::ChunkSpan<
                cell,
                ddc::DiscreteDomain<DDimX, DDimY>,
                std::experimental::layout_stride> chunk)
{
    ddc::for_each(
            ddc::DiscreteDomain<DDimY>(chunk.domain()),
            [&](ddc::DiscreteElement<DDimY> const iy) {
                ddc::for_each(
                        ddc::DiscreteDomain<DDimX>(chunk.domain()),
                        [&](ddc::DiscreteElement<DDimX> const ix) {
                            os << (chunk(ix, iy) ? "*" : ".");
                        });
                os << "\n";
            });
    return os;
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    std::size_t const nt = 10;
    std::size_t const length = 5;
    std::size_t const height = 5;

    ddc::DiscreteDomain<DDimX, DDimY> const domain_xy(
            ddc::DiscreteElement<DDimX, DDimY>(0, 0),
            ddc::DiscreteVector<DDimX, DDimY>(length, height));

    ddc::DiscreteDomain<DDimX, DDimY> const inner_domain_xy
            = domain_xy
                      .remove(ddc::DiscreteVector<DDimX, DDimY>(1, 1),
                              ddc::DiscreteVector<DDimX, DDimY>(1, 1));

    ddc::Chunk cells_in_dev_alloc(
            "cells_in_dev",
            domain_xy,
            ddc::DeviceAllocator<cell>());
    ddc::Chunk cells_out_dev_alloc(
            "cells_out_dev",
            domain_xy,
            ddc::DeviceAllocator<cell>());
    ddc::Chunk cells_in_host_alloc
            = ddc::create_mirror(cells_in_dev_alloc.span_cview());

    // Initialize the whole domain
    blinker_init(domain_xy, cells_in_dev_alloc.span_view());

    ddc::parallel_deepcopy(
            cells_in_host_alloc,
            cells_in_dev_alloc.span_cview());
    display(std::cout, cells_in_host_alloc[inner_domain_xy]) << "\n";

    for (std::size_t iter = 0; iter < nt; ++iter) {
        ddc::ChunkSpan const cells_in = cells_in_dev_alloc.span_view();
        ddc::ChunkSpan const cells_out = cells_out_dev_alloc.span_view();

        ddc::parallel_for_each(
                inner_domain_xy,
                KOKKOS_LAMBDA(
                        ddc::DiscreteElement<DDimX, DDimY> const ixy) {
                    ddc::DiscreteElement<DDimX> const ix(ixy);
                    ddc::DiscreteElement<DDimY> const iy(ixy);
                    int alive_neighbors = cells_in(ixy) ? -1 : 0;
                    // Iterate on neighbors and increase the count of alive neighbors when necessary
                    for (int i = -1; i < 2; ++i) {
                        for (int j = -1; j < 2; ++j) {
                            if (cells_in(ix + i, iy + j)) {
                                ++alive_neighbors;
                            }
                        }
                    }

                    // Update the future status of the current cell depending on its current status and
                    // its current number of alive neighbors
                    cells_out(ixy) = cells_in(ixy);
                    if (cells_out(ixy)) {
                        if (alive_neighbors < 2 || alive_neighbors > 3) {
                            cells_out(ixy) = false;
                        }
                    } else {
                        if (alive_neighbors == 3) {
                            cells_out(ixy) = true;
                        }
                    }
                });

        ddc::parallel_deepcopy(cells_in_host_alloc, cells_out);
        display(std::cout, cells_in_host_alloc[inner_domain_xy]) << "\n";

        std::swap(cells_in_dev_alloc, cells_out_dev_alloc);
    }

    ddc::parallel_deepcopy(
            cells_in_host_alloc,
            cells_in_dev_alloc.span_cview());
    display(std::cout, cells_in_host_alloc[inner_domain_xy]) << "\n";

    return 0;
}
