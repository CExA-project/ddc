// SPDX-License-Identifier: MIT

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

struct DDimX;
struct DDimY;

int main(int argc, char** argv)
{
    ddc::ScopeGuard scope(argc, argv);

    ddc::DiscreteDomain<DDimX> x_domain(
            ddc::DiscreteElement<DDimX>(0),
            ddc::DiscreteVector<DDimX>(10));
    ddc::DiscreteDomain<DDimY> y_domain(
            ddc::DiscreteElement<DDimY>(0),
            ddc::DiscreteVector<DDimY>(100));

    ddc::Chunk
            chunk(ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain),
                  ddc::DeviceAllocator<int>());
    ddc::ChunkSpan chunkspan = chunk.span_view();

    ddc::fill(chunkspan, 10);
    ddc::for_each(
            ddc::policies::parallel_device,
            x_domain,
            DDC_LAMBDA(ddc::DiscreteElement<DDimX> const ix) {
                printf("----- DEBUG LOG -----");
                // auto subview = std::experimental::submdspan(
                //         chunkspan.allocation_mdspan(),
                //         (ix - x_domain.front()).value(),
                //         std::experimental::full_extent);

                // ddc::ChunkSpan<int, ddc::DiscreteDomain<DDimX>, std::experimental::layout_right> slice(subview, x_domain);

                auto slice = chunkspan[ix];
                printf("size = %i", int(slice.size()));
            });
}
