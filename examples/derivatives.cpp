// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

// Include the DDC library for discrete domain management
#include <ddc/ddc.hpp>

// Include Kokkos for parallel computing
#include <Kokkos_Core.hpp>

// Define a struct to represent the dimension for derivative order
struct DimDerivOrder
{
};

int main()
{
    // Initialize Kokkos runtime (automatically finalized at the end of scope)
    Kokkos::ScopeGuard const kokkos_scope;

    // Initialize DDC runtime (automatically finalized at the end of scope)
    ddc::ScopeGuard const ddc_scope;

    // Create a discrete domain of derivative orders with 100 elements
    ddc::DiscreteDomain<DimDerivOrder> const all_orders
            = ddc::init_trivial_space<DimDerivOrder>(ddc::DiscreteVector<DimDerivOrder>(100));

    // Define a lambda function to compute the order index relative to the first element
    auto const order = [order_0_idx
                        = all_orders.front()](ddc::DiscreteElement<DimDerivOrder> order_id) -> int {
        return order_id - order_0_idx;
    };

    // Create a chunk (array) to store the computed cosine derivatives
    // The domain is adjusted to exclude the 0th order (removing first element)
    ddc::Chunk cosine_derivatives(
            "cosine_derivatives", // Name for debugging/tracking
            all_orders.remove_first(ddc::DiscreteVector<DimDerivOrder>(1)),
            ddc::HostAllocator<double>()); // Allocate in host memory

    // Define a fixed value x = 2π/3
    double const x = 2 * Kokkos::numbers::pi / 3;

    // Loop over the domain to compute and store cosine derivatives
    for (ddc::DiscreteElement<DimDerivOrder> order_idx : cosine_derivatives.domain()) {
        // Compute the cosine derivative using the pattern:
        // cos(x + order * π/2): follows the derivative cycle of cosine
        cosine_derivatives(order_idx) = Kokkos::cos(x + order(order_idx) * Kokkos::numbers::pi / 2);
    }

    return 0;
}
