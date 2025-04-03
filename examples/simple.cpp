// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

// Define two independent dimensions
struct Dim1
{
};

struct Dim2
{
};

int main()
{
    // Initialize Kokkos runtime (automatically finalized at the end of scope)
    Kokkos::ScopeGuard const kokkos_scope;

    // Initialize DDC runtime (automatically finalized at the end of scope)
    ddc::ScopeGuard const ddc_scope;

    // Create a discrete domain for Dim1 with 5 elements
    ddc::DiscreteDomain<Dim1> const dom1
            = ddc::init_trivial_space<Dim1>(ddc::DiscreteVector<Dim1>(5));

    // Create a discrete domain for Dim2 with 7 elements
    ddc::DiscreteDomain<Dim2> const dom2
            = ddc::init_trivial_space<Dim2>(ddc::DiscreteVector<Dim2>(7));

    // Define a 2D discrete domain combining Dim1 and Dim2
    ddc::DiscreteDomain<Dim1, Dim2> const dom(dom1, dom2);

    // Create a 2D array (Chunk) to store integer values on the combined domain
    ddc::Chunk my_array("my_array", dom, ddc::HostAllocator<int>());

    // Iterate over the first dimension (Dim1)
    for (ddc::DiscreteElement<Dim1> idx1 : dom1) {
        // Iterate over the second dimension (Dim2)
        for (ddc::DiscreteElement<Dim2> idx2 : dom2) {
            // Assign the value 1 to each element in the 2D array
            my_array(idx1, idx2) = 1;

            // The following would not compile as my_array expects a DiscreteElement over Dim2
            // my_array(idx1, idx1) = 1;
        }
    }

    return 0;
}
