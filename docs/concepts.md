# DDC concepts

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

DDC introduces labels in the form of dimensions, coordinates, and attributes on top of Kokkos views, which allows for a more intuitive, more concise, and less error-prone developer experience.

In fact, in Kokkos, the indices of views are weakly typed, meaning that each index is a simple integer. Let's consider a multidimensional view intended to represent a physical quantity, such as velocity in our example. The first index represents the velocity along the x-axis, and the second index represents the velocity along the y-axis. In reality, there is nothing to distinguish between these two indices from the computer's perspective; they are both simply integers. This can lead to a situation where the user of the code mistakenly swaps the two indices. The code would compile successfully, but the resulting behavior would be incorrect, and the source of the error could be difficult to trace.

The advantage of using a DDC is that it provides chunk and chunkspan that have strongly typed indices. The indices are of type 'X' or 'Y' depending on the dimension. This strong typing prevents the user from making such mistakes, as the type system will enforce correct usage of the indices, leading to safer and more reliable code.

## ddc::Chunk and ddc::ChunkSpan

The `ddc::chunk` is a container that holds the data, while `ddc::chunkspan` behaves like `mdspan` and `kokkos::view`, meaning they are pointers to the data contained within the chunk. Similarly to `kokkos::view`, `ddc::chunkspan` includes reference counting to free allocated memory when the counter reaches zero.

As mentioned in the introduction, ddc is a library that offers strongly typed indexing. Chunks contain data that can be located in space by coordinates. Thus, to access the data at a specific point in a 2D space, instead of entering two integers corresponding to the x and y positions, ddc requires entering the  coordinate `x` as a discrete element of the x position: `ddc::DiscreteElement<DDimX>`, and entering the `y` coordinate as a discrete element following the y dimension: `ddc::DiscreteElement<DDimY>`. This is done after predefining a strong typing for the x dimension as DDimX and a strong typing for the y dimension as DDimY (see the heat equation example \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp"). 

Note that swapping the `ddc::DiscreteElement<DDimX>` and `ddc::DiscreteElement<DDimY>` indices when calling the chunkspan does not affect the correctness of the code; the result remains the same.

## ddc::DiscreteDomain

As mentioned earlier, DDC operates on a coordinate system. These coordinates are part of a domain. Users must start their code by constructing a `ddc::DiscreteDomain` that contains each `ddc::DiscreteElement`. To construct a domain, you need to build a 1D domain along each direction. This can be done using the function `ddc::init_discrete_space`. 

Note that it is possible to construct a domain with both uniform (see \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp") and non-uniform (see \subpage non_uniform_heat_equation "examples/non_uniform_heat_equation.cpp") distribution of points.

