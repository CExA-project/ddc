# DDC concepts

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

DDC introduces labels in the form of dimensions, and attributes on top of Kokkos views, which allows for a more intuitive, and less error-prone developer experience.

In fact, in Kokkos, the indices of views are weakly typed, meaning that each index is a simple integer. Let's consider a multidimensional view intended to represent a physical quantity, such as velocity in our example. The first index represents the velocity along the X-axis, and the second index represents the velocity along the Y-axis. In reality, there is nothing to distinguish between these two indices from the computer's perspective; they are both simply integers. This can lead to a situation where one mistakenly swaps the two indices. The code would compile successfully, but the resulting behavior would be incorrect, and the source of the error could be difficult to trace.

The advantage of using DDC is that it provides array-like containers that have labeled dimensions using strongly typed indices. For instance, by labeling dimensions by `X` and `Y`, the indices along those labeled dimensions become strongly typed preventing the user from making such mistakes. 

> Note that the use of DDC is not restricted to solving equations. Indeed, one can easily imagine strong typing of variables corresponding to names or ages in a registry. The operations must then be adapted accordingly. Here, we largely base our approach on the uniform and non-uniform resolution of the heat equation in two dimensions, which is why we focus the presentation on solving equations using finite differences on a 2D grid.

## DDC::Chunk and DDC::ChunkSpan

The `ddc::chunk` is a container that holds the data, while `ddc::chunkspan` behaves like `mdspan`, meaning they are pointers to the data contained within the chunk.

As mentioned in the introduction, ddc is a library that offers strongly typed indexing. Chunks contain data that can be located in space by coordinates. Thus, to access the data at a specific point in a 2D space, instead of entering two integers corresponding to the x and y positions, ddc requires entering the  coordinate `x` as a discrete element of the x position: `ddc::DiscreteElement<DDimX>`, and entering the `y` coordinate as a discrete element following the y dimension: `ddc::DiscreteElement<DDimY>`. This is done after predefining a strong typing for the discretized `X` dimension as `DDimX` and a strong typing for the discretized `Y` dimension as `DDimY` (see the heat equation example \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp"). 

Note that swapping the `ddc::DiscreteElement<DDimX>` and `ddc::DiscreteElement<DDimY>` indices when calling the chunkspan does not affect the correctness of the code; the result remains the same.

## DDC::DiscreteElement and DDC::DiscreteVector

### DDC::DiscreteElement
Let's continue with our previous example of a 2D grid labeled along two discretized dimensions labeled as `DDimX` and `DDimY`. In the previous paragraph, we discussed how `DDC::DiscreteElement` could be used as indices to have access to a physical value at a precised point on the 2D grid. 

More precisely, a DDC::DiscreteElement` is a C++ variable that carries the strong type of the dimension in which it is defined. Let's return to our example by defining a variable `y` as follows:

```cpp
DDC::DiscreteElement<DDimY> y(0);
```

The variable `y` carries the strong typing of the ordinate dimension DDimY and corresponds to the first point along the `Y` dimension.

Moreover, `DDC::DiscreteElement` are very useful for another reason. If we take the example of a classic container in C++, let's say we want to access the element `(i,j)` of this container, we would do it like this:

```cpp
container(i,j);
```

Now, if we take a slice of this container and still want to access the same element `(i,j)` from the grid, we will need to adjust the indices because the indexing of the new sliced container along each dimension starts at 0. However, with DDC, this is not the case. If we take a slice of a chunkspan, accessing a `DDC::DiscreteElement` is the same between the slice and the original chunkspan because of the uniqueness of each discrete element on the grid and because of the way we access data using DDC.

### DDC::DiscreteVector

A `DDC::DiscreteVector` corresponds to an integer that, like `DDC::DiscreteElement`, carries the strong typing of the dimension in which it is defined. For instance in the uniform heat equation example, defining the `DDC::DiscreteVector` `gwx` as follows: 

```cpp 
DDC::DiscreteVector<DDimX> gwx(1);
```

is equivalent to defining a number of points, here 1, along the `x` dimension.

> Note that the difference between two `DDC::DiscreteElement` creates a `DDC::DiscreteVector`, and the sum of a `DDC::DiscreteVector` and a `DDC::DiscreteElement` results in a `DDC::DiscreteElement`. This illustrates how `DDC::DiscreteElement` could correspond to points in an affine space, while `DDC::DiscreteVector` could correspond to vectors in a vector space or to a distance between two points.

In summary; 

+ `DDC::DiscreteElement` corresponds to each unique point of the mesh, fixed throughout the duration of the simulation and defined according to the discretization that has been done (uniform or non-uniform: see the examples of the heat equation). They are similar to fixed points in an affine space. 
+ On the other hand, `DDC::DiscreteVector` corresponds to a number of points along a particular axis or to a distance between two points; in DDC, it corresponds to a distance between two DiscreteElements. These are integers that carry the strong typing of the dimension in which they are defined.

## ddc::DiscreteDomain

As mentioned earlier, DDC operates on a coordinate system. These coordinates are part of a domain. Users must start their code by constructing a `ddc::DiscreteDomain` that contains each `ddc::DiscreteElement`. To construct a domain, you need to build a 1D domain along each direction. This can be done using the function `ddc::init_discrete_space`. 

Note that it is possible to construct a domain with both uniform (see \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp") and non-uniform (see \subpage non_uniform_heat_equation "examples/non_uniform_heat_equation.cpp") distribution of points.

## How to iterate in DDC ?
