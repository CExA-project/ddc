# ddc concepts

<!--
Copyright (C) The ddc development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

DDC introduces labels in the form of dimensions and attributes on top of Kokkos arrays, which allows for a more intuitive and less error-prone developer experience.

In fact, in Kokkos, the indices of views are weakly typed, meaning that each index is a simple integer. Let's consider a multidimensional view intended to represent a physical quantity, such as a velocity field for example. The first index represents the velocity along the X-axis, and the second index represents the velocity along the Y-axis. As both are simply integers this can lead to a situation where one mistakenly swaps the two indices. The code would compile successfully, but the resulting behavior would be incorrect, and the source of the error could be difficult to trace.

The advantage of using ddc is that it provides array-like containers that have labeled dimensions using strongly typed indices. For instance, by labeling dimensions by `X` and `Y`, the indices along those labeled dimensions become strongly typed preventing the user from making such mistakes.

> Note that the use of ddc is not restricted to solving equations. Indeed, one can easily imagine strong typing of variables corresponding to names or ages in a registry. The operations must then be adapted accordingly. Here, we largely base our approach on the uniform and non-uniform resolution of the heat equation in two dimensions, which is why we focus the presentation on solving equations using finite differences on a 2D grid.

## ddc::Chunk and ddc::ChunkSpan

The `ddc::chunk` is a container that holds the data, while `ddc::chunkspan` behaves like `mdspan`, meaning they are pointers to the data contained within the chunk.

As mentioned in the introduction, ddc is a library that offers strongly typed indexing. Chunks contain data that can be located in space by coordinates. Thus, to access the data at a specific point in a 2D space, instead of entering two integers corresponding to the x and y positions, ddc requires entering the  coordinate `x` as a discrete element of the x position: `ddc::DiscreteElement<DDimX>`, and entering the `y` coordinate as a discrete element following the y dimension: `ddc::DiscreteElement<DDimY>`. This is done after predefining a strong typing for the discretized `X` dimension as `DDimX` and a strong typing for the discretized `Y` dimension as `DDimY` (see the heat equation example \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp").

Note that swapping the `ddc::DiscreteElement<DDimX>` and `ddc::DiscreteElement<DDimY>` indices when calling the chunkspan does not affect the correctness of the code; the result remains the same.

## ddc::DiscreteElement, ddc::DiscreteVector and ddc::Coordinate

### ddc::DiscreteElement

Let's continue with our previous example of a 2D grid along two discretized dimensions labeled as `DDimX` and `DDimY`. In the previous paragraph, we discussed how `ddc::DiscreteElement` could be used as indices to have access to a physical value at a precised point on the 2D grid.

More precisely, a `ddc::DiscreteElement` is a C++ variable that carries the strong type of the dimension in which it is defined.
Let's return to our example, If we take a discretization of the Y-axis as follows: {y<sub>0</sub>, y<sub>1</sub>, ..., y<sub>n</sub>}, the variable `y` defined as follows:

```cpp
ddc::DiscreteElement<DDimY> y(0);
```

carries the strong typing of the ordinate dimension `DDimY` and corresponds to y<sub>0</sub> , the first point along the `Y` dimension.

Moreover, `ddc::DiscreteElement` are very useful for another reason. If we take the example of a classic container in C++, let's say we want to access the element `(i,j)` of this container which corresponds to the grid point at the i th row and j th column, we would do it like this:

```cpp
container(i,j);
```

Now, if we take a slice of this container and still want to access the same element `(i,j)` from the grid, we will need to adjust the indices because the indexing of the new sliced container along each dimension starts at zero. However, with ddc, this is not the case. If we take a slice of a chunkspan, accessing a `ddc::DiscreteElement` is the same between the slice and the original chunkspan because of the uniqueness of each discrete element on the grid and because of the way we access data using ddc.

### ddc::Coordinate

A `ddc::Coordinate` is one of the only ddc types which represents a continuous data type. This means that it can take any value that can be represented by a double. It represents the position of a coordinate in the vector space.

Let's take the same discretization of the Y-axis {y<sub>0</sub>, y<sub>1</sub>, ..., y<sub>n</sub>}. We saw that ```cpp ddc::DiscreteElement<DDimY> y(0)``` points to the first element y<sub>0</sub> along the Y-axis. Now if we take ```cpp ddc::Coordinate<DDimY> (ddc::DiscreteElement<DDimY> y(0))``` it will correspond to the actual value of y<sub>0</sub> in the simulation. If the first point is at a position `y = -1`, it will be characterized by a `ddc::DiscreteElement` as the first point of the discretization, but physically its position is -1 and this is the value that `ddc::Coordinate` of this element will be.

The `ddc::Coordinate` thus bridges the gap between the discretization accessible by `ddc::DiscreteElement` and the actual physical value of the coordinate.

### ddc::DiscreteVector

A `ddc::DiscreteVector` corresponds to an integer that, like `ddc::DiscreteElement`, carries the strong typing of the dimension in which it is defined. For instance in the uniform heat equation example, defining the `ddc::DiscreteVector` `gwx` as follows:

```cpp
ddc::DiscreteVector<DDimX> gwx(1);
```

is equivalent to defining a number of points, here 1, along the `x` dimension.

> Note that the difference between two `ddc::DiscreteElement` creates a `ddc::DiscreteVector`, and the sum of a `ddc::DiscreteVector` and a `ddc::DiscreteElement` results in a `ddc::DiscreteElement`. This illustrates how `ddc::DiscreteElement` could correspond to points in an affine space, while `ddc::DiscreteVector` could correspond to vectors in a vector space or to a distance between two points.

In summary;

+ `ddc::DiscreteElement` corresponds to each unique point of the mesh, fixed throughout the duration of the simulation. They are similar to fixed points in an affine space.
+ `ddc::Coordinate` represents the actual value of the discretized points on the grid indexed with the `ddc::DiscreteElement`
+ On the other hand, `ddc::DiscreteVector` corresponds to a number of points along a particular axis or to a distance between two points; in ddc, it corresponds to a distance between two DiscreteElements. These are integers that carry the strong typing of the dimension in which they are defined.

## ddc::DiscreteDomain

As mentioned earlier, ddc operates on a coordinate system. These coordinates are part of a domain. Users must start their code by constructing a `ddc::DiscreteDomain` that contains each `ddc::DiscreteElement`. To construct a domain, you need to build a 1D domain along each direction. This can be done using the function `ddc::init_discrete_space`.

Note that it is possible to construct a domain with both uniform (see \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp") and non-uniform (see \subpage non_uniform_heat_equation "examples/non_uniform_heat_equation.cpp") distribution of points.

## Algorithms in ddc ?

+ `for_each`: this algorithm allows iterating over each DiscreteElement in the domain of study.
+ `fill`: this algorithm allows filling a borrowed chunk with a given value.
+ `reduce`: this algorithm allows reducing operations on a chunk.
+ `copy`: this algorithm allows copying the content of a borrowed chunk into another
