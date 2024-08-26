# DDC concepts

<!--
Copyright (C) The ddc development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

DDC introduces labels in the form of dimensions and attributes on top of Kokkos arrays, which offer a more intuitive and less error-prone development experience.

Using Kokkos, the indices of views are weakly typed, meaning that each index is a simple integer. Let's consider a multidimensional view intended to represent a physical quantity, such as a temperature field for example. The first index represents the temperature along the $x$ axis, and the second index represents the temperature along the $y$ axis. As both indices are simple integers, they may be swapped by mistake. The code would compile successfully, but the result would be incorrect, and the source of the error would be difficult to trace.

DDC provides array-like containers that have labeled dimensions and strongly typed indices. For instance, by labeling dimensions `X` and `Y`, the indices along those labeled dimensions become strongly typed and cannot be swapped anymore.

\remark Note that the use of DDC is not restricted to solving equations. Indeed, one can easily imagine strong typing of variables corresponding to names or ages in a database. The operations must then be adapted accordingly. Here, we largely base our approach on the uniform and non-uniform resolution of the heat equation in two dimensions, which is why we focus the presentation on solving equations using finite differences on a 2D grid.

## ddc::Chunk and ddc::ChunkSpan

The `ddc::Chunk` is a container that holds the data, while `ddc::ChunkSpan` behaves like `std::mdspan`, meaning it is a pointer to the data contained within the `ddc::Chunk`.

Chunks contain data that can be accessed by unique identifiers called *discrete elements*. Usually, to access the data at a specific point of a 2D space, we would use two integers corresponding to the usual 'i' and 'j' indices. Instead, DDC uses the coordinate `ix` as a discrete element of the $x$ position (`ddc::DiscreteElement<DDimX>`), and the coordinate `iy` as a discrete element following the $y$ dimension (`ddc::DiscreteElement<DDimY>`). This is done after defining a strong typing for the discretized `X` dimension as `DDimX` and a strong typing for the discretized `Y` dimension as `DDimY` (see the heat equation example \subpage uniform_heat_equation "examples/uniform_heat_equation.cpp").

Note that swapping the `ddc::DiscreteElement<DDimX>` and `ddc::DiscreteElement<DDimY>` indices when calling the `ddc::ChunkSpan` does not affect the correctness of the code; the result remains the same.

## ddc::DiscreteElement, ddc::DiscreteVector and ddc::Coordinate

### ddc::DiscreteElement

Let's continue with our previous example of a 2D grid along two discretized dimensions `DDimX` and `DDimY`. In the previous section, we discussed how `ddc::DiscreteElement` is used as an index.

More precisely, a `ddc::DiscreteElement` is a type that carries the label of the dimension it is defined from.
Let's discretize the $y$ axis as follows: {$y_0$, $y_1$, ..., $y_n$}. Then the index `iy` defined as follows:

```cpp
ddc::DiscreteElement<DDimY> iy(0);
```

carries the strong typing of the ordinate dimension `DDimY`, and corresponds to the index 0 as in $y_0$, i.e. the index of the first point along the $y$ dimension.

`ddc::DiscreteElement` is used to access arrays. Let's take a multidimensional container and access its element `(i, j)`, which corresponds to the grid point at the $i$th row and the $j$th column. We would do it like this:

```cpp
container(i, j);
```

Now, if we take a slice of this container and still want to access the same element `(i, j)` from the grid, we need to adjust the indices, because they are relative to the first point of the slice. Using DDC and a slice (`ddc::ChunkSpan`), accessing with a `ddc::DiscreteElement` is the same between the slice and the original multidimensional array (`ddc::Chunk` or `ddc::ChunkSpan`), because of the uniqueness of each discrete element on the grid and because of the way we access data using DDC.


### ddc::DiscreteVector

A `ddc::DiscreteVector` corresponds to an integer that, like `ddc::DiscreteElement`, carries the label of the dimension in which it is defined. For instance in the uniform heat equation example, defining the `ddc::DiscreteVector` `gwx` as follows:

```cpp
ddc::DiscreteVector<DDimX> gwx(5);
```

This defines five point, along the $x$ dimension.

\remark Note that the difference between two `ddc::DiscreteElement` creates a `ddc::DiscreteVector`, and the sum of a `ddc::DiscreteVector` and a `ddc::DiscreteElement` results in a `ddc::DiscreteElement`. This illustrates how `ddc::DiscreteElement` could correspond to points in an affine space, while `ddc::DiscreteVector` could correspond to vectors in a vector space or to a distance between two points.

In summary;

+ `ddc::DiscreteElement` corresponds to each unique point of the mesh, fixed throughout the duration of the simulation. They are similar to fixed points in an affine space.
- `ddc::DiscreteVector` corresponds to a *number* of points along a particular axis or to a distance between two points (i.e. between two `ddc::DiscreteElement`s).

Both are integers that carry the strong typing of the dimension they are defined from.

## ddc::DiscreteDomain

As mentioned earlier, DDC operates on a coordinate system. These coordinates are part of a domain. Users must start their code by constructing a `ddc::DiscreteDomain` that contains each `ddc::DiscreteElement`. To construct a domain, you need to build a 1D domain along each direction. This is usually done when calling the function `ddc::init_discrete_space`.

## Algorithms in ddc ?

+ `for_each`: this algorithm allows iterating over each `ddc::DiscreteElement` in the domain of study.
+ `fill`: this algorithm allows filling a borrowed chunk with a given value.
+ `reduce`: this algorithm allows reducing operations on a chunk.
+ `copy`: this algorithm allows copying the content of a borrowed chunk into another
