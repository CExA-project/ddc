# DDC concepts

<!--
Copyright (C) The ddc development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

DDC introduces labels in the form of dimensions and attributes on top of Kokkos arrays, which offer a more intuitive and less error-prone development experience.

Using Kokkos, the indices of views are weakly typed, meaning that each index is a simple integer. Let's consider a multidimensional view intended to represent a physical quantity, such as a temperature field for example. The first index represents the temperature along the $x$ axis, and the second index represents the temperature along the $y$ axis. As both indices are simple integers, they may be swapped by mistake. The code would compile successfully, but the result would be incorrect, and the source of the error would be difficult to trace.

DDC provides array-like containers that have labeled dimensions and strongly typed indices. For instance, by labeling dimensions `X` and `Y`, the indices along those labeled dimensions become strongly typed and cannot be swapped anymore.

## ddc::DiscreteElement and ddc::DiscreteVector

### ddc::DiscreteElement

A `ddc::DiscreteElement` is a type that carries the label of the dimension it is defined from.
Let's discretize the $y$ axis as follows: {$y_0$, $y_1$, ..., $y_n$}. Then the index `iy` defined as follows:

```cpp
ddc::DiscreteElement<DDimY> iy(0);
```

carries the strong typing of the ordinate dimension `DDimY`, and corresponds to the index 0 as in $y_0$, i.e. the index of the first point along the $y$ dimension.

Another example would be the study of a charged plasma. 
One dimension could represent species with charge and mass information, and another could represent particle momentum. 

Here, `ddc::DiscreteElement<Species_dimension> ispecies(0)` would correspond to a unique tuple of mass and charge for a particle in the model. 
Same would apply for the momentum dimension.

`ddc::DiscreteElement` is used to access arrays. 
Let's take a multidimensional container and access its element `(i, j)`, which corresponds to the grid point at the $i$th row and the $j$th column. We would do it like this:

```cpp
container(i, j);
```

Now, if we take a slice of this container and still want to access the same element `(i, j)` from the grid, we need to adjust the indices, because they are relative to the first point of the slice. Using DDC and a slice  accessing with a `ddc::DiscreteElement` is the same between the slice and the original multidimensional array because of the uniqueness of each discrete element on the grid and because of the way we access data using DDC.

### ddc::DiscreteVector

A `ddc::DiscreteVector` corresponds to an integer that, like `ddc::DiscreteElement`, carries the label of the dimension in which it is defined. For instance in the uniform heat equation example, defining the `ddc::DiscreteVector` `gwx` as follows:

```cpp
ddc::DiscreteVector<DDimX> gwx(5);
```

defines five point, along the $x$ dimension.

\remark Note that the difference between two `ddc::DiscreteElement`s gives a `ddc::DiscreteVector`, and the sum of a `ddc::DiscreteVector` and a `ddc::DiscreteElement` gives a `ddc::DiscreteElement`. This illustrates how `ddc::DiscreteElement`s correspond to points in an affine space, while `ddc::DiscreteVector`s correspond to vectors in a vector space, or to a distance between two points.

In summary:

- `ddc::DiscreteElement` corresponds to unique points in the simulation labeled by a dimension. They can for instance represent discrete points along a dimension or be linked to particles, allowing access to certain data (e.g., mass, charge, etc.).
- `ddc::DiscreteVector` corresponds to a *number* of points along a labeled dimensions.

## ddc::DiscreteDomain

`ddc::DiscreteDomain` is an interval of [unique identifiers](#ddcdiscreteelement) distributed according to each labeled dimension.
It allows for the distribution of [unique identifiers](#ddcdiscreteelement) across each previously defined labeled dimension. 
It is constructed using the method `ddc::init_discrete_space`. 

## ddc::Chunk and ddc::ChunkSpan

The `ddc::Chunk` is a container that holds the data, while `ddc::ChunkSpan` behaves like `std::mdspan`, meaning it is a pointer to the data contained within the `ddc::Chunk`.

Chunks contain data that can be accessed by [unique identifiers](#ddcdiscreteelement). Usually, to access the data at a specific point of a 2D space, we would use two integers corresponding to the usual 'i' and 'j' indices. Instead, DDC uses the coordinate `ix` as a discrete element of the $x$ position (`ddc::DiscreteElement<DDimX>`), and the coordinate `iy` as a discrete element following the $DDim$ dimension (`ddc::DiscreteElement<DDim>`). This is done after defining a strong typing for each dimension as `DDim`.

Note that swapping the `ddc::DiscreteElement<DDim1>` and `ddc::DiscreteElement<DDim2>` indices when calling the `ddc::ChunkSpan` does not affect the correctness of the code; the result remains the same: 

```cpp
ddc::Chunkspan(ddc::DiscreteElement<DDim1>,ddc::DiscreteElement<DDim2> ) == ddc::Chunkspan(ddc::DiscreteElement<DDim2>,ddc::DiscreteElement<DDim1> );
```

## Algorithms in ddc ?

+ `for_each`: this algorithm allows iterating over each `ddc::DiscreteElement` in the domain of study.
+ `fill`: this algorithm allows filling a borrowed chunk with a given value.
+ `reduce`: this algorithm allows reducing operations on a chunk.
+ `copy`: this algorithm allows copying the content of a borrowed chunk into another
