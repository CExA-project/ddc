<!--
Copyright (C) The DDC development team, see below

SPDX-License-Identifier: MIT
-->

# DDC: The Discrete Domain Computation library

---
title: 'DDC: The Discrete Domain Computation library'
tags:

- C++
- HPC
- labelled arrays
- xarray

authors:

- name: Thomas Padioleau
  orcid: 0000-0001-5496-0013
  equal-contrib: true
  affiliation: 1

- name: Julien Bigot
  orcid: 0000-0002-0015-4304
  affiliation: 1

affiliations:

- name: Université Paris-Saclay, UVSQ, CNRS, CEA, Maison de la Simulation, 91191, Gif-sur-Yvette, France
  index: 1
date: 13 March 2025
bibliography: paper.bib

---

## Summary

## Statement of need

<!-- - problem of reinterpretation of indices -->
<!--
intro:

1. The md arrays are widely used in scientific computing oriented languages (Python and NumPy).
2. List of identified limitations
3. They recently were introduced in C++ so we take advantage of them to provide a solution of the previously listed problems.
 -->

The use of multidimensional arrays is widespread across various fields, particularly in scientific computing, where they serve as fundamental data containers. A primary motivation for their use is their potential to improve computational performance by leveraging problem-specific structure. For instance, when solving a partial differential equation that results in a stencil problem, computations typically achieve higher efficiency on a structured mesh compared to an unstructured mesh. This advantage primarily stems from a better usage of memory like predictable memory accesses and better cache utilization.

Many programming languages commonly used in scientific computing support multidimensional arrays in different ways. Fortran, a longstanding choice in the field, and Julia, a more recent language, both natively support these data structures. In contrast, the Python ecosystem relies on the popular NumPy library’s `numpy.Array`. Meanwhile, C++23 introduced `std::mdspan` to the standard library. This container was inspired by `Kokkos::View` from the Kokkos library which also serves as the foundation of DDC.

Despite their importance, multidimensional arrays introduce several practical challenges. In a sense, they encourage the usage of implicits in the source code. A frequent source of errors is the inadvertent swapping of indices when accessing elements. Such errors can be difficult to detect, especially given the common convention of using single-letter variable names like `i` and `j` for indexing. Another challenge arises in medium to large codebases because of raw multidimensional lacking semantics clarity in function signatures. When array dimensions carry specific meanings, this information is not explicitly represented in the source code, leaving it up to the user to ensure that dimensions are ordered correctly according to implicit expectations.

Along with this problem, when performing slices that remove dimensions of arrays, this operation changes the relative ordering of dimensions.

Solutions have been proposed in Python and Julia to address these issues. In Python, the Xarray library allows users to label dimensions that can then be used to perform computation. Following a similar approach, the "Discrete Domain Computation" (DDC) library aims to bring equivalent functionality to the C++ ecosystem. It uses a zero overhead abstraction approach, i.e. with labels fixed at compile-time, on top of different performant portable libraries, such as: Kokkos, Kokkos Kernels, Kokkos-fft and Ginkgo.

## DDC Core key features

<!-- Questions:
Shall we prefer the term "domain" to "set" -->
<!--
DDC core:
- Data structures
- Algorithms

out of scope of the paper:
- *discretization*, not well defined
- *performance*, we see DDC as a thin wrapper over existing performant portable libraries
-->

The DDC library is a C++ library designed for expressive and safe handling of multidimensional data. Its core component provides flexible data containers along with algorithms built on top of the performance portable Kokkos library.

### Containers

DDC offers two containers designed over the C++ 23 `std::mdspan` interface:

- `Chunk` an owning container, i.e. it manages the lifetime of the underlying memory allocation,
- `ChunkSpan` is a non-owning container view over existing memory allocation.

### Strongly-typed labelled indices

DDC employs strongly-typed multi-indices to label dimensions and access data. It introduces two types of indices to access the container's data:

- `DiscreteVector` indices:
  - strongly-typed labelled integers,
  - provide a multidimensional array access semantics,
  - always as fast access as raw multidimensional array,
- `DiscreteElement` indices:
  - strongly-typed labelled opaque identifiers/keys,
  - provide an associative access semantics, as keys in a map container,
  - potentially slower access, depending on the type of set of `DiscreteElement`.

Unlike `DiscreteVector` indices, users cannot directly interpret the internal representation of `DiscreteElement` and must reason about them based solely on their relative position.

### Algebra semantics

The relationship between `DiscreteVector` and `DiscreteElement` is analogous to vector and affine spaces in mathematics, more specifically:

- a `DiscreteVector` behaves like a vector in a vector space,
- a `DiscreteElement` behaves like a point in an affine space.

That is to say, if `v1` and `v2` are `DiscreteVector`, `e1` and `e2` are `DiscreteElement`, the following operations are valid:

- v1 + v2 returns a`DiscreteVector`,
- v2 - v1 returns a`DiscreteVector`,
- e2 - e1 returns a`DiscreteVector`,
- e1 + v1 and v1 + e1 return a `DiscreteElement`.

### Sets of `DiscreteElement`

The semantics of DDC containers is to associate data to a set of `DiscreteElement` indices. Let us note that the set of all possible `DiscreteElement` has a total order that is typically established once and for all at program initialisation. Thus to be able to construct a DDC container one must provide a multidimensional set of `DiscreteElement` indices, only these indices can be later used to access the container’s data.

The set of `DiscreteElement` is a customization point of the library. It takes the form of a Cartesian product of the different dimensions. DDC predefines the following sets:

- `DiscreteDomain`, a Cartesian product of intervals of `DiscreteElement` in each dimension,
- `StridedDiscreteDomain`, a Cartesian product of sets of `DiscreteElement` with a fixed step/stride in each dimension,
- `SparseDiscreteDomain`, a Cartesian product of explicitly ordered lists of `DiscreteElement` in each dimension.

The reason to introduce multiple sets is to mitigate the cost of containers access through `DiscreteElement` indices. Indeed they are used to retrieve the position of a given multi-index relatively to the front multi-index. Thus the cost of this operation depends on the information available in the set.

### Slicing

Like `std::mdspan`, DDC containers support slicing through the bracket operator that always returns a `ChunkSpan`. The key property is that the resulting slice’s set of `DiscreteElement` is a subset of the original set.

### Interoperability

Some interoperability with existing code is also supported:

- a `ChunkSpan` can be constructed from a raw pointer,
- conversely, a raw pointer can be retrieved from a `ChunkSpan`.

### Multidimensional algorithms

Finally, DDC offers multidimensional algorithms to manipulate the containers and associated `DiscreteElement` indices. Here is a list as of today:

- `parallel_deepcopy`, copies two `ChunkSpan` that are defined on the same set of `DiscreteElement`,
- `parallel_for_each`, applies a function once per `DiscreteElement` of a set of `DiscreteElement`,
- `parallel_fill`, fills a `ChunkSpan` with a value,
- `parallel_transform_reduce`, applies a function once per `DiscreteElement` of a set of `DiscreteElement` and combines the returned values with a reducer.

<!-- The labelling at compile-time then allows one to pass indices in any order to access container's data, for example `chunk_xyz(x, y, z)` would result in the same access as `chunk_xyz(z, x, y)`. -->

## Example

Let us illustrate some of the concepts introduced above with an example that stores the derivatives of the cosine function at a fixed value.

```cpp
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
```

## DDC extensions

Built on top of DDC core, we also provide optional extensions as in the Xarray ecosystem. As of today we provide three extensions: fft, pdi and splines. The fft and splines extensions work with dimensions of a specific form provided by DDC called `UniformPointSampling` and `NonUniformPointSampling`.

### DDC fft

This extension provides a thin wrapper on top of the Kokkos-fft library to provide labelled semantics of the discrete Fourier transform. The input array is expected to be defined over `UniformPointSampling` dimensions. The output of the transformation is an array where dimensions over `PeriodicPointSampling`.

### DDC splines

This extension provides a Spline transform either from `UniformPointSampling` or `NonUniformPointSampling` array dimensions.

### DDC pdi

PDI is a C data interface library allowing loose coupling with external libraries through PDI plugins like HDF5, NetCDF, Catalyst... This extension eases the metadata serialisation of the DDC containers. Instead of manually expose, sizes, strides and the pointer of the underlying array, the user can directly expose the DDC container.

## Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:

- `@author:2001` ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

## Acknowledgements

This work has received support by the CExA Moonshot project of the CEA [cexa-project](https://cexa-project.org). We acknowledge contributions from the Maison de la Simulation. We also thank the developers and contributors of the DDC project for their efforts in making numerical modeling more accessible and efficient.

## References

- Pandas: 1D and 2D labelled arrays
- XArray: multi-dimensional arrays
- [equivalents](https://docs.xarray.dev/en/latest/ecosystem.html#non-python-projects)
- Kokkos
- Kokkos-fft
- Kokkos Kernels
- Ginkgo
- PDI
