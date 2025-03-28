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

Despite their importance, multidimensional arrays introduce several practical challenges. In a sense, they encourage the usage of implicits in the source code. A frequent source of errors is the inadvertent swapping of indices when accessing elements. Such errors can be difficult to detect, especially given the common convention of using single-letter variable names like `i` and `j` for indexing. Another challenge arises in medium to large codebases because of raw multidimensional lacking semantic clarity in function signatures. When array dimensions carry specific meanings, this information is not explicitly represented in the source code, leaving it up to the user to ensure that dimensions are ordered correctly according to implicit expectations.

Along with this problem, when performing slices that remove dimensions of arrays, this operation changes the relative ordering of dimensions.

Solutions have been proposed in Python and Julia to address these issues. In Python, the Xarray library allows users to label dimensions that can then be used to perform computation. Following a similar approach, the "Discrete Domain Computation" (DDC) library aims to bring equivalent functionality to the C++ ecosystem using a zero overhead abstraction approach, i.e. with labels fixed at compile-time.

## DDC Core key features

<!--
DDC core:
- Data structures
- Algorithms
-->

The DDC library is a C++ library designed for safety handling of multidimensional data. Its core components, `Chunk` and `ChunkSpan`, provide flexible data containers along with algorithms for efficient manipulation built on top of the performance portable Kokkos library.

DDC offers two containers `Chunk` and `ChunkSpan`. They are designed over the C++ 23 `std::mdspan` interface. The former one, `Chunk`, is owning the data whereas the latter one, `ChunkSpan`, is a view on the data.

DDC employs strongly-typed indices to label dimensions and access data. It introduces two types of indices:

- `DiscreteVector` indices:
  - these are strongly-typed labelled integers,
  - they provide a multidimensional array access semantic to the container's data,
- `DiscreteElement` indices:
  - these are strongly-typed labelled opaque identifiers.
  - they provide an associative access semantic to the container's data.

Unlike `DiscreteVector` indices, users cannot directly interpret the internal representation of `DiscreteElement` and must reason about them based solely on their relative ordering. The set of possible `DiscreteElement` is typically established at the initialisation of the program.

These two sets of indices are linked together by a similar semantic to vector and affine spaces in mathematics. As the names suggest, a `DiscreteVector` behaves as a vector in a vector space and a `DiscreteElement` behaves as a point in an affine space. That is to say, if `v1` and `v2` are `DiscreteVector`, `e1` and `e2` are `DiscreteElement`, the following operations are valid:

- v1 + v2 returns a`DiscreteVector`,
- v2 - v1 returns a`DiscreteVector`,
- e1 + v1 = v1 + e1 is `DiscreteElement`,
- e2 - e1 returns a`DiscreteVector`.

When constructing a DDC container, a set of `DiscreteElement` indices must be provided. Only the `DiscreteElement` from this predefined set can be used to access the container’s data, ensuring strict control over indexing. The set of `DiscreteElement` is a customization point but has always the form of a Cartesian product of the different dimensions. DDC predefines the following sets:

- `DiscreteDomain`, a contiguous set of `DiscreteElement`,
- `StridedDiscreteDomain`, a set of `DiscreteElement` with a fixed step/stride in each dimension,
- `SparseDiscreteDomain`, an explicit ordered list of `DiscreteElement` in each dimension.

As for `std::mdspan`, DDC containers define slicing operators in the form of the bracket operator that always return a `ChunkSpan`. One key point is that the set of `DiscreteElement` of the slice result is a subset of the initial set of `DiscreteElement`.

Finally, DDC offers multidimensional algorithms to manipulate the containers and the `DiscreteElement`. Here is a list as of today:

- `parallel_for_each`
- `parallel_fill`
- `parallel_deepcopy`
- `parallel_transform_reduce`

We emphasize that some interoperability is possible with existing code. It is possible to construct a `ChunkSpan` from a raw pointer or on the contrary to retrieve the pointer from a `ChunkSpan`.

## Example

Let us illustrate the concepts introduced above with the example of the game of life. We start by identifying the dimensions of the problem: two axes X and Y that will be used to locate the cells. In DDC this is corresponds to create two `struct` that will be used as labels for the two dimensions. We call them `DDimX` and `DDimY` respectively

```cpp
struct DDimX {};
struct DDimY {};
```

In order to store the state of each cell, alive or dead, we need to create the set of `DiscreteElement` that will uniquely identify each cell. This is achieved by calling the function `init_trivial_space`

```cpp
DiscreteDomain<DDimX> domain_x = init_trivial_space(DiscreteVector<DDimX>(nx));
DiscreteDomain<DDimY> domain_y = init_trivial_space(DiscreteVector<DDimY>(ny));
DiscreteDomain<DDimX, DDimY> domain_xy(domain_x, domain_y);
```

This step also sets up the relative ordering of each element. The last line creates a Cartesian product of the two sets.
<!-- It is also responsible of initialising static attributes if present. -->

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

This work has received support by the CExA Moonshot project of the CEA (https://cexa-project.org). We acknowledge contributions from the Maison de la Simulation. We also thank the developers and contributors of the DDC project for their efforts in making numerical modeling more accessible and efficient.

## References

- Pandas: 1D and 2D labelled arrays
- XArray: multi-dimensional arrays
- [equivalents](https://docs.xarray.dev/en/latest/ecosystem.html#non-python-projects)
- Kokkos
- Kokkos-fft
- Kokkos Kernels
- Ginkgo
- PDI
