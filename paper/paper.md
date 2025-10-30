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
- name: Emily Bourne
  orcid: 0000-0002-3469-2338
  affiliation: 2
affiliations:
- name: Université Paris-Saclay, UVSQ, CNRS, CEA, Maison de la Simulation, 91191, Gif-sur-Yvette, France
  index: 1
- name: SCITAS, EPFL, CH-1015 Lausanne, Switzerland
  index: 2
date: 03 April 2025
bibliography: paper.bib
---

## Summary

The Discrete Domain Computation (DDC) library is a C++ library designed to provide high-performance, strongly-typed labelled multidimensional arrays. Inspired by Python's Xarray and built on top of performance-portable libraries like Kokkos, DDC enables expressive, safe, and efficient numerical computations. It provides a coherent ecosystem to work with labelled dimensions from data structures to algorithms. Additionally, DDC extends functionality through modules such as FFT (based on kokkos-fft), splines, and with a bridge to the PDI library. The library is actively used to modernize legacy scientific codes, such as the Fortran-based Gysela plasma simulation code [@grandgirard2016].

## Statement of need

The use of multidimensional arrays is widespread across various fields, particularly in scientific computing, where they serve as fundamental data containers. A primary motivation for their use is their potential to improve computational performance by leveraging problem-specific structures. For instance, when solving a partial differential equation that results in a stencil problem, computations typically achieve higher efficiency on a structured mesh compared to an unstructured mesh. This advantage primarily stems from a better usage of memory, with predictable accesses, and better cache utilization.

Many programming languages commonly used in scientific computing support multidimensional arrays in different ways. Fortran, a longstanding choice in the field, and Julia, a more recent language, both natively support these data structures. In contrast, the Python ecosystem relies on the popular NumPy library’s `numpy.Array` [@harris2020array]. Meanwhile, C++23 introduced `std::mdspan` to the standard library. This container was inspired by `Kokkos::View` from the Kokkos library which also serves as the foundation of DDC.

Despite their importance, multidimensional arrays introduce several practical challenges. In a sense, they encourage the usage of implicit information in the source code. A frequent source of errors is the inadvertent swapping of indices when accessing elements. Such errors can be difficult to detect, especially given the common convention of using single-letter variable names like `i` and `j` for indexing. Another challenge in medium-to-large codebases is the lack of semantic clarity in function signatures when using raw multidimensional arrays. When array dimensions carry specific meanings, this information is not explicitly represented in the source code, leaving it up to the user to ensure that dimensions are ordered correctly according to implicit expectations. For example it is quite usual to use the same index for multiple interpretations: looping over mesh cells identified by `i` and interpreting `i+1` as the face to the right. Another example is slicing that removes dimensions, this can shift the positions of remaining dimensions, altering the correspondence between axis indices and their semantic meanings.

Solutions have been proposed to address these issues. For example, in Python, the Xarray [@hoyer2017xarray] library allows users to label dimensions that can then be used to perform computations. Following a similar approach, the "Discrete Domain Computation" (DDC) library aims to bring equivalent functionality to the C++ ecosystem. It uses a zero overhead abstraction approach, i.e., with labels fixed at compile-time, on top of different performant portable libraries, such as Kokkos [@9485033], Kokkos Kernels [@rajamanickam2021kokkos], kokkos-fft [@kokkos-fft], and Ginkgo [@GinkgoJoss2020]. Labelling at compile time is achieved by strongly typing dimensions, an approach similar to that used in units libraries such as mp-units [@Pusz_mp-units_2024], which strongly type quantities rather than dimensions.

The library is actively used to modernize the Fortran-based Gysela plasma simulation code [@Bourne_Gyselalib]. This simulation code relies heavily on high-dimensional arrays. While the data stored in the arrays has 7 dimensions, each dimension can have multiple representations, including Fourier, spline, Cartesian, and various curvilinear meshes. The legacy Fortran implementation was used to manipulate multi-dimensional arrays that stored slices of all the possible dimensions with very limited information about which dimensions were actually represented to enforce correctness at the API level. DDC enables a more explicit, strongly-typed representation of these arrays, ensuring at compile-time that function calls respect the expected dimensions. This reduces indexing errors and improves code maintainability, particularly in large-scale scientific software.

## DDC Core key features

The DDC library is a C++ library designed for expressive and safe handling of multidimensional data. Its core component provides flexible data containers along with algorithms built on top of the performance portable Kokkos library. The library is fully compatible with Kokkos and does not attempt to hide it, allowing users to leverage Kokkos' full capabilities while benefiting from DDC’s strongly-typed, labelled multidimensional arrays when and where it makes sense.

### Strongly-typed labelled indices

DDC employs strongly-typed multi-indices to label dimensions and access data. It introduces two types of multi-indices to access the container's data:

- `DiscreteVector` multi-indices:
  - strongly-typed labelled integers,
  - provide a multidimensional array access semantics,
  - always as fast access as raw multidimensional array,
- `DiscreteElement` multi-indices:
  - strongly-typed labelled keys or opaque identifiers,
  - provide an associative access semantics, as keys in a map container,
  - potentially slower access, depending on the type of set of `DiscreteElement`.

In a DDC container, `DiscreteElement` indices represent absolute positions, while `DiscreteVector` indices are always relative to the beginning of the container.

![Example of two sets of `DiscreteElement`.\label{fig:domains}](domains.pdf)

For example, consider \autoref{fig:domains} that illustrates a two-dimensional data chunk with axes `X` and `Y`. Here `chunk_r` is a container defined over the red area and `chunk_b` is a slice of `chunk_r` over the blue area. Let us define

- `DiscreteElement<X, Y> e(x_c, y_b)`,
- `DiscreteVector<X, Y> v(2, 1)`,
- `DiscreteVector<X, Y> w(0, 1)`.

In this case, the following expressions all refer to the same memory location:

- `chunk_r(e)`,
- `chunk_b(e)`,
- `chunk_r(v)`,
- `chunk_b(w)`.

This highlights the fact that `DiscreteElement` provides a globally consistent indexing mechanism, while `DiscreteVector` is context-dependent and relative to the container’s origin.

### Sets of `DiscreteElement`

The semantics of DDC containers associates data to a set of `DiscreteElement` indices. Let us note that the set of all possible `DiscreteElement` has a total order that is typically established once and for all at program initialization. Thus, to be able to construct a DDC container, one must provide a multidimensional set of `DiscreteElement` indices, where only these indices can be later used to access the container’s data.

The library provides several ways to group `DiscreteElement` into sets, each represented as a Cartesian product of per-dimension sets. These sets offer a lookup function to retrieve the position of a multi-index relative to the front of the set. The performance of container data access depends significantly on the compile-time properties of the set used.

### Multidimensional algorithms

Finally, DDC offers multidimensional algorithms to manipulate the containers and associated `DiscreteElement` indices such as parallel loops and reductions. The parallel versions are based on Kokkos providing performance portability. DDC also provides transform-based algorithms such as discrete Fourier transforms (via a Kokkos-fft wrapper) and spline transforms, enabling conversions between sampled data and coefficients in Fourier or B-spline bases over labeled dimensions.

## Acknowledgements

We acknowledge the financial support of the Cross-Disciplinary Program on "Numerical simulation" of CEA, the French Alternative Energies and Atomic Energy Commission. This work has received support by the CExA Moonshot project of the CEA [cexa-project](https://cexa-project.org). We acknowledge contributions from the Maison de la Simulation. We also thank the developers and contributors of the DDC project for their efforts in making numerical modeling more accessible and efficient.

## References
