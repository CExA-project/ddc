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

The Discrete Domain Computation (DDC) library is a C++ library designed to provide high-performance, strongly-typed labelled multidimensional arrays. Inspired by Python's Xarray and built on top of performance-portable libraries like Kokkos, DDC enables expressive, safe, and efficient numerical computations. It provides a coherent ecosystem to work with labelled dimensions from data structures to algorithms. Additionally, DDC extends functionality through modules such as FFT (based on Kokkos-fft), splines, and with a bridge to the PDI library. The library is actively used to modernize legacy scientific codes, such as the Fortran-based Gysela plasma simulation code [@grandgirard2016].

## Statement of need

The use of multidimensional arrays is widespread across various fields, particularly in scientific computing, where they serve as fundamental data containers. A primary motivation for their use is their potential to improve computational performance by leveraging problem-specific structure. For instance, when solving a partial differential equation that results in a stencil problem, computations typically achieve higher efficiency on a structured mesh compared to an unstructured mesh. This advantage primarily stems from a better usage of memory like predictable memory accesses and better cache utilization.

Many programming languages commonly used in scientific computing support multidimensional arrays in different ways. Fortran, a longstanding choice in the field, and Julia, a more recent language, both natively support these data structures. In contrast, the Python ecosystem relies on the popular NumPy library’s `numpy.Array` [@harris2020array]. Meanwhile, C++23 introduced `std::mdspan` to the standard library. This container was inspired by `Kokkos::View` from the Kokkos library which also serves as the foundation of DDC.

Despite their importance, multidimensional arrays introduce several practical challenges. In a sense, they encourage the usage of implicits in the source code. A frequent source of errors is the inadvertent swapping of indices when accessing elements. Such errors can be difficult to detect, especially given the common convention of using single-letter variable names like `i` and `j` for indexing. Another challenge in medium to large codebases is the lack of semantic clarity in function signatures when using raw multidimensional arrays. When array dimensions carry specific meanings, this information is not explicitly represented in the source code, leaving it up to the user to ensure that dimensions are ordered correctly according to implicit expectations. For example it is quite usual to use the same index for multiple interpretations: looping over mesh cells identified by `i` and interpreting `i+1` as the face to the right. Another example is slicing that removes dimensions, this operation may change the relative ordering of dimensions.

Solutions have been proposed in Python and Julia to address these issues. In Python, the Xarray [@hoyer2017xarray] and Pandas [@reback2020pandas] libraries allow users to label dimensions that can then be used to perform computation. Following a similar approach, the "Discrete Domain Computation" (DDC) library aims to bring equivalent functionality to the C++ ecosystem. It uses a zero overhead abstraction approach, i.e. with labels fixed at compile-time, on top of different performant portable libraries, such as: Kokkos [@9485033], Kokkos Kernels (TODO: add a citation), Kokkos-fft (TODO: add a citation) and Ginkgo [@GinkgoJoss2020].

The library is actively used to modernize the Fortran-based Gysela plasma simulation code (TODO: add a citation). This simulation code relies heavily on high-dimensional arrays that span multiple representations, including Fourier, spline, Cartesian, and various curvilinear meshes. In the legacy Fortran implementation, these arrays were manipulated using implicit conventions, making it difficult to enforce correctness at the API level. DDC enables a more explicit, strongly-typed representation of these arrays, ensuring at compile-time that function calls respect the expected dimensions. This reduces indexing errors and improves code maintainability, particularly in large-scale scientific software.

## DDC Core key features

The DDC library is a C++ library designed for expressive and safe handling of multidimensional data. Its core component provides flexible data containers along with algorithms built on top of the performance portable Kokkos library. The library is fully compatible with Kokkos and does not attempt to hide it, allowing users to leverage Kokkos' full capabilities while benefiting from DDC’s strongly-typed, labelled multidimensional arrays.

### Containers

DDC offers two multidimensional containers designed over the C++ 23 multidimensional array `std::mdspan`:

- `Chunk` an owning container, i.e. it manages the lifetime of the underlying memory allocation,
- `ChunkSpan` is a non-owning container view over existing memory allocation.

### Strongly-typed labelled indices

DDC employs strongly-typed multi-indices to label dimensions and access data. It introduces two types of indices to access the container's data:

- `DiscreteVector` indices:
  - strongly-typed labelled integers,
  - provide a multidimensional array access semantics,
  - always as fast access as raw multidimensional array,
- `DiscreteElement` indices:
  - strongly-typed labelled keys or opaque identifiers,
  - provide an associative access semantics, as keys in a map container,
  - potentially slower access, depending on the type of set of `DiscreteElement`.

Unlike `DiscreteVector` indices, users cannot directly interpret the internal representation of `DiscreteElement` and must reason about them based solely on their relative position.

### Sets of `DiscreteElement`

The semantics of DDC containers is to associate data to a set of `DiscreteElement` indices. Let us note that the set of all possible `DiscreteElement` has a total order that is typically established once and for all at program initialisation. Thus to be able to construct a DDC container one must provide a multidimensional set of `DiscreteElement` indices, only these indices can be later used to access the container’s data.

The set of `DiscreteElement` is a customization point of the library. It takes the form of a Cartesian product of the different dimensions. DDC predefines the following sets:

- `DiscreteDomain`, a Cartesian product of intervals of `DiscreteElement` in each dimension,
- `StridedDiscreteDomain`, a Cartesian product of sets of `DiscreteElement` with a fixed step/stride in each dimension,
- `SparseDiscreteDomain`, a Cartesian product of explicitly ordered lists of `DiscreteElement` in each dimension.

The performance of the container's data access depends on the properties of the set considered. Indeed the set is used to retrieve the position of a given multi-index relatively to the front multi-index. Thus the performance of this operation depends on the information available in the set.

### Slicing

Similar to `std::mdspan`, DDC containers support slicing through the bracket operator that always returns a `ChunkSpan`. The key property is that the resulting slice’s set of `DiscreteElement` is a subset of the original set.

### Multidimensional algorithms

Finally, DDC offers multidimensional algorithms to manipulate the containers and associated `DiscreteElement` indices. The parallel versions are based on Kokkos providing performance portability. Here is a list as of today:

- `parallel_deepcopy`, copies two `ChunkSpan` that are defined on the same set of `DiscreteElement`,
- `parallel_for_each`, applies a function once per `DiscreteElement` of a set of `DiscreteElement`,
- `parallel_fill`, fills a `ChunkSpan` with a value,
- `parallel_transform_reduce`, applies a function once per `DiscreteElement` of a set of `DiscreteElement` and combines the returned values with a reducer.

## Example

Let us illustrate the basic concepts introduced above with an example that initializes an array with a given value.

```cpp
#include <iostream>

#include <ddc/ddc.hpp>

#include <Kokkos_Core.hpp>

// Define two independent dimensions
struct Dim1
{
};
struct Dim2
{
};

// For the purpose of the demonstration, this function makes only sense with Dim2
int sum_over_dim2(ddc::ChunkSpan<int, ddc::DiscreteDomain<Dim2>> slice)
{
    int sum = 0;
    for (ddc::DiscreteElement<Dim2> idx2 : slice.domain()) {
        sum += slice(idx2);
    }
    return sum;
}

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
    for (ddc::DiscreteElement<Dim1> const idx1 : dom1) {
        // Iterate over the second dimension (Dim2)
        for (ddc::DiscreteElement<Dim2> const idx2 : dom2) {
            // Assign the value 1 to each element in the 2D array
            my_array(idx1, idx2) = 1;

            // The following would NOT compile as my_array expects a DiscreteElement over Dim2
            // my_array(idx1, idx1) = 1;
        }
    }

    // Extract a 1D view over Dim2 for each idx1
    for (ddc::DiscreteElement<Dim1> const idx1 : dom1) {
        // The following would NOT compile if sum_over_dim2 was called
        // with a `DiscreteDomain<Dim1>`, ensuring type safety.
        std::cout << sum_over_dim2(my_array[idx1]) << '\n';
    }

    return 0;
}
```

## DDC components

Built on top of DDC core, we also provide optional components as in the Xarray ecosystem. As of today we provide three components: fft, pdi and splines.

### DDC fft and splines

The FFT and splines components both operate on specialized dimensions provided by DDC: `UniformPointSampling` and `NonUniformPointSampling`.

- FFT extension: This component wraps the Kokkos-fft library to provide labelled semantics for the discrete Fourier transform (DFT). It converts data between a uniform sampling and Fourier coefficients in the Fourier basis. Depending on the transformation direction, the input array is defined over `UniformPointSampling`, while the output is defined over `PeriodicPointSampling`, which represents the Fourier dimension in DDC.
- Splines extension: Similar to FFT, this component provides a spline transform, converting between sampled data and B-spline coefficients in a spline basis. Depending on the transformation direction, the input array is defined over `UniformPointSampling` or `NonUniformPointSampling`, while the output is defined over `UniformBSplines` or `NonUniformBSplines`. (TODO: add a citation)

### DDC pdi

[PDI](https://pdi.dev/main) is a C data interface library allowing loose coupling with external libraries through PDI plugins like HDF5, NetCDF, Catalyst and more. This extension eases the metadata serialisation of the DDC containers. Instead of manually exposing sizes, strides and the pointer of the underlying array, the user can directly expose the DDC container.

## Acknowledgements

This work has received support by the CExA Moonshot project of the CEA [cexa-project](https://cexa-project.org). We acknowledge contributions from the Maison de la Simulation. We also thank the developers and contributors of the DDC project for their efforts in making numerical modeling more accessible and efficient.

## References
