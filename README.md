# The discrete domain computation library (DDC)
<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![test](https://github.com/CExA-project/ddc/actions/workflows/tests.yml/badge.svg)](https://github.com/CExA-project/ddc/actions/workflows/tests.yml)

See <https://ddc.mdls.fr/>

[DDC](https://ddc.mdls.fr/), is a C++-17 library that aims to offer to the C++/MPI world an equivalent to the [`xarray.DataArray`](https://xarray.pydata.org/en/stable/generated/xarray.DataArray.html)/[`dask.Array`](https://docs.dask.org/en/stable/array.html) python environment.
Where these two libraries are based on [numpy](https://numpy.org/), DDC relies on [Kokkos](https://github.com/kokkos/kokkos) and [mdspan](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0009r10.html) to offer CPU/GPU performance-portable multi-dimensional arrays and iterators.

DDC aims to offer abstractions to represent:

* tagged continuous dimensions,
* discretizations of these (multiple kinds of meshes, function spaces, Fourier, etc.),
* domains in these discretized dimensions,
* associating data to the elements of these discrete domains,
* efficient distribution and iteration over this data.

All these abstractions are handled at compilation using C++ template meta-programming to ensure zero overhead at execution and enable developers to change their design choices (*e.g.* from a regular mesh to a non-uniform one) with performance portability.

[Find out more...](https://ddc.mdls.fr/)

If you like the project, please leave us a github star.

If you want to know more, join un on [Slack](https://join.slack.com/t/ddc-lib/shared_invite/zt-14b6rjcrn-AwSfM6_arEamAKk_VgQPhg)

## Prerequisites

To use DDC core, one needs the following dependencies:

* a C++17-compliant compiler
* CMake 3.22...<4
* Kokkos 4.4...4.5
* (optional) Benchmark 1.8...<2
* (optional) Doxygen 1.8.13...<2
* (optional) GoogleTest 1.14...<2

To use DDC components, one needs the following dependencies:

* (optional) DDC::fft
  * Kokkos-fft 0.2.1...<1
* (optional) DDC::pdi
  * PDI 1.6...<2
* (optional) DDC::splines
  * Ginkgo 1.8.0
  * Kokkos Kernels fork <https://github.com/yasahi-hpc/kokkos-kernels> on branch develop-spline-kernels-v2


## Getting the code and basic configuration

```bash
git clone --recurse-submodules -j4 https://github.com/CExA-project/ddc.git
cd ddc
cmake -B build -D DDC_BUILD_KERNELS_FFT=OFF -D DDC_BUILD_KERNELS_SPLINES=OFF -D DDC_BUILD_PDI_WRAPPER=OFF
cmake --build build
```

## Contributing

### Formatting

The project makes use of formatting tools for the C++ ([clang-format](https://clang.llvm.org/docs/ClangFormat.html)) and cmake ([gersemi](https://github.com/BlankSpruce/gersemi)) files. The formatting must be applied for a PR to be accepted.

To format a cmake file, please apply the command

```bash
gersemi -i the-cmake-file
```

One can find the formatting style in the file `.gersemirc`.

To format a C++ file, please apply the command

```bash
clang-format -i the-cpp-file
```

One can find the formatting style in the file `.clang-format`.

> [!WARNING]
> The formatting might not give the same result with different versions of a tool.
