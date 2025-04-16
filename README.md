# The discrete domain computation library (DDC)
<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Nightly early integration](https://github.com/CExA-project/ddc/actions/workflows/early_integration.yaml/badge.svg?event=schedule)](https://github.com/CExA-project/ddc/actions/workflows/early_integration.yaml)
[![Nightly tests on Ubuntu](https://github.com/CExA-project/ddc/actions/workflows/tests-ubuntu.yaml/badge.svg?event=schedule)](https://github.com/CExA-project/ddc/actions/workflows/tests-ubuntu.yaml)
[![Nightly tests on macOS](https://github.com/CExA-project/ddc/actions/workflows/tests-macos.yaml/badge.svg?event=schedule)](https://github.com/CExA-project/ddc/actions/workflows/tests-macos.yaml)
[![Nightly tests on Windows](https://github.com/CExA-project/ddc/actions/workflows/tests-windows.yaml/badge.svg?event=schedule)](https://github.com/CExA-project/ddc/actions/workflows/tests-windows.yaml)
[![Pages](https://github.com/CExA-project/ddc/actions/workflows/pages.yaml/badge.svg)](https://github.com/CExA-project/ddc/actions/workflows/pages.yaml)

See <https://ddc.mdls.fr/>

[DDC](https://ddc.mdls.fr/), is a C++-17 library that aims to offer to the C++/MPI world an equivalent to the [`xarray.DataArray`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html)/[`dask.Array`](https://docs.dask.org/en/stable/array.html) python environment.
Where these two libraries are based on [numpy](https://numpy.org/), DDC relies on [Kokkos](https://github.com/kokkos/kokkos) and [mdspan](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p0009r10.html) to offer CPU/GPU performance-portable multi-dimensional arrays and iterators.

DDC aims to offer abstractions to represent:

* tagged continuous dimensions,
* discretizations of these (multiple kinds of meshes, function spaces, Fourier, etc.),
* domains in these discretized dimensions,
* associating data to the elements of these discrete domains,
* efficient distribution and iteration over this data.

All these abstractions are handled at compilation using C++ template meta-programming to ensure zero overhead at execution and enable developers to change their design choices (*e.g.* from a regular mesh to a non-uniform one) with performance portability.

[Find out more...](https://ddc.mdls.fr/)

If you like the project, please leave us a github star.

If you want to know more, join un on [Slack](https://ddc-lib.slack.com/join/shared_invite/zt-33v61oiip-UHnWCavFC0cmff5a94HYwQ)

## Prerequisites

To use DDC core, one needs the following dependencies:

* a C++17-compliant compiler
* CMake 3.22...<4
* Kokkos 4.4...<5
* (optional, micro benchmarking) Benchmark 1.8...<2
* (optional, documentation) Doxygen 1.8.13...<2
* (optional, unit-testing) GoogleTest 1.14...<2

To use DDC components, one needs the following dependencies:

* (optional, fft interface) DDC::fft
  * Kokkos-fft 0.2.1...<1
* (optional, IO interface) DDC::pdi
  * PDI 1.6...<2
* (optional, spline interpolation) DDC::splines
  * Ginkgo 1.8...<2
  * Kokkos Kernels 4.5.1...<5

## Getting the code and basic configuration

```bash
git clone --recurse-submodules --jobs 4 https://github.com/CExA-project/ddc.git
cd ddc
cmake -D DDC_BUILD_KERNELS_FFT=OFF -D DDC_BUILD_KERNELS_SPLINES=OFF -D DDC_BUILD_PDI_WRAPPER=OFF -B build
cmake --build build --parallel 4
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

## Known issues

* Kokkos 4.5.0 embeds a version of mdspan that is not compatible with DDC, see <https://github.com/kokkos/mdspan/pull/368>. This issue has been fixed in Kokkos 4.5.1.
