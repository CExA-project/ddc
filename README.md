<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

# The discrete domain computation library (DDC)

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/CExA-project/ddc/badge)](https://scorecard.dev/viewer/?uri=github.com/CExA-project/ddc)
[![Codecov](https://codecov.io/gh/CExA-project/ddc/graph/badge.svg?token=4CZS4MNERP)](https://codecov.io/gh/CExA-project/ddc)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Nightly early integration](https://github.com/CExA-project/ddc/actions/workflows/early_integration.yaml/badge.svg?event=schedule)](https://github.com/CExA-project/ddc/actions/workflows/early_integration.yaml)
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
  * Kokkos-fft 0.3.0...<1
* (optional, IO interface) DDC::pdi
  * PDI 1.6...<2
* (optional, spline interpolation) DDC::splines
  * Ginkgo 1.8...<2
  * Kokkos Kernels 4.5.1...<5

## Getting the code and basic configuration

Please see [this page](docs/installation.md) for a detailed installation guide.

## Contributing

Please see [this page](CONTRIBUTING.md) for details on how to contribute.

## Known issues

* Kokkos 4.5.0 embeds a version of mdspan that is not compatible with DDC, see <https://github.com/kokkos/mdspan/pull/368>. This issue has been fixed in Kokkos 4.5.1.
