# Installation {#installation}

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

## Quick start

A quick way to compile DDC is to disable all optional components providing only core features

```bash
git clone --recurse-submodules --jobs 4 https://github.com/CExA-project/ddc.git
cmake -D DDC_BUILD_KERNELS_FFT=OFF -D DDC_BUILD_KERNELS_SPLINES=OFF -D DDC_BUILD_PDI_WRAPPER=OFF -B ddc/build
cmake --build ddc/build --parallel 4
ctest --test-dir ddc/build
cmake --install ddc/build --prefix path/to/ddc/install
```

## Using Spack package manager

\note We are currently working on providing a DDC recipe for the Spack package manager.

In the meantime, as of Spack 0.23.1, we provide a CPU Spack environment that contains almost all dependencies. The rest of the dependencies are provided by the git submodules.

## Spack installation

### Download

```bash
wget https://github.com/spack/spack/releases/download/v0.23.1/spack-0.23.1.tar.gz
tar -xvf spack-0.23.1.tar.gz
rm spack-0.23.1.tar.gz
```

### Activation

One first needs to activate Spack before executing any command.

```bash
. spack-0.23.1/share/spack/setup-env.sh
```

### DDC environment installation

```bash
spack env create ddc-env ddc-env.yaml
```

Here is the content of `ddc-env.yaml`:

```yaml
# ddc-env.yaml
spack:
  concretizer:
    unify: true
  definitions:
    - compilers:
        - 'gcc@10:14'
    - packages:
        - 'benchmark@1.8:1 ~performance_counters'
        - 'cmake@3.22:3'
        - 'doxygen@1.9.8:1'
        - 'fftw@3.3:3'
        - 'ginkgo@1.8:1'
        - 'googletest@1.14:1 +gmock'
        - 'kokkos@4.4.1:4'
  specs:
    - lapack
    - matrix:
        - [$packages]
        - [$%compilers]
  view:
    default:
      root: .spack-env/view
      exclude: ['gcc-runtime']
  packages:
    all:
      providers:
        lapack: [openblas]
      variants: cxxstd=17
```

### Compilation of DDC

```bash
git clone --recurse-submodules --jobs 4 https://github.com/CExA-project/ddc.git
cmake -D DDC_BUILD_PDI_WRAPPER=OFF -B ddc/build
ctest --test-dir ddc/build
cmake --build ddc/build --parallel 4
```

### Installation of DDC

\warning Installing DDC solely using the bundled libraries (provided via git submodules) is currently untested and not recommended. For better stability, we recommend using system-installed or Spack-installed dependencies whenever possible.
