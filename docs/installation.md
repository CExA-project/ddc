# Installation guide {#installation}

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

This guide describes two ways to install **DDC**:

* **Quick start build** → fastest way to compile the core features.
* **Spack installation** → recommended for reproducible builds and HPC environments.

---

## Quick Start (core features only)

This method compiles DDC with all optional components disabled.

```bash
git clone --recurse-submodules --jobs 4 https://github.com/CExA-project/ddc.git
cmake -D DDC_BUILD_KERNELS_FFT=OFF \
      -D DDC_BUILD_KERNELS_SPLINES=OFF \
      -D DDC_BUILD_PDI_WRAPPER=OFF \
      -B ddc/build \
      -S ddc
cmake --build ddc/build --parallel 4
ctest --test-dir ddc/build
cmake --install ddc/build --prefix $HOME/.local/ddc
```

\note Replace `$HOME/.local/ddc` with your preferred installation path.

### Using the installed package

To use DDC in your CMake project, point CMake to the installation prefix:

```bash
cmake -D DDC_ROOT=$HOME/.local/ddc <your_project>
```

Then in your `CMakeLists.txt`:

```cmake
find_package(DDC REQUIRED)
target_link_libraries(myapp PRIVATE DDC::core)
```

---

## Installing with Spack

This method leverages the [Spack package manager](https://spack.io/) for reproducible builds and dependency management.
It is recommended on HPC systems or if you want an isolated environment.

### 1. Install Spack

Download and extract Spack:

```bash
wget https://github.com/spack/spack/releases/download/v1.1.0/spack-1.1.0.tar.gz
tar -xvf spack-1.1.0.tar.gz
rm spack-1.1.0.tar.gz
```

\note If you already have a Spack installation, the DDC package is available starting from builtin packages `v2025.11.0`.

### 2. Activate Spack

Spack must be activated before use:

```bash
. spack-1.1.0/share/spack/setup-env.sh
```

\note Add this line to your `~/.bashrc` or `~/.zshrc` to avoid reactivating in each new shell.

### 3. Install DDC

To avoid polluting other Spack setups, create a dedicated environment:

```bash
spack env create ddc-env
```

Add and install DDC with default variants:

```bash
spack --env ddc-env add ddc
spack --env ddc-env install --jobs 1
```

\note Installation can take a while since Spack builds dependencies from source. Use the `--jobs` option to run builds in parallel, but avoid setting it too high to prevent running out of memory.

### 4. Use DDC from Spack

Activate the environment:

```bash
spack env activate ddc-env
```

Now any CMake project can find DDC automatically:

```cmake
find_package(DDC REQUIRED)
target_link_libraries(myapp PRIVATE DDC::core)
```

To discover available build variants:

```bash
spack info ddc
```

If you encounter issues with the Spack recipe, please open an issue on the [spack-packages repository](https://github.com/spack/spack-packages/issues).

---

✅ You now have DDC installed and ready to use!
