# Using DDC {#using_ddc}

\tableofcontents

<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

## Public targets and headers

The CMake package name is `DDC`, you will find below a table listing the targets and headers.

| CMake component name | CMake target name | C++ headers                 |
|----------------------|-------------------|-----------------------------|
| none                 | `DDC::core`       | `<ddc/ddc.hpp>`             |
| `pdi`                | `DDC::pdi`        | `<ddc/pdi.hpp>`             |
| `fft`                | `DDC::fft`        | `<ddc/kernels/fft.hpp>`     |
| `splines`            | `DDC::splines`    | `<ddc/kernels/splines.hpp>` |

Keep in mind that the DDC components are optional.

> [!WARNING]
> Please note that all other targets and headers are private and thus should not be used.

## CMake examples

If you only need the core features of the library, you can simply write the following cmake code

```cmake
# ...
find_package(DDC X.Y.Z REQUIRED)
# ...
target_link_libraries(YOUR_TARGET DDC::core)
```

If you also rely on the FFT features of DDC, you will need the target `DDC::fft` that is available when the associated component is requested, as below

```cmake
# ...
find_package(DDC X.Y.Z REQUIRED fft)
# ...
target_link_libraries(YOUR_TARGET_USING_FFT DDC::core DDC::fft)
```

Note that you must still link to `DDC::core` because most component functionalities (such as FFT) depend on features provided by the core library.
