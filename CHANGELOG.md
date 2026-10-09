<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [UNRELEASED]

### Added

* Add concept `ddc::concepts::bsplines` in <https://github.com/CExA-project/ddc/pull/1255>
* Add concept `ddc::concepts::discrete dimension` in <https://github.com/CExA-project/ddc/pull/1246>
* Add concept `ddc::concepts::type_seq` in <https://github.com/CExA-project/ddc/pull/1258>

### Changed

* The class `ddc::detail::TypeSeq` is now part of the public API, `ddc::TypeSeq` in <https://github.com/CExA-project/ddc/pull/1249>

### Deprecated

* Deprecate `ddc::detail::TypeSeq` in <https://github.com/CExA-project/ddc/pull/1249>

### Removed

* Remove deprecated code from v0.16.0 in <https://github.com/CExA-project/ddc/pull/1251>

### Fixed

### Dependency requirements

## [v0.16.0] - 2026-09-22

### Added

* Add a layout conversion constructor by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1198>
* Add `static_assert` for MemorySpace in `ChunkSpan` by @Quntized in <https://github.com/CExA-project/ddc/pull/1202>
* Add Spack CI by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1219>
* Add CMake components by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1225>
* Allow different precisions for spline calculations by @EmilyBourne in <https://github.com/CExA-project/ddc/pull/1232>
* Add helper to combine tagged types by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1238>
* Generalise `ConstantExtrapolationRule` implementation to ND by @EmilyBourne in <https://github.com/CExA-project/ddc/pull/1236>

### Changed

* Create BSplines on a subset of a periodic domain by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1233>

### Deprecated

* Deprecate aliases in `view.hpp` by @EmilyBourne in <https://github.com/CExA-project/ddc/pull/1234>

### Removed

* Remove vendored dependencies and related CMake options by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1222>
* Remove `Kokkos::Experimental` symbols by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1227>

### Fixed

* Fix 2D extrapolation rule usage in `SplineEvaluatorND` by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1210>

### Dependency requirements

* Ginkgo: bump minimum requirement to 1.9 by @tpadioleau in <https://github.com/CExA-project/ddc/pull/1218>
