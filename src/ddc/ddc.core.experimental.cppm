// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

module;

#include <ddc/ddc.hpp>

export module ddc.core.experimental;

export namespace ddc::experimental {

using ::ddc::experimental::Dims;
using ::ddc::experimental::parallel_transform_exclusive_scan;
using ::ddc::experimental::parallel_transform_inclusive_scan;
using ::ddc::experimental::parallel_transform_reduce;
using ::ddc::experimental::save_npy;

} // namespace ddc::experimental
