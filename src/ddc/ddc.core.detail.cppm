// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

module;

#include <ddc/ddc.hpp>

export module ddc.core.detail;

export namespace ddc::detail {

// used by the tests, this should be reworked;
using ::ddc::detail::array;
using ::ddc::detail::convert_to_npy_dtype;
using ::ddc::detail::ddc_to_kokkos_execution_policy;
using ::ddc::detail::display_discretization_store;
using ::ddc::detail::distribute_blocks;
using ::ddc::detail::g_discrete_space_dual;
using ::ddc::detail::NpyArrayView;
using ::ddc::detail::print_demangled_type_name;
using ::ddc::detail::print_uniform_point_sampling;
using ::ddc::detail::save_npy;
using ::ddc::detail::TaggedVector;

using ::ddc::detail::operator<<;
using ::ddc::detail::operator-;
using ::ddc::detail::operator+;
using ::ddc::detail::operator*;

} // namespace ddc::detail
