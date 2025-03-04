// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

//! @brief The top-level namespace of DDC.
//! All DDC symbols are defined either in this namespace or in a nested namespace.
namespace ddc {
}

// Misc
#include "ddc/config.hpp"
#include "ddc/detail/type_seq.hpp"

// Containers
#include "ddc/chunk.hpp"
#include "ddc/chunk_span.hpp"
#include "ddc/chunk_traits.hpp"
#include "ddc/kokkos_allocator.hpp"

// Discretizations
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_vector.hpp"
#include "ddc/trivial_space.hpp"

// Algorithms
#include "ddc/create_mirror.hpp"
#include "ddc/for_each.hpp"
#include "ddc/parallel_deepcopy.hpp"
#include "ddc/parallel_fill.hpp"
#include "ddc/parallel_for_each.hpp"
#include "ddc/parallel_transform_reduce.hpp"
#include "ddc/reducer.hpp"
