// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

//! @brief The top-level namespace of DDC.
//! All DDC symbols are defined either in this namespace or in a nested namespace.
namespace ddc {
}

// Misc
#include <ddc/config.hpp>

#include "detail/macros.hpp"
#include "detail/tagged_vector.hpp"
#include "detail/type_seq.hpp"

#include "real_type.hpp"
#include "scope_guard.hpp"

// Containers
#include "aligned_allocator.hpp"
#include "chunk.hpp"
#include "chunk_span.hpp"
#include "chunk_traits.hpp"
#include "kokkos_allocator.hpp"

// Discretizations
#include "discrete_domain.hpp"
#include "discrete_element.hpp"
#include "discrete_space.hpp"
#include "discrete_vector.hpp"
#include "non_uniform_point_sampling.hpp"
#include "periodic_sampling.hpp"
#include "uniform_point_sampling.hpp"

// Algorithms
#include "create_mirror.hpp"
#include "for_each.hpp"
#include "parallel_deepcopy.hpp"
#include "parallel_fill.hpp"
#include "parallel_for_each.hpp"
#include "parallel_transform_reduce.hpp"
#include "reducer.hpp"
#include "transform_reduce.hpp"
