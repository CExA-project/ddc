// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#    if !defined(KOKKOS_ENABLE_CUDA_CONSTEXPR)
static_assert(false, "DDC requires option -DKokkos_ENABLE_CUDA_CONSTEXPR=ON");
#    endif

#    if !defined(KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE)
static_assert(false, "DDC requires option -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON");
#    endif
#endif

#if defined(KOKKOS_ENABLE_HIP)
#    if !defined(KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE)
static_assert(false, "DDC requires option -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON");
#    endif
#endif

#if defined(KOKKOS_ENABLE_SYCL)
#    if !defined(KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE)
static_assert(false, "DDC requires option -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON");
#    endif
#endif

//! @brief The top-level namespace of DDC.
//! All DDC symbols are defined either in this namespace or in a nested namespace.
namespace ddc {
}

// Misc
#include "ddc/config.hpp"
#include "ddc/detail/macros.hpp"
#include "ddc/detail/tagged_vector.hpp"
#include "ddc/detail/type_seq.hpp"
#include "ddc/real_type.hpp"
#include "ddc/scope_guard.hpp"

// Containers
#include "ddc/aligned_allocator.hpp"
#include "ddc/chunk.hpp"
#include "ddc/chunk_span.hpp"
#include "ddc/chunk_traits.hpp"
#include "ddc/kokkos_allocator.hpp"

// Discretizations
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"
#include "ddc/discrete_vector.hpp"
#include "ddc/non_uniform_point_sampling.hpp"
#include "ddc/periodic_sampling.hpp"
#include "ddc/sparse_discrete_domain.hpp"
#include "ddc/strided_discrete_domain.hpp"
#include "ddc/trivial_space.hpp"
#include "ddc/uniform_point_sampling.hpp"

// Algorithms
#include "ddc/create_mirror.hpp"
#include "ddc/for_each.hpp"
#include "ddc/parallel_deepcopy.hpp"
#include "ddc/parallel_fill.hpp"
#include "ddc/parallel_for_each.hpp"
#include "ddc/parallel_transform_reduce.hpp"
#include "ddc/reducer.hpp"
#include "ddc/transform_reduce.hpp"

// Output
#include "ddc/print.hpp"
