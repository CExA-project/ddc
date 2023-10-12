#pragma once

// Misc
#include <memory>

#include "ddc/detail/macros.hpp"
#include "ddc/detail/tagged_vector.hpp"
#include "ddc/detail/type_seq.hpp"
#include "ddc/real_type.hpp"
#include "ddc/scope_guard.hpp"

// Containers
#include "ddc/aligned_allocator.hpp"
#include "ddc/chunk.hpp"
#include "ddc/chunk_span.hpp"
#include "ddc/kokkos_allocator.hpp"

// Discretizations
#include "ddc/coordinate_md.hpp"
#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_element.hpp"
#include "ddc/discrete_space.hpp"
#include "ddc/discrete_vector.hpp"
#include "ddc/non_uniform_point_sampling.hpp"
#include "ddc/periodic_sampling.hpp"
#include "ddc/rectilinear_domain.hpp"
#include "ddc/uniform_point_sampling.hpp"

// Algorithms
#include "ddc/deepcopy.hpp"
#include "ddc/fill.hpp"
#include "ddc/for_each.hpp"
#include "ddc/reducer.hpp"
#include "ddc/transform_reduce.hpp"

// PDI wrapper
#if defined(DDC_BUILD_PDI_WRAPPER)
#include "ddc/pdi.hpp"
#endif

// PETSc
#if petsc_AVAIL
#include "petscsys.h"
#endif

#if ginkgo_AVAIL
#include <ginkgo/ginkgo.hpp>
#endif

static std::shared_ptr<gko::Executor> create_default_host_executor()
{
#ifdef KOKKOS_ENABLE_SERIAL
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::Serial>) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace, Kokkos::OpenMP>) {
        return gko::OmpExecutor::create();
    }
#endif
} // Comes from "Basic Kokkos Extension" Ginkgo MR

template <typename ExecSpace>
static std::shared_ptr<gko::Executor> create_gko_exec()
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return gko::OmpExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return gko::CudaExecutor::create(0, create_default_host_executor());
    }
#endif
}

class DDCInitializer
{
public:
    DDCInitializer()
    {
    }
};

static DDCInitializer ddc_initializer;
