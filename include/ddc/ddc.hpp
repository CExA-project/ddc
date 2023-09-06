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
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                                 Kokkos::Serial>) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                                 Kokkos::OpenMP>) {
        return gko::OmpExecutor::create();
    }
#endif
} // Comes from "Basic Kokkos Extension" Ginkgo MR 

# if 0
template <typename ExecSpace,
          typename MemorySpace = typename ExecSpace::memory_space>
static std::shared_ptr<gko::Executor> create_executor(ExecSpace, MemorySpace = {})
{
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible);
#ifdef KOKKOS_ENABLE_SERIAL
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return gko::ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return gko::OmpExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
            return gko::CudaExecutor::create(Kokkos::device_id(),
                                        create_default_host_executor(),
                                        std::make_shared<gko::CudaAllocator>());

        }
        if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
            return gko::CudaExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<gko::CudaUnifiedAllocator>(
                    Kokkos::device_id()));
        }
        if constexpr (std::is_same_v<MemorySpace,
                                     Kokkos::CudaHostPinnedSpace>) {
            return gko::CudaExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<gko::CudaHostAllocator>(Kokkos::device_id()));
        }
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
            return gko::HipExecutor::create(Kokkos::device_id(),
                                       create_default_host_executor(),
                                       std::make_shared<gko::HipAllocator>());
        }
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPManagedSpace>) {
            return gko::HipExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<gko::HipUnifiedAllocator>(
                    Kokkos::device_id()));
        }
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPHostPinnedSpace>) {
            return gko::HipExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<gko::HipHostAllocator>(Kokkos::device_id()));
        }
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Experimental::SYCL>) {
        // for now Ginkgo doesn't support different allocators for SYCL
        return gko::DpcppExecutor::create(Kokkos::device_id(),
                                     create_default_host_executor());
    }
#endif
} // Comes from "Basic Kokkos Extension" Ginkgo MR 

static std::shared_ptr<gko::Executor> create_default_executor()
{
    return create_executor(Kokkos::DefaultExecutionSpace{});
} // Comes from "Basic Kokkos Extension" Ginkgo MR
# endif // Not working for some reason

static std::shared_ptr<gko::Executor> create_default_executor() {
#ifdef KOKKOS_ENABLE_SERIAL
        if (std::is_same_v<Kokkos::DefaultExecutionSpace,
                         Kokkos::Serial>) {
            return gko::ReferenceExecutor::create();
        }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        if (std::is_same_v<Kokkos::DefaultExecutionSpace,
                         Kokkos::OpenMP>) {
            return gko::OmpExecutor::create();
        }
#endif
#ifdef KOKKOS_ENABLE_CUDA
        if (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>) {
            return gko::CudaExecutor::create(0,
                                             create_default_host_executor());
        }
#endif
} // Comes from kokkos_assembly example in Ginkgo develop branch

static std::shared_ptr<gko::Executor> gko_default_host_exec = create_default_host_executor();
static std::shared_ptr<gko::Executor> gko_default_exec = create_default_executor();

class DDCInitializer {
public:
  DDCInitializer() {
	# if 0
	// gko_omp_exec = gko::OmpExecutor::create();
	gko_default_host_exec = gko::OmpExecutor::create();
	// gko_cuda_exec = gko::CudaExecutor::create(0, gko_default_host_exec);
	
	if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) {
	  gko_default_exec->create();
	}
	else if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>) {
	  gko_default_exec->create(0, gko_default_host_exec);
	}

	# endif
  }
};

static DDCInitializer ddc_initializer;
