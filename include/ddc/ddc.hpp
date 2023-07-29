#pragma once

// Misc
#include "ddc/detail/macros.hpp"
#include "ddc/detail/tagged_vector.hpp"
#include "ddc/detail/type_seq.hpp"
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

# if 0 


static auto gkoExecutors() {
  struct {
	std::shared_ptr<gko::OmpExecutor> host_par;
	std::shared_ptr<gko::CudaExecutor> device;
  } executors;
  executors.host_par =  gko::OmpExecutor::create();
  executors.device = gko::CudaExecutor::create(0, executors.host_par);
  return executors;
};
# endif

# if 1
static std::shared_ptr<gko::OmpExecutor> gko_host_par_exec = gko::OmpExecutor::create();
static std::shared_ptr<gko::CudaExecutor> gko_device_exec = gko::CudaExecutor::create(0, gko_host_par_exec);
;
# endif

class DDCInitializer {
public:
  DDCInitializer() { }
};

static DDCInitializer ddc_initializer;
