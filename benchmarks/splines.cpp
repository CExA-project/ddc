// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(SPLINES_CPP) {

ddc::SplineSolver const Backend = ddc::SplineSolver::LAPACK;

struct X
{
    static constexpr bool PERIODIC = true;
};

template <bool IsNonUniform, std::size_t s_degree_x>
struct BSplinesX
    : std::conditional_t<
              IsNonUniform,
              ddc::NonUniformBSplines<X, s_degree_x>,
              ddc::UniformBSplines<X, s_degree_x>>
{
};

template <bool IsNonUniform, std::size_t s_degree_x>
using GrevillePoints = ddc::GrevilleInterpolationPoints<
        BSplinesX<IsNonUniform, s_degree_x>,
        ddc::BoundCond::PERIODIC,
        ddc::BoundCond::PERIODIC>;

template <bool IsNonUniform, std::size_t s_degree_x>
struct DDimX : GrevillePoints<IsNonUniform, s_degree_x>::interpolation_discrete_dimension_type
{
};

struct Y;

struct DDimY : ddc::UniformPointSampling<Y>
{
};

// Function to monitor GPU memory asynchronously
void monitorMemoryAsync(std::mutex& mutex, bool& monitorFlag, std::size_t& maxUsedMem)
{
    while (monitorFlag) {
        std::this_thread::sleep_for(std::chrono::microseconds(10)); // Adjust the interval as needed

        // Acquire a lock to ensure thread safety when accessing CUDA functions
        std::lock_guard<std::mutex> const lock(mutex);

#if defined(__CUDACC__)
        std::size_t freeMem = 0;
        std::size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        std::size_t const usedMem = totalMem - freeMem;
#else
        std::size_t const usedMem = 0;
#endif

        maxUsedMem = std::max(maxUsedMem, usedMem);
    }
}

template <typename ExecSpace, bool IsNonUniform, std::size_t s_degree_x>
void characteristics_advection_unitary(benchmark::State& state)
{
    std::size_t const nx = state.range(3);
    std::size_t const ny = state.range(4);
    int cols_per_chunk = state.range(5);
    int preconditioner_max_block_size = state.range(6);


#if defined(__CUDACC__)
    std::size_t freeMem = 0;
    std::size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    // cudaMemGetInfo gives GPU total memory occupation, we consider that other users of the GPU have constant occupancy and substract it.
    std::size_t const initUsedMem = totalMem - freeMem;
#else
    std::size_t const initUsedMem = 0;
#endif
    std::size_t maxUsedMem = initUsedMem;

    bool monitorFlag = true;
    std::mutex mutex;
    // Create a thread to monitor GPU memory asynchronously
    std::thread monitorThread(
            monitorMemoryAsync,
            std::ref(mutex),
            std::ref(monitorFlag),
            std::ref(maxUsedMem));

    if constexpr (!IsNonUniform) {
        ddc::create_uniform_bsplines<BSplinesX<
                IsNonUniform,
                s_degree_x>>(ddc::Coordinate<X>(0.), ddc::Coordinate<X>(1.), nx);
    } else {
        std::vector<ddc::Coordinate<X>> breaks(nx + 1);
        for (std::size_t i(0); i < nx + 1; ++i) {
            breaks[i] = ddc::Coordinate<X>(static_cast<double>(i) / nx);
        }
        ddc::create_non_uniform_bsplines<BSplinesX<IsNonUniform, s_degree_x>>(breaks);
    }
    ddc::init_discrete_space_from_impl<DDimX<IsNonUniform, s_degree_x>>(
            ddc::GrevilleInterpolationPoints<
                    BSplinesX<IsNonUniform, s_degree_x>,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC>::
                    template get_sampling<DDimX<IsNonUniform, s_degree_x>>());
    ddc::DiscreteDomain<DDimY> const y_domain = ddc::create_uniform_point_sampling<
            DDimY>(ddc::Coordinate<Y>(-1.), ddc::Coordinate<Y>(1.), ddc::DiscreteVector<DDimY>(ny));

    auto const x_domain = ddc::GrevilleInterpolationPoints<
            BSplinesX<IsNonUniform, s_degree_x>,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC>::template get_domain<DDimX<IsNonUniform, s_degree_x>>();
    ddc::Chunk density_alloc(
            ddc::DiscreteDomain<DDimX<IsNonUniform, s_degree_x>, DDimY>(x_domain, y_domain),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan const density = density_alloc.span_view();
    // Initialize the density on the main domain
    ddc::DiscreteDomain<DDimX<IsNonUniform, s_degree_x>, DDimY> const x_mesh
            = ddc::DiscreteDomain<DDimX<IsNonUniform, s_degree_x>, DDimY>(x_domain, y_domain);
    ddc::parallel_for_each(
            ExecSpace(),
            x_mesh,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX<IsNonUniform, s_degree_x>, DDimY> const ixy) {
                double const x = ddc::coordinate(
                        ddc::DiscreteElement<DDimX<IsNonUniform, s_degree_x>>(ixy));
                double const y = ddc::coordinate(ddc::DiscreteElement<DDimY>(ixy));
                density(ixy) = 9.999 * Kokkos::exp(-(x * x + y * y) / 0.1 / 2);
                // initial_density(ixy) = 9.999 * ((x * x + y * y) < 0.25);
            });
    ddc::SplineBuilder<
            ExecSpace,
            typename ExecSpace::memory_space,
            BSplinesX<IsNonUniform, s_degree_x>,
            DDimX<IsNonUniform, s_degree_x>,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC,
            Backend,
            DDimX<IsNonUniform, s_degree_x>,
            DDimY> const spline_builder(x_mesh, cols_per_chunk, preconditioner_max_block_size);
    ddc::PeriodicExtrapolationRule<X> const periodic_extrapolation;
    ddc::SplineEvaluator<
            ExecSpace,
            typename ExecSpace::memory_space,
            BSplinesX<IsNonUniform, s_degree_x>,
            DDimX<IsNonUniform, s_degree_x>,
            ddc::PeriodicExtrapolationRule<X>,
            ddc::PeriodicExtrapolationRule<X>,
            DDimX<IsNonUniform, s_degree_x>,
            DDimY> const spline_evaluator(periodic_extrapolation, periodic_extrapolation);
    ddc::Chunk coef_alloc(
            spline_builder.batched_spline_domain(),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan const coef = coef_alloc.span_view();
    ddc::Chunk feet_coords_alloc(
            spline_builder.batched_interpolation_domain(),
            ddc::KokkosAllocator<ddc::Coordinate<X>, typename ExecSpace::memory_space>());
    ddc::ChunkSpan const feet_coords = feet_coords_alloc.span_view();

    for (auto _ : state) {
        Kokkos::Profiling::pushRegion("FeetCharacteristics");
        ddc::parallel_for_each(
                ExecSpace(),
                feet_coords.domain(),
                KOKKOS_LAMBDA(
                        ddc::DiscreteElement<DDimX<IsNonUniform, s_degree_x>, DDimY> const e) {
                    feet_coords(e)
                            = ddc::coordinate(
                                      ddc::DiscreteElement<DDimX<IsNonUniform, s_degree_x>>(e))
                              - ddc::Coordinate<X>(0.0176429863);
                });
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion("SplineBuilder");
        spline_builder(coef, density.span_cview());
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion("SplineEvaluator");
        spline_evaluator(density, feet_coords.span_cview(), coef.span_cview());
        Kokkos::Profiling::popRegion();
        Kokkos::fence("End of advection step");
    }
    monitorFlag = false;
    monitorThread.join();
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(nx * ny * sizeof(double)));
    state.counters["gpu_mem_occupancy"] = maxUsedMem - initUsedMem;
    ////////////////////////////////////////////////////
    /// --------------- HUGE WARNING --------------- ///
    /// The following lines are forbidden in a prod- ///
    /// uction code. It is a necessary workaround    ///
    /// which must be used ONLY for Google Benchmark.///
    /// The reason is it acts on underlying global   ///
    /// variables, which is always a bad idea.       ///
    ////////////////////////////////////////////////////
    ddc::detail::g_discrete_space_dual<BSplinesX<IsNonUniform, s_degree_x>>.reset();
    if constexpr (!IsNonUniform) {
        ddc::detail::g_discrete_space_dual<ddc::UniformBsplinesKnots<BSplinesX<IsNonUniform, s_degree_x>>>.reset();
    } else {
        ddc::detail::g_discrete_space_dual<ddc::NonUniformBsplinesKnots<BSplinesX<IsNonUniform, s_degree_x>>>.reset();
    }
    ddc::detail::g_discrete_space_dual<DDimX<IsNonUniform, s_degree_x>>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ////////////////////////////////////////////////////
}

void characteristics_advection(benchmark::State& state)
{
    long const host = 0;
    long const dev = 1;
    long const uniform = 0;
    long const non_uniform = 1;
    // Preallocate 12 unitary benchs for each combination of cpu/gpu execution space, uniform/non-uniform and spline degree we may want to benchmark (those are determined at compile-time, that's why we need to build explicitely 12 variants of the bench even if we call only one of them)
    std::map<std::array<long, 3>, std::function<void(benchmark::State&)>> benchs;
    benchs[std::array {host, uniform, 3L}]
            = characteristics_advection_unitary<Kokkos::DefaultHostExecutionSpace, false, 3>;
    benchs[std::array {host, uniform, 4L}]
            = characteristics_advection_unitary<Kokkos::DefaultHostExecutionSpace, false, 4>;
    benchs[std::array {host, uniform, 5L}]
            = characteristics_advection_unitary<Kokkos::DefaultHostExecutionSpace, false, 5>;
    benchs[std::array {host, non_uniform, 3L}]
            = characteristics_advection_unitary<Kokkos::DefaultHostExecutionSpace, true, 3>;
    benchs[std::array {host, non_uniform, 4L}]
            = characteristics_advection_unitary<Kokkos::DefaultHostExecutionSpace, true, 4>;
    benchs[std::array {host, non_uniform, 5L}]
            = characteristics_advection_unitary<Kokkos::DefaultHostExecutionSpace, true, 5>;
    benchs[std::array {dev, uniform, 3L}]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, false, 3>;
    benchs[std::array {dev, uniform, 4L}]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, false, 4>;
    benchs[std::array {dev, uniform, 5L}]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, false, 5>;
    benchs[std::array {dev, non_uniform, 3L}]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, true, 3>;
    benchs[std::array {dev, non_uniform, 4L}]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, true, 4>;
    benchs[std::array {dev, non_uniform, 5L}]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, true, 5>;

    // Run the desired bench
    benchs.at(std::array {state.range(0), state.range(1), state.range(2)})(state);
}

} // namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(SPLINES_CPP)

// Reference parameters: the benchmarks sweep on two parameters and fix all the others according to those reference parameters.
bool on_gpu_ref = true;
bool non_uniform_ref = false;
std::size_t degree_x_ref = 3;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
std::size_t cols_per_chunk_ref = 65535;
unsigned int preconditioner_max_block_size_ref = 1U;
#elif defined(KOKKOS_ENABLE_OPENMP)
std::size_t cols_per_chunk_ref = 8192;
unsigned int preconditioner_max_block_size_ref = 1U;
#elif defined(KOKKOS_ENABLE_SERIAL)
std::size_t cols_per_chunk_ref = 8192;
unsigned int preconditioner_max_block_size_ref = 32U;
#endif
// std::size_t ny_ref = 100000;
std::size_t ny_ref = 1000;

// Sweep on spline order
std::string name = "degree_x";
// NOLINTBEGIN(misc-use-anonymous-namespace)
BENCHMARK(characteristics_advection)
        ->RangeMultiplier(2)
        ->Ranges(
                {{false, true},
                 {false, true},
                 {3, 5},
                 {64, 1024},
                 {ny_ref, ny_ref},
                 {cols_per_chunk_ref, cols_per_chunk_ref},
                 {preconditioner_max_block_size_ref, preconditioner_max_block_size_ref}})
        ->MinTime(3)
        ->UseRealTime();
// NOLINTEND(misc-use-anonymous-namespace)

/*
// Sweep on ny
std::string name = "ny";
BENCHMARK(characteristics_advection)
        ->RangeMultiplier(2)
        ->Ranges(
                {{false, true},
                 {false, true},
                 {degree_x_ref, degree_x_ref},
                 {64, 1024},
                 {100, 200000},
                 {cols_per_chunk_ref, cols_per_chunk_ref},
                 {preconditioner_max_block_size_ref, preconditioner_max_block_size_ref}})
        ->MinTime(3)
        ->UseRealTime();
*/
/*
// Sweep on cols_per_chunk
std::string name = "cols_per_chunk";
BENCHMARK(characteristics_advection)
        ->RangeMultiplier(2)
        ->Ranges(
                {{false, true},
                 {false, true},
                 {degree_x_ref, degree_x_ref},
                 {64, 1024},
                 {ny_ref, ny_ref},
                 {64, 65535},
                 {preconditioner_max_block_size_ref, preconditioner_max_block_size_ref}})
        ->MinTime(3)
        ->UseRealTime();
*/
/*
// Sweep on preconditioner_max_block_size
std::string name = "preconditioner_max_block_size";
BENCHMARK(characteristics_advection)
        ->RangeMultiplier(2)
        ->Ranges(
                {{on_gpu_ref, on_gpu_ref},
                 {false, true},
                 {degree_x_ref, degree_x_ref},
                 {64, 1024},
                 {ny_ref, ny_ref},
                 {cols_per_chunk_ref, cols_per_chunk_ref},
                 {1, 32}})
        ->MinTime(3)
        ->UseRealTime();
*/

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::AddCustomContext("name", name);
    ::benchmark::
            AddCustomContext("backend", Backend == ddc::SplineSolver::GINKGO ? "GINKGO" : "LAPACK");
    ::benchmark::AddCustomContext("cols_per_chunk_ref", std::to_string(cols_per_chunk_ref));
    ::benchmark::AddCustomContext(
            "preconditioner_max_block_size_ref",
            std::to_string(preconditioner_max_block_size_ref));
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    {
        Kokkos::ScopeGuard const kokkos_scope(argc, argv);
        ddc::ScopeGuard const ddc_scope(argc, argv);
        ::benchmark::RunSpecifiedBenchmarks();
    }
    ::benchmark::Shutdown();
    return 0;
}
