// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <thread>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <benchmark/benchmark.h>

static const ddc::SplineSolver Backend = ddc::SplineSolver::GINKGO;

namespace DDC_HIP_5_7_ANONYMOUS_NAMESPACE_WORKAROUND(SPLINES_CPP)
{
    struct X
    {
        static constexpr bool PERIODIC = true;
    };

    template <typename NonUniform, std::size_t s_degree_x>
    struct BSplinesX
        : std::conditional_t<
                  NonUniform::value,
                  ddc::NonUniformBSplines<X, s_degree_x>,
                  ddc::UniformBSplines<X, s_degree_x>>
    {
    };

    template <typename NonUniform, std::size_t s_degree_x>
    using GrevillePoints = ddc::GrevilleInterpolationPoints<
            BSplinesX<NonUniform, s_degree_x>,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC>;

    template <typename NonUniform, std::size_t s_degree_x>
    struct DDimX : GrevillePoints<NonUniform, s_degree_x>::interpolation_mesh_type
    {
    };

    struct Y;

    struct DDimY : ddc::UniformPointSampling<Y>
    {
    };

} // namespace )

// Function to monitor GPU memory asynchronously
void monitorMemoryAsync(std::mutex& mutex, bool& monitorFlag, size_t& maxUsedMem)
{
    size_t freeMem = 0;
    size_t totalMem = 0;
    while (monitorFlag) {
        std::this_thread::sleep_for(
                std::chrono::microseconds(100)); // Adjust the interval as needed

        // Acquire a lock to ensure thread safety when accessing CUDA functions
        std::lock_guard<std::mutex> lock(mutex);

#if defined(__CUDACC__)
        cudaMemGetInfo(&freeMem, &totalMem);
#endif
        maxUsedMem = std::max(maxUsedMem, totalMem - freeMem);
    }
}

template <typename ExecSpace, typename NonUniform, std::size_t s_degree_x>
static void characteristics_advection_unitary(benchmark::State& state)
{
    std::size_t nx = state.range(3);
    std::size_t ny = state.range(4);
    int cols_per_chunk = state.range(5);
    int preconditionner_max_block_size = state.range(6);

    size_t freeMem = 0;
    size_t totalMem = 0;
#if defined(__CUDACC__)
    cudaMemGetInfo(&freeMem, &totalMem);
#endif
    size_t initUsedMem
            = totalMem
              - freeMem; // cudaMemGetInfo gives GPU total memory occupation, we consider that other users of the GPU have constant occupancy and substract it.
    size_t maxUsedMem = initUsedMem;

    bool monitorFlag = true;
    std::mutex mutex;
    // Create a thread to monitor GPU memory asynchronously
    std::thread monitorThread(
            monitorMemoryAsync,
            std::ref(mutex),
            std::ref(monitorFlag),
            std::ref(maxUsedMem));

    if constexpr (!NonUniform::value) {
        ddc::init_discrete_space<BSplinesX<
                NonUniform,
                s_degree_x>>(ddc::Coordinate<X>(0.), ddc::Coordinate<X>(1.), nx);
    } else {
        std::vector<ddc::Coordinate<X>> breaks(nx + 1);
        for (std::size_t i(0); i < nx + 1; ++i) {
            breaks[i] = ddc::Coordinate<X>(static_cast<double>(i) / nx);
        }
        ddc::init_discrete_space<BSplinesX<NonUniform, s_degree_x>>(breaks);
    }
    ddc::init_discrete_space<DDimX<NonUniform, s_degree_x>>(
            ddc::GrevilleInterpolationPoints<
                    BSplinesX<NonUniform, s_degree_x>,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC>::
                    template get_sampling<DDimX<NonUniform, s_degree_x>>());
    ddc::DiscreteDomain<DDimY> y_domain = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(-1.),
            ddc::Coordinate<Y>(1.),
            ddc::DiscreteVector<DDimY>(ny)));

    auto const x_domain = ddc::GrevilleInterpolationPoints<
            BSplinesX<NonUniform, s_degree_x>,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC>::template get_domain<DDimX<NonUniform, s_degree_x>>();
    ddc::Chunk density_alloc(
            ddc::DiscreteDomain<DDimX<NonUniform, s_degree_x>, DDimY>(x_domain, y_domain),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan const density = density_alloc.span_view();
    // Initialize the density on the main domain
    ddc::DiscreteDomain<DDimX<NonUniform, s_degree_x>, DDimY> x_mesh
            = ddc::DiscreteDomain<DDimX<NonUniform, s_degree_x>, DDimY>(x_domain, y_domain);
    ddc::parallel_for_each(
            ExecSpace(),
            x_mesh,
            KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX<NonUniform, s_degree_x>, DDimY> const ixy) {
                double const x = ddc::coordinate(ddc::select<DDimX<NonUniform, s_degree_x>>(ixy));
                double const y = ddc::coordinate(ddc::select<DDimY>(ixy));
                density(ixy) = 9.999 * Kokkos::exp(-(x * x + y * y) / 0.1 / 2);
                // initial_density(ixy) = 9.999 * ((x * x + y * y) < 0.25);
            });
    ddc::SplineBuilder<
            ExecSpace,
            typename ExecSpace::memory_space,
            BSplinesX<NonUniform, s_degree_x>,
            DDimX<NonUniform, s_degree_x>,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC,
            Backend,
            DDimX<NonUniform, s_degree_x>,
            DDimY>
            spline_builder(x_mesh, cols_per_chunk, preconditionner_max_block_size);
    ddc::PeriodicExtrapolationRule<X> periodic_extrapolation;
    ddc::SplineEvaluator<
            ExecSpace,
            typename ExecSpace::memory_space,
            BSplinesX<NonUniform, s_degree_x>,
            DDimX<NonUniform, s_degree_x>,
            ddc::PeriodicExtrapolationRule<X>,
            ddc::PeriodicExtrapolationRule<X>,
            DDimX<NonUniform, s_degree_x>,
            DDimY>
            spline_evaluator(periodic_extrapolation, periodic_extrapolation);
    ddc::Chunk coef_alloc(
            spline_builder.batched_spline_domain(),
            ddc::KokkosAllocator<double, typename ExecSpace::memory_space>());
    ddc::ChunkSpan coef = coef_alloc.span_view();
    ddc::Chunk feet_coords_alloc(
            spline_builder.batched_interpolation_domain(),
            ddc::KokkosAllocator<ddc::Coordinate<X, Y>, typename ExecSpace::memory_space>());
    ddc::ChunkSpan feet_coords = feet_coords_alloc.span_view();

    for (auto _ : state) {
        Kokkos::Profiling::pushRegion("FeetCharacteristics");
        ddc::parallel_for_each(
                ExecSpace(),
                feet_coords.domain(),
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX<NonUniform, s_degree_x>, DDimY> const e) {
                    feet_coords(e) = ddc::Coordinate<X, Y>(
                            ddc::coordinate(ddc::select<DDimX<NonUniform, s_degree_x>>(e))
                                    - ddc::Coordinate<X>(0.0176429863),
                            ddc::coordinate(ddc::select<DDimY>(e)));
                });
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion("SplineBuilder");
        spline_builder(coef, density.span_cview());
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion("SplineEvaluator");
        spline_evaluator(density, feet_coords.span_cview(), coef.span_cview());
        Kokkos::Profiling::popRegion();
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
    ddc::detail::g_discrete_space_dual<BSplinesX<NonUniform, s_degree_x>>.reset();
    if constexpr (!NonUniform::value) {
        ddc::detail::g_discrete_space_dual<ddc::UniformBsplinesKnots<BSplinesX<NonUniform, s_degree_x>>>.reset();
    } else {
        ddc::detail::g_discrete_space_dual<ddc::NonUniformBsplinesKnots<BSplinesX<NonUniform, s_degree_x>>>.reset();
    }
    ddc::detail::g_discrete_space_dual<DDimX<NonUniform, s_degree_x>>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    ////////////////////////////////////////////////////
}

static void characteristics_advection(benchmark::State& state)
{
    // Preallocate 12 unitary benchs for each combination of cpu/gpu execution space, uniform/non-uniform and spline degree we may want to benchmark (those are determined at compile-time, that's why we need to build explicitely 12 variants of the bench even if we call only one of them)
    std::array<std::function<void(benchmark::State&)>, 12> benchs;
    benchs[0] = characteristics_advection_unitary<
            Kokkos::DefaultHostExecutionSpace,
            std::false_type,
            3>;
    benchs[1] = characteristics_advection_unitary<
            Kokkos::DefaultHostExecutionSpace,
            std::false_type,
            4>;
    benchs[2] = characteristics_advection_unitary<
            Kokkos::DefaultHostExecutionSpace,
            std::false_type,
            5>;
    benchs[3] = characteristics_advection_unitary<
            Kokkos::DefaultHostExecutionSpace,
            std::true_type,
            3>;
    benchs[4] = characteristics_advection_unitary<
            Kokkos::DefaultHostExecutionSpace,
            std::true_type,
            4>;
    benchs[5] = characteristics_advection_unitary<
            Kokkos::DefaultHostExecutionSpace,
            std::true_type,
            5>;
    benchs[6]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, std::false_type, 3>;
    benchs[7]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, std::false_type, 4>;
    benchs[8]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, std::false_type, 5>;
    benchs[9] = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, std::true_type, 3>;
    benchs[10]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, std::true_type, 4>;
    benchs[11]
            = characteristics_advection_unitary<Kokkos::DefaultExecutionSpace, std::true_type, 5>;

    // Run the desired bench
    benchs[state.range(0) * 6 + state.range(1) * 3 + state.range(2) - 3](state);
}

// Reference parameters: the benchmarks sweep on two parameters and fix all the others according to those reference parameters.
bool on_gpu_ref = true;
bool non_uniform_ref = false;
std::size_t degree_x_ref = 3;
#if (defined(KOKKOS_ENABLE_CUDA) or defined(KOKKOS_ENABLE_HIP))
std::size_t cols_per_chunk_ref = 65535;
unsigned int preconditionner_max_block_size_ref = 1u;
#elif defined(KOKKOS_ENABLE_OPENMP)
std::size_t cols_per_chunk_ref = 8192;
unsigned int preconditionner_max_block_size_ref = 1u;
#elif defined(KOKKOS_ENABLE_SERIAL)
std::size_t cols_per_chunk_ref = 8192;
unsigned int preconditionner_max_block_size_ref = 32u;
#endif
// std::size_t ny_ref = 100000;
std::size_t ny_ref = 1000;

// Sweep on spline order
std::string name = "degree_x";
BENCHMARK(characteristics_advection)
        ->RangeMultiplier(2)
        ->Ranges(
                {{false, true},
                 {false, true},
                 {3, 5},
                 {64, 1024},
                 {ny_ref, ny_ref},
                 {cols_per_chunk_ref, cols_per_chunk_ref},
                 {preconditionner_max_block_size_ref, preconditionner_max_block_size_ref}})
        ->MinTime(3)
        ->UseRealTime();
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
                 {preconditionner_max_block_size_ref, preconditionner_max_block_size_ref}})
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
                 {preconditionner_max_block_size_ref, preconditionner_max_block_size_ref}})
        ->MinTime(3)
        ->UseRealTime();
*/
/*
// Sweep on preconditionner_max_block_size
std::string name = "preconditionner_max_block_size"
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
    ::benchmark::AddCustomContext(
            "backend",
            std::is_same_v<backend, ddc::SplineSolver::GINKGO> ? "GINKGO" : "LAPACK");
    ::benchmark::AddCustomContext("cols_per_chunk_ref", std::to_string(cols_per_chunk_ref));
    ::benchmark::AddCustomContext(
            "preconditionner_max_block_size_ref",
            std::to_string(preconditionner_max_block_size_ref));
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
