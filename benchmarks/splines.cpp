// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <vector>

#include <ddc/ddc.hpp>
#include <ddc/kernels/splines.hpp>

#include <benchmark/benchmark.h>

namespace {

static constexpr std::size_t s_degree_x = 3;

struct X
{
    static constexpr bool PERIODIC = true;
};

using BSplinesX = ddc::UniformBSplines<X, s_degree_x>;
using GrevillePoints = ddc::
        GrevilleInterpolationPoints<BSplinesX, ddc::BoundCond::PERIODIC, ddc::BoundCond::PERIODIC>;
using DDimX = GrevillePoints::interpolation_mesh_type;

struct Y;
using DDimY = ddc::UniformPointSampling<Y>;

static std::size_t constexpr large_dim1_2D = 2000;
static std::size_t constexpr large_dim2_2D = large_dim1_2D;

} // namespace

static void DoSetup() {}

static void characteristics_advection(benchmark::State& state)
{
    ddc::init_discrete_space<
            BSplinesX>(ddc::Coordinate<X>(-1.), ddc::Coordinate<X>(1.), state.range(0));
    ddc::init_discrete_space<DDimX>(ddc::GrevilleInterpolationPoints<
                                    BSplinesX,
                                    ddc::BoundCond::PERIODIC,
                                    ddc::BoundCond::PERIODIC>::get_sampling());
    ddc::DiscreteDomain<DDimY> y_domain
            = ddc::init_discrete_space(DDimY::
                                               init(ddc::Coordinate<Y>(-1.),
                                                    ddc::Coordinate<Y>(1.),
                                                    ddc::DiscreteVector<DDimY>(state.range(1))));

    auto const x_domain = ddc::GrevilleInterpolationPoints<
            BSplinesX,
            ddc::BoundCond::PERIODIC,
            ddc::BoundCond::PERIODIC>::get_domain();
    ddc::Chunk density_alloc(
            ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain),
            ddc::DeviceAllocator<double>());
    ddc::ChunkSpan const density = density_alloc.span_view();
    // Initialize the density on the main domain
    ddc::DiscreteDomain<DDimX, DDimY> x_mesh
            = ddc::DiscreteDomain<DDimX, DDimY>(x_domain, y_domain);
    ddc::for_each(
            ddc::policies::parallel_device,
            x_mesh,
            DDC_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> const ixy) {
                double const x = ddc::coordinate(ddc::select<DDimX>(ixy));
                double const y = ddc::coordinate(ddc::select<DDimY>(ixy));
                density(ixy) = 9.999 * Kokkos::exp(-(x * x + y * y) / 0.1 / 2);
                // initial_density(ixy) = 9.999 * ((x * x + y * y) < 0.25);
            });
    ddc::SplineBuilderBatched<
            ddc::SplineBuilder<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::DefaultExecutionSpace::memory_space,
                    BSplinesX,
                    DDimX,
                    ddc::BoundCond::PERIODIC,
                    ddc::BoundCond::PERIODIC>,
            DDimX,
            DDimY>
            spline_builder(x_mesh);
    ddc::SplineEvaluatorBatched<
            ddc::SplineEvaluator<
                    Kokkos::DefaultExecutionSpace,
                    Kokkos::DefaultExecutionSpace::memory_space,
                    BSplinesX,
                    DDimX>,
            DDimX,
            DDimY>
            spline_evaluator(
                    spline_builder.spline_domain(),
                    ddc::g_null_boundary<BSplinesX>,
                    ddc::g_null_boundary<BSplinesX>);
    ddc::Chunk coef_alloc(
            spline_builder.spline_domain(),
            ddc::KokkosAllocator<double, Kokkos::DefaultExecutionSpace::memory_space>());
    ddc::ChunkSpan coef = coef_alloc.span_view();
    ddc::Chunk feet_coords_alloc(
            spline_builder.vals_domain(),
            ddc::KokkosAllocator<
                    ddc::Coordinate<X, Y>,
                    Kokkos::DefaultExecutionSpace::memory_space>());
    ddc::ChunkSpan feet_coords = feet_coords_alloc.span_view();

    for (auto _ : state) {
        ddc::for_each(
                ddc::policies::parallel_device,
                feet_coords.domain(),
                DDC_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> const e) {
                    feet_coords(e) = ddc::Coordinate<X, Y>(
                            ddc::coordinate(ddc::select<DDimX>(e))
                                    - ddc::Coordinate<X>(0.0176429863),
                            ddc::coordinate(ddc::select<DDimY>(e)));
                });
        spline_builder(coef, density);
        spline_evaluator(density, feet_coords.span_cview(), coef.span_cview());
    }
    state.SetBytesProcessed(
            int64_t(state.iterations())
            * int64_t(state.range(0) * state.range(1) * sizeof(double)));
    ////////////////////////////////////////////////////
    /// --------------- HUGE WARNING --------------- ///
    /// The following lines are forbidden in a prod- ///
    /// uction code. It is a necessary workaround    ///
    /// which must be used ONLY for Google Benchmark.///
    /// The reason is it acts on underlying global   ///
    /// variables, which is always a bad idea.       ///
    ////////////////////////////////////////////////////
    ddc::detail::g_discrete_space_dual<BSplinesX>.reset();
    ddc::detail::g_discrete_space_dual<BSplinesX::mesh_type>.reset();
    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
}

BENCHMARK(characteristics_advection)
        ->RangeMultiplier(10)
        ->Ranges({{10, 10000}, {10, 100000}})
        ->Iterations(10);

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    {
        ddc::ScopeGuard const guard;
        DoSetup();
        ::benchmark::RunSpecifiedBenchmarks();
    }
    ::benchmark::Shutdown();
    return 0;
}
