// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>

#include <ddc/ddc.hpp>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

inline namespace anonymous_namespace_workaround_parallel_copy_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;


struct DDimY
{
};
using DElemY = ddc::DiscreteElement<DDimY>;
using DVectY = ddc::DiscreteVector<DDimY>;
using DDomY = ddc::DiscreteDomain<DDimY>;


using DElemXY = ddc::DiscreteElement<DDimX, DDimY>;
using DVectXY = ddc::DiscreteVector<DDimX, DDimY>;

using DDomXY = ddc::DiscreteDomain<DDimX, DDimY>;
using SDDomXY = ddc::SparseDiscreteDomain<DDimX, DDimY>;


void benchmark_ddc_manual_copy(benchmark::State& state)
{
    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(state.range(0)));
    ddc::DiscreteDomain<DDimY> const domain_y
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimY>(state.range(0)));

    Kokkos::View<ddc::DiscreteElement<DDimX>*, Kokkos::SharedSpace> const
            sdomain_x("", domain_x.size());
    ddc::host_for_each(domain_x, [=](ddc::DiscreteElement<DDimX> ix) {
        sdomain_x(static_cast<std::size_t>(ix - domain_x.front())) = ix;
    });

    Kokkos::View<ddc::DiscreteElement<DDimY>*, Kokkos::SharedSpace> const
            sdomain_y("", domain_y.size());
    ddc::host_for_each(domain_y, [=](ddc::DiscreteElement<DDimY> iy) {
        sdomain_y(static_cast<std::size_t>(iy - domain_y.front())) = iy;
    });

    ddc::SparseDiscreteDomain<DDimX, DDimY> const sddom_xy(sdomain_x, sdomain_y);
    ddc::Chunk chk_dst("chk_dst", sddom_xy, ddc::DeviceAllocator<int>());
    ddc::Chunk chk_src("chk_src", sddom_xy, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan const chk_span_dst = chk_dst.span_view();
    ddc::ChunkSpan const chk_span_src = chk_src.span_view();
    Kokkos::DefaultExecutionSpace const exec_space;

    for (auto _ : state) {
        ddc::parallel_for_each(
                exec_space,
                sddom_xy,
                KOKKOS_LAMBDA(ddc::DiscreteElement<DDimX, DDimY> ixy) {
                    chk_span_dst(ixy) = chk_span_src(ixy);
                });
        exec_space.fence();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations())
            * static_cast<std::int64_t>((chk_span_src.size() + chk_span_dst.size()) * sizeof(int)));
}

void benchmark_ddc_parallel_copy(benchmark::State& state)
{
    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(state.range(0)));
    ddc::DiscreteDomain<DDimY> const domain_y
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimY>(state.range(0)));

    Kokkos::View<ddc::DiscreteElement<DDimX>*, Kokkos::SharedSpace> const
            sdomain_x("", domain_x.size());
    ddc::host_for_each(domain_x, [=](ddc::DiscreteElement<DDimX> ix) {
        sdomain_x(static_cast<std::size_t>(ix - domain_x.front())) = ix;
    });

    Kokkos::View<ddc::DiscreteElement<DDimY>*, Kokkos::SharedSpace> const
            sdomain_y("", domain_y.size());
    ddc::host_for_each(domain_y, [=](ddc::DiscreteElement<DDimY> iy) {
        sdomain_y(static_cast<std::size_t>(iy - domain_y.front())) = iy;
    });

    ddc::SparseDiscreteDomain<DDimX, DDimY> const sddom_xy(sdomain_x, sdomain_y);
    ddc::Chunk chk_dst("chk_dst", sddom_xy, ddc::DeviceAllocator<int>());
    ddc::Chunk chk_src("chk_src", sddom_xy, ddc::DeviceAllocator<int>());
    ddc::ChunkSpan const chk_span_dst = chk_dst.span_view();
    ddc::ChunkSpan const chk_span_src = chk_src.span_view();
    Kokkos::DefaultExecutionSpace const exec_space;

    for (auto _ : state) {
        ddc::parallel_copy(exec_space, chk_span_dst, chk_span_src);
        exec_space.fence();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations())
            * static_cast<std::int64_t>((chk_span_src.size() + chk_span_dst.size()) * sizeof(int)));
}

void benchmark_kokkos_deepcopy(benchmark::State& state)
{
    Kokkos::View<int**> const src("src", state.range(0), state.range(0));
    Kokkos::View<int**> const dst("dst", state.range(0), state.range(0));
    Kokkos::DefaultExecutionSpace const exec_space;

    for (auto _ : state) {
        Kokkos::deep_copy(exec_space, dst, src);
        exec_space.fence();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations())
            * static_cast<std::int64_t>((src.size() + dst.size()) * sizeof(int)));
}

std::size_t constexpr small_dim1_1D = 2'000;
std::size_t constexpr large_dim1_1D = 10 * small_dim1_1D;

} // namespace anonymous_namespace_workaround_parallel_copy_cpp

// NOLINTBEGIN(misc-use-anonymous-namespace)
BENCHMARK(benchmark_ddc_manual_copy)->Arg(small_dim1_1D);
BENCHMARK(benchmark_ddc_parallel_copy)->Arg(small_dim1_1D);
BENCHMARK(benchmark_kokkos_deepcopy)->Arg(small_dim1_1D);
BENCHMARK(benchmark_ddc_manual_copy)->Arg(large_dim1_1D);
BENCHMARK(benchmark_ddc_parallel_copy)->Arg(large_dim1_1D);
BENCHMARK(benchmark_kokkos_deepcopy)->Arg(large_dim1_1D);
// NOLINTEND(misc-use-anonymous-namespace)

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, argv);
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
