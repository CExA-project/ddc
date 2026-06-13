// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <ddc/ddc.hpp>

#include <benchmark/benchmark.h>
#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <umpire/strategy/QuickPool.hpp>

#include <Kokkos_Core.hpp>

#include "ddc/discrete_vector.hpp"

namespace ddc {

template <typename MemorySpace>
umpire::Allocator get_default_umpire_allocator(MemorySpace /*memory_space*/)
{
    std::string resource;
    if (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        resource = "HOST";
    } else if (std::is_same_v<MemorySpace, Kokkos::SharedSpace>) {
        resource = "UM";
    } else {
        resource = "DEVICE";
    }
    return umpire::ResourceManager::getInstance().getAllocator(resource);
}

template <class T, class MemorySpace>
class UmpireAllocator
{
public:
    using value_type = T;

    using memory_space = MemorySpace;

    template <class U>
    struct rebind
    {
        using other = UmpireAllocator<U, MemorySpace>;
    };

    UmpireAllocator() : m_allocator(get_default_umpire_allocator(MemorySpace())) {}

    explicit UmpireAllocator(umpire::Allocator const& allocator) : m_allocator(allocator) {}

    UmpireAllocator(UmpireAllocator const& x) = default;

    UmpireAllocator(UmpireAllocator&& x) noexcept = default;

    template <class U>
    explicit UmpireAllocator(UmpireAllocator<U, MemorySpace> const& rhs) noexcept
        : m_allocator(rhs.m_allocator)
    {
    }

    ~UmpireAllocator() = default;

    UmpireAllocator& operator=(UmpireAllocator const& x) = default;

    UmpireAllocator& operator=(UmpireAllocator&& x) noexcept = default;

    template <class U>
    UmpireAllocator& operator=(UmpireAllocator<U, MemorySpace> const& rhs) noexcept
    {
        if (this == &rhs) {
            return *this;
        }
        m_allocator = rhs.m_allocator;
    }

    [[nodiscard]] T* allocate(std::size_t n)
    {
        void* const ptr = m_allocator.allocate(sizeof(T) * n);
        Kokkos::fence("after umpire allocate");
        return static_cast<T*>(ptr);
    }

    [[nodiscard]] T* allocate(std::string const& /*label*/, std::size_t n)
    {
        return allocate(n);
    }

    void deallocate(T* p, std::size_t)
    {
        Kokkos::fence("before umpire deallocate");
        m_allocator.deallocate(p);
    }

private:
    umpire::Allocator m_allocator;
};

template <class T, class MST, class U, class MSU>
bool operator==(UmpireAllocator<T, MST> const& lhs, UmpireAllocator<U, MSU> const& rhs) noexcept
{
    return lhs.m_allocator.getId() == rhs.m_allocator.getId();
}

template <class T>
using DeviceUmpireAllocator = UmpireAllocator<T, Kokkos::DefaultExecutionSpace::memory_space>;

template <class T>
using HostUmpireAllocator = UmpireAllocator<T, Kokkos::HostSpace>;

} // namespace ddc

inline namespace anonymous_namespace_workaround_repeated_allocations_cpp {

struct DDimX
{
};
using DElemX = ddc::DiscreteElement<DDimX>;
using DVectX = ddc::DiscreteVector<DDimX>;
using DDomX = ddc::DiscreteDomain<DDimX>;


using value_type = std::byte;
inline constexpr value_type value = value_type {};


void benchmark_ddc_kokkos_allocator(benchmark::State& state)
{
    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(state.range(0)));

    Kokkos::fence();
    for (auto _ : state) {
        ddc::Chunk chk_dst("chk_dst", domain_x, ddc::DeviceAllocator<value_type>());
        ddc::parallel_fill(chk_dst, value);
        Kokkos::fence();
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations())
            * static_cast<std::int64_t>(state.range(0)));
}

void benchmark_ddc_preallocated_kokkos_allocator(benchmark::State& state)
{
    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(state.range(0)));

    ddc::Chunk chk_dst("chk_dst", domain_x, ddc::DeviceAllocator<value_type>());
    Kokkos::fence();
    for (auto _ : state) {
        ddc::parallel_fill(chk_dst, value);
        Kokkos::fence();
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations())
            * static_cast<std::int64_t>(state.range(0)));
}

void benchmark_ddc_umpire_default_allocator(benchmark::State& state)
{
    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(state.range(0)));

    std::size_t alloc_size = state.range(0);

    Kokkos::fence();
    for (auto _ : state) {
        ddc::Chunk chk_dst("chk_dst", domain_x, ddc::DeviceUmpireAllocator<value_type>());
        ddc::parallel_fill(chk_dst, value);
        Kokkos::fence();
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(alloc_size));
}

void benchmark_ddc_umpire_pool_allocator(benchmark::State& state)
{
    ddc::DiscreteDomain<DDimX> const domain_x
            = ddc::init_trivial_bounded_space(ddc::DiscreteVector<DDimX>(state.range(0)));

    std::size_t alloc_size = state.range(0);

    umpire::Allocator const base_allocator
            = ddc::get_default_umpire_allocator(Kokkos::DefaultExecutionSpace::memory_space());

    umpire::Allocator pooled_allocator;
    std::string const allocator_name
            = base_allocator.getName() + "_pool" + std::to_string(alloc_size);
    umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();
    if (rm.isAllocator(allocator_name)) {
        pooled_allocator = rm.getAllocator(allocator_name);
    } else {
        pooled_allocator = rm.makeAllocator<umpire::strategy::QuickPool>(
                allocator_name,
                base_allocator,
                0,
                alloc_size,
                alignof(value_type));
    }

    Kokkos::fence();
    for (auto _ : state) {
        ddc::Chunk
                chk_dst("chk_dst",
                        domain_x,
                        ddc::DeviceUmpireAllocator<value_type>(pooled_allocator));
        ddc::parallel_fill(chk_dst, value);
        Kokkos::fence();
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations()) * static_cast<std::int64_t>(alloc_size));
}

std::size_t constexpr small_dim1_1D = 1'000'000;
std::size_t constexpr large_dim1_1D = 1'000'000'000;

} // namespace anonymous_namespace_workaround_repeated_allocations_cpp

// NOLINTBEGIN(misc-use-anonymous-namespace)
// BENCHMARK(benchmark_ddc_kokkos_allocator)
//         ->Name("benchmark_ddc_kokkos_allocator_small")
//         ->Arg(small_dim1_1D);
// BENCHMARK(benchmark_ddc_umpire_default_allocator)
//         ->Name("benchmark_ddc_umpire_default_allocator_small")
//         ->Arg(small_dim1_1D);

BENCHMARK(benchmark_ddc_kokkos_allocator)->Arg(small_dim1_1D)->Arg(large_dim1_1D);
BENCHMARK(benchmark_ddc_preallocated_kokkos_allocator)->Arg(small_dim1_1D)->Arg(large_dim1_1D);
BENCHMARK(benchmark_ddc_umpire_default_allocator)->Arg(small_dim1_1D)->Arg(large_dim1_1D);
BENCHMARK(benchmark_ddc_umpire_pool_allocator)->Arg(small_dim1_1D)->Arg(large_dim1_1D);
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
