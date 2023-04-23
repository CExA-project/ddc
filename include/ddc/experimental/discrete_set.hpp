// SPDX-License-Identifier: MIT

#pragma once

#include "ddc/discrete_space.hpp"

namespace ddc::experimental {

// Global CPU variable owning discrete spaces data for CPU and GPU
template <class NamedDSet>
inline std::optional<DualDiscretization<NamedDSet>> g_discrete_space_dual;

#if defined(__CUDACC__)
// Global GPU variable viewing data owned by the CPU
template <class NamedDSet>
__constant__ detail::gpu_proxy<detail::ddim_impl_t<NamedDSet, Kokkos::CudaSpace>>
        g_discrete_space_device;
#elif defined(__HIPCC__)
// Global GPU variable viewing data owned by the CPU
// WARNING: do not put the `inline` keyword, seems to fail on MI100 rocm/4.5.0
template <class NamedDSet>
__constant__ detail::gpu_proxy<detail::ddim_impl_t<NamedDSet, Kokkos::Experimental::HIPSpace>>
        g_discrete_space_device;
#endif

/** Initialize (emplace) a global singleton discrete space
 * 
 * @param args the constructor arguments
 */
template <class NamedDSet, class... Args>
void init_discrete_set(Args&&... args)
{
    if (g_discrete_space_dual<NamedDSet>) {
        throw std::runtime_error("Discrete space function already initialized.");
    }
    g_discrete_space_dual<NamedDSet>.emplace(std::forward<Args>(args)...);
    detail::g_discretization_store->emplace(typeid(NamedDSet).name(), []() {
        g_discrete_space_dual<NamedDSet>.reset();
    });
#if defined(__CUDACC__)
    cudaMemcpyToSymbol(
            g_discrete_space_device<NamedDSet>,
            &g_discrete_space_dual<NamedDSet>->get_device(),
            sizeof(g_discrete_space_dual<NamedDSet>->get_device()));
#elif defined(__HIPCC__)
    hipMemcpyToSymbol(
            g_discrete_space_device<NamedDSet>,
            &g_discrete_space_dual<NamedDSet>->get_device(),
            sizeof(g_discrete_space_dual<NamedDSet>->get_device()));
#endif
}

/** Move construct a global singleton discrete space and pass through the other argument
 *
 * @param a - the discrete space to move at index 0
 *          - the arguments to pass through at index 1
 */
template <class NamedDSet, class DSetImpl, class Arg>
Arg init_discrete_set(std::tuple<DSetImpl, Arg>&& a)
{
    init_discrete_set<NamedDSet>(std::move(std::get<0>(a)));
    return std::get<1>(a);
}

/** Move construct a global singleton discrete space and pass through remaining arguments
 *
 * @param a - the discrete space to move at index 0
 *          - the (2+) arguments to pass through in other indices
 */
template <class NamedDSet, class DSetImpl, class... Args>
std::enable_if_t<2 <= sizeof...(Args), std::tuple<Args...>> init_discrete_set(
        std::tuple<DSetImpl, Args...>&& a)
{
    init_discrete_set<NamedDSet>(std::move(std::get<0>(a)));
    return detail::extract_after(std::move(a), std::index_sequence_for<Args...>());
}

template <class NamedDSet, class MemorySpace = DDC_CURRENT_KOKKOS_SPACE>
DDC_INLINE_FUNCTION detail::ddim_impl_t<NamedDSet, MemorySpace> const& discrete_set()
{
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        return g_discrete_space_dual<NamedDSet>->get_host();
    }
#if defined(__CUDACC__)
    else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
        return *detail::g_discrete_space_device<NamedDSet>;
    }
#elif defined(__HIPCC__)
    else if constexpr (std::is_same_v<MemorySpace, Kokkos::Experimental::HIPSpace>) {
        return *detail::g_discrete_space_device<NamedDSet>;
    }
#endif
    else {
        static_assert(std::is_same_v<MemorySpace, MemorySpace>, "Memory space not handled");
    }
}

} // namespace ddc::experimental
