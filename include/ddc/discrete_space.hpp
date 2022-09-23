// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#if defined(__CUDACC__)
#include <cuda.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "ddc/discrete_domain.hpp"
#include "ddc/discrete_space.hpp"
#include "ddc/dual_discretization.hpp"

namespace detail {

template <class DDim, class MemorySpace>
using ddim_impl_t = typename DDim::template Impl<MemorySpace>;

template <class T>
class gpu_proxy
{
    static_assert(std::is_standard_layout_v<T>, "Not standard layout");
    // Ideally T should be trivially destructible and copy constructible
    // static_assert(std::is_trivially_destructible_v<T>, "Not trivially destructible");
    // static_assert(std::is_trivially_copy_constructible_v<T>, "Not trivially copy-constructible");

private:
    alignas(T) std::byte m_data[sizeof(T)];

public:
    DDC_INLINE_FUNCTION
    T* operator->()
    {
        return reinterpret_cast<T*>(m_data);
    }

    DDC_INLINE_FUNCTION
    T& operator*()
    {
        return *reinterpret_cast<T*>(m_data);
    }

    DDC_INLINE_FUNCTION
    T* data()
    {
        return reinterpret_cast<T*>(m_data);
    }
};

// Global CPU variable storing resetters. Required to correctly free data.
inline std::optional<std::map<std::string, std::function<void()>>> g_discretization_store;

// Global CPU variable owning discrete spaces data for CPU and GPU
template <class DDim>
inline std::optional<DualDiscretization<DDim>> g_discrete_space_dual;

#if defined(__CUDACC__)
// Global GPU variable viewing data owned by the CPU
template <class DDim>
__constant__ gpu_proxy<ddim_impl_t<DDim, Kokkos::CudaSpace>> g_discrete_space_device;
#elif defined(__HIPCC__)
// Global GPU variable viewing data owned by the CPU
// WARNING: do not put the `inline` keyword, seems to fail on MI100 rocm/4.5.0
template <class DDim>
__constant__ gpu_proxy<ddim_impl_t<DDim, Kokkos::Experimental::HIPSpace>> g_discrete_space_device;
#endif

template <class DDim, class MemorySpace>
struct DiscreteSpaceImpl;

template <class DDim>
struct DiscreteSpaceImpl<DDim, Kokkos::HostSpace>
{
    static inline ddim_impl_t<DDim, Kokkos::HostSpace> const& get()
    {
        return g_discrete_space_dual<DDim>->get_host();
    }
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <class DDim, class MemorySpace>
struct DiscreteSpaceImpl
{
    DDC_INLINE_FUNCTION
    static ddim_impl_t<DDim, MemorySpace> const& get()
    {
        return *g_discrete_space_device<DDim>;
    }
};
#endif

inline void display_discretization_store(std::ostream& os)
{
    if (g_discretization_store) {
        os << "The host discretization store is initialized:\n";
        for (auto const& [key, value] : *g_discretization_store) {
            os << " - " << key << "\n";
        }
    } else {
        os << "The host discretization store is not initialized:\n";
    }
}

template <class Tuple, std::size_t... Ids>
auto extract_after(Tuple&& t, std::index_sequence<Ids...>)
{
    return std::make_tuple(std::move(std::get<Ids + 1>(t))...);
}

} // namespace detail

/** Initialize (emplace) a global singleton discrete space
 * 
 * @param a the constructor arguments
 */
template <class DDim, class... Args>
void init_discrete_space(Args&&... args)
{
    if (detail::g_discrete_space_dual<DDim>) {
        throw std::runtime_error("Discrete space function already initialized.");
    }
    detail::g_discrete_space_dual<DDim>.emplace(std::forward<Args>(args)...);
    detail::g_discretization_store->emplace(typeid(DDim).name(), []() {
        detail::g_discrete_space_dual<DDim>.reset();
    });
#if defined(__CUDACC__)
    cudaMemcpyToSymbol(
            detail::g_discrete_space_device<DDim>,
            &detail::g_discrete_space_dual<DDim>->get_device(),
            sizeof(detail::g_discrete_space_dual<DDim>->get_device()));
#elif defined(__HIPCC__)
    hipMemcpyToSymbol(
            detail::g_discrete_space_device<DDim>,
            &detail::g_discrete_space_dual<DDim>->get_device(),
            sizeof(detail::g_discrete_space_dual<DDim>->get_device()));
#endif
}

/** Move construct a global singleton discrete space and pass through the other argument
 * 
 * @param a - the discrete space to move at index 0
 *          - the arguments to pass through at index 1
 */
template <class DDimImpl, class Arg>
Arg init_discrete_space(std::tuple<DDimImpl, Arg>&& a)
{
    using DDim = typename DDimImpl::discrete_dimension_type;
    init_discrete_space<DDim>(std::move(std::get<0>(a)));
    return std::get<1>(a);
}

/** Move construct a global singleton discrete space and pass through remaining arguments
 * 
 * @param a - the discrete space to move at index 0
 *          - the (2+) arguments to pass through in other indices
 */
template <class DDimImpl, class... Args>
std::enable_if_t<2 <= sizeof...(Args), std::tuple<Args...>> init_discrete_space(
        std::tuple<DDimImpl, Args...>&& a)
{
    using DDim = typename DDimImpl::discrete_dimension_type;
    init_discrete_space<DDim>(std::move(std::get<0>(a)));
    return detail::extract_after(std::move(a), std::index_sequence_for<Args...>());
}

template <class DDim, class MemorySpace = DDC_CURRENT_KOKKOS_SPACE>
DDC_INLINE_FUNCTION detail::ddim_impl_t<DDim, MemorySpace> const& discrete_space()
{
    return detail::DiscreteSpaceImpl<DDim, MemorySpace>::get();
}
