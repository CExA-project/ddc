// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail/dual_discretization.hpp"
#include "detail/macros.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#    include <sstream>

#    include <cuda.h>
#elif defined(KOKKOS_ENABLE_HIP)
#    include <sstream>

#    include <hip/hip_runtime.h>
#endif

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#    define DDC_DETAIL_DEVICE_THROW_ON_ERROR(val)                                                  \
        ddc::detail::device_throw_on_error((val), #val, __FILE__, __LINE__)
#endif

namespace ddc {

namespace detail {

#if defined(KOKKOS_ENABLE_CUDA)
inline void device_throw_on_error(
        cudaError_t const err,
        const char* const func,
        const char* const file,
        const int line)
{
    if (err != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA Runtime Error at: " << file << ":" << line << "\n";
        ss << cudaGetErrorString(err) << " " << func << "\n";
        throw std::runtime_error(ss.str());
    }
}
#elif defined(KOKKOS_ENABLE_HIP)
inline void device_throw_on_error(
        hipError_t const err,
        const char* const func,
        const char* const file,
        const int line)
{
    if (err != hipSuccess) {
        std::stringstream ss;
        ss << "HIP Runtime Error at: " << file << ":" << line << "\n";
        ss << hipGetErrorString(err) << " " << func << "\n";
        throw std::runtime_error(ss.str());
    }
}
#endif

template <class DDim, class MemorySpace>
using ddim_impl_t = typename DDim::template Impl<DDim, MemorySpace>;

template <class T>
class gpu_proxy
{
    // Here are some reasonable concepts that T should satisfy to avoid undefined behaviors:
    // - copy-constructible: objects may be memcopied to the device,
    // - standard layout: objects will be ensured to have the same, standard, representation on the host and the device,
    // - trivially destructible: the destructor of objects located on a device may not be called.
    // static_assert(std::is_standard_layout_v<T>, "Not standard layout");
    // static_assert(std::is_trivially_destructible_v<T>, "Not trivially destructible");
    // static_assert(std::is_trivially_copy_constructible_v<T>, "Not trivially copy-constructible");
    // Currently not trivially destructible because for example of the Kokkos::View (mostly reference-counting)
    // Currently not trivially copy-constructible because of discrete spaces that have deleted copy-constructors and Kokkos::View (mostly reference-counting)

private:
    alignas(T) Kokkos::Array<std::byte, sizeof(T)> m_data;

public:
    KOKKOS_FUNCTION
    T* operator->()
    {
        return reinterpret_cast<T*>(m_data.data());
    }

    KOKKOS_FUNCTION
    T& operator*()
    {
        return *reinterpret_cast<T*>(m_data.data());
    }

    KOKKOS_FUNCTION
    T* data()
    {
        return reinterpret_cast<T*>(m_data.data());
    }
};

// Global CPU variable storing resetters. Required to correctly free data.
inline std::optional<std::map<std::string, std::function<void()>>> g_discretization_store;

// Global CPU variable owning discrete spaces data for CPU and GPU
template <class DDim>
inline std::optional<DualDiscretization<DDim>> g_discrete_space_dual;

#if defined(KOKKOS_ENABLE_CUDA)
// Global GPU variable viewing data owned by the CPU
template <class DDim>
__constant__ gpu_proxy<ddim_impl_t<DDim, GlobalVariableDeviceSpace>> g_discrete_space_device;
#elif defined(KOKKOS_ENABLE_HIP)
// Global GPU variable viewing data owned by the CPU
// WARNING: do not put the `inline` keyword, seems to fail on MI100 rocm/4.5.0
template <class DDim>
__constant__ gpu_proxy<ddim_impl_t<DDim, GlobalVariableDeviceSpace>> g_discrete_space_device;
#elif defined(KOKKOS_ENABLE_SYCL)
// Global GPU variable viewing data owned by the CPU
template <class DDim>
SYCL_EXTERNAL inline sycl::ext::oneapi::experimental::device_global<
        gpu_proxy<ddim_impl_t<DDim, GlobalVariableDeviceSpace>>>
        g_discrete_space_device;
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
 * @param args the constructor arguments
 */
template <class DDim, class... Args>
void init_discrete_space(Args&&... args)
{
    static_assert(
            !std::is_same_v<DDim, typename DDim::discrete_dimension_type>,
            "Discrete dimensions should inherit from the discretization, not use an alias");
    if (detail::g_discrete_space_dual<DDim>) {
        throw std::runtime_error("Discrete space function already initialized.");
    }
    detail::g_discrete_space_dual<DDim>.emplace(std::forward<Args>(args)...);
    detail::g_discretization_store->emplace(typeid(DDim).name(), []() {
        detail::g_discrete_space_dual<DDim>.reset();
    });
#if defined(KOKKOS_ENABLE_CUDA)
    DDC_DETAIL_DEVICE_THROW_ON_ERROR(cudaMemcpyToSymbol(
            detail::g_discrete_space_device<DDim>,
            &detail::g_discrete_space_dual<DDim>->get_device(),
            sizeof(detail::g_discrete_space_dual<DDim>->get_device())));
#elif defined(KOKKOS_ENABLE_HIP)
    DDC_DETAIL_DEVICE_THROW_ON_ERROR(hipMemcpyToSymbol(
            detail::g_discrete_space_device<DDim>,
            &detail::g_discrete_space_dual<DDim>->get_device(),
            sizeof(detail::g_discrete_space_dual<DDim>->get_device())));
#elif defined(KOKKOS_ENABLE_SYCL)
    Kokkos::DefaultExecutionSpace exec;
    sycl::queue q = exec.sycl_queue();
    q.memcpy(detail::g_discrete_space_device<DDim>,
             &detail::g_discrete_space_dual<DDim>->get_device())
            .wait();
#endif
}

/** Move construct a global singleton discrete space and pass through the other argument
 *
 * @param a - the discrete space to move at index 0
 *          - the arguments to pass through at index 1
 */
template <class DDim, class DDimImpl, class Arg0>
Arg0 init_discrete_space(std::tuple<DDimImpl, Arg0>&& a)
{
    init_discrete_space<DDim>(std::move(std::get<0>(a)));
    return std::get<1>(a);
}

/** Move construct a global singleton discrete space and pass through remaining arguments
 *
 * @param a - the discrete space to move at index 0
 *          - the (2+) arguments to pass through in other indices
 */
template <class DDim, class DDimImpl, class Arg0, class Arg1, class... Args>
std::tuple<Arg0, Arg1, Args...> init_discrete_space(std::tuple<DDimImpl, Arg0, Arg1, Args...>&& a)
{
    init_discrete_space<DDim>(std::move(std::get<0>(a)));
    return detail::extract_after(std::move(a), std::index_sequence_for<Arg0, Arg1, Args...>());
}

/**
 * @tparam DDim a discrete dimension
 * @return a boolean indicating whether DDim is initialized.
 * This function indicates whether a dimension is initialized.
 */
template <class DDim>
bool is_discrete_space_initialized() noexcept
{
    return detail::g_discrete_space_dual<DDim>.has_value();
}

/**
 * @tparam DDim a discrete dimension
 * @return the discrete space instance associated with `DDim`.
 * This function must be called from a `KOKKOS_FUNCTION`.
 * Call `ddc::host_discrete_space` for a host-only function instead.
 */
template <class DDim, class MemorySpace = DDC_CURRENT_KOKKOS_SPACE>
KOKKOS_FUNCTION detail::ddim_impl_t<DDim, MemorySpace> const& discrete_space()
{
    // This function requires that `ddc::init_discrete_space<DDim>(...);` be called first
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        assert(is_discrete_space_initialized<DDim>());
        return detail::g_discrete_space_dual<DDim>->get_host();
    }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    else if constexpr (std::is_same_v<MemorySpace, detail::GlobalVariableDeviceSpace>) {
        return *detail::g_discrete_space_device<DDim>;
    }
#elif defined(KOKKOS_ENABLE_SYCL)
    else if constexpr (std::is_same_v<MemorySpace, detail::GlobalVariableDeviceSpace>) {
        return *detail::g_discrete_space_device<DDim>.get();
    }
#endif
    else {
        static_assert(std::is_same_v<MemorySpace, MemorySpace>, "Memory space not handled");
    }
}

template <class DDim>
detail::ddim_impl_t<DDim, Kokkos::HostSpace> const& host_discrete_space()
{
    // This function requires that `ddc::init_discrete_space<DDim>(...);` be called first
    assert(is_discrete_space_initialized<DDim>());
    return detail::g_discrete_space_dual<DDim>->get_host();
}

} // namespace ddc

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#    undef DDC_DETAIL_DEVICE_THROW_ON_ERROR
#endif
