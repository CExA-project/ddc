// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

namespace ddc::detail {

#if defined(KOKKOS_ENABLE_CUDA)
using GlobalVariableDeviceSpace = Kokkos::CudaSpace;
#elif defined(KOKKOS_ENABLE_HIP)
using GlobalVariableDeviceSpace = Kokkos::HIPSpace;
#elif defined(KOKKOS_ENABLE_SYCL)
using GlobalVariableDeviceSpace = Kokkos::SYCLDeviceUSMSpace;
#endif

template <class DDim>
class DualDiscretization
{
    using DDimImplHost = typename DDim::template Impl<DDim, Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    using DDimImplDevice = typename DDim::template Impl<DDim, GlobalVariableDeviceSpace>;
#else
    using DDimImplDevice = DDimImplHost;
#endif

    DDimImplHost m_host;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    DDimImplDevice m_device_on_host;
#endif

public:
    template <class... Args>
    explicit DualDiscretization(Args&&... args)
        : m_host(std::forward<Args>(args)...)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        , m_device_on_host(m_host)
#endif
    {
    }

    template <class MemorySpace>
    typename DDim::template Impl<DDim, MemorySpace> const& get()
    {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
            return m_host;
        }
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        else if constexpr (std::is_same_v<MemorySpace, GlobalVariableDeviceSpace>) {
            return m_device_on_host;
        }
#elif defined(KOKKOS_ENABLE_SYCL)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::SYCLDeviceUSMSpace>) {
            return m_device_on_host;
        }
#endif
        else {
            static_assert(!std::is_same_v<MemorySpace, MemorySpace>);
        }
    }

    DDimImplHost const& get_host()
    {
        return m_host;
    }

    DDimImplDevice const& get_device()
    {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        return m_device_on_host;
#else
        return m_host;
#endif
    }
};

} // namespace ddc::detail
