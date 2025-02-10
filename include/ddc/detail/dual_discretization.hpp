// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#if defined(__CUDACC__)
#include <cuda.h>
#endif
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace ddc::detail {

template <class DDim>
class DualDiscretization
{
    using DDimImplHost = typename DDim::template Impl<DDim, Kokkos::HostSpace>;
#if defined(__CUDACC__)
    using DDimImplDevice = typename DDim::template Impl<DDim, Kokkos::CudaSpace>;
#elif defined(__HIPCC__)
    using DDimImplDevice = typename DDim::template Impl<DDim, Kokkos::HIPSpace>;
#elif defined(KOKKOS_ENABLE_SYCL)
    using DDimImplDevice = typename DDim::template Impl<DDim, Kokkos::SYCLDeviceUSMSpace>;
#else
    using DDimImplDevice = DDimImplHost;
#endif

    DDimImplHost m_host;
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(KOKKOS_ENABLE_SYCL)
    DDimImplDevice m_device_on_host;
#endif

public:
    template <class... Args>
    explicit DualDiscretization(Args&&... args)
        : m_host(std::forward<Args>(args)...)
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(KOKKOS_ENABLE_SYCL)
        , m_device_on_host(m_host)
#endif
    {
    }

    template <class MemorySpace>
    KOKKOS_FUNCTION typename DDim::template Impl<DDim, MemorySpace> const& get()
    {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
            return m_host;
        }
#if defined(__CUDACC__)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
            return m_device_on_host;
        }
#elif defined(__HIPCC__)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
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

    KOKKOS_FUNCTION DDimImplHost const& get_host()
    {
        return m_host;
    }

    KOKKOS_FUNCTION DDimImplDevice const& get_device()
    {
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(KOKKOS_ENABLE_SYCL)
        return m_device_on_host;
#else
        return m_host;
#endif
    }
};

} // namespace ddc::detail
