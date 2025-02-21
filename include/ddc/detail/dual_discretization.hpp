// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda.h>
#endif
#if defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace ddc::detail {

template <class DDim>
class DualDiscretization
{
    using DDimImplHost = typename DDim::template Impl<DDim, Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA)
    using DDimImplDevice = typename DDim::template Impl<DDim, Kokkos::CudaSpace>;
#elif defined(KOKKOS_ENABLE_HIP)
    using DDimImplDevice = typename DDim::template Impl<DDim, Kokkos::HIPSpace>;
#else
    using DDimImplDevice = DDimImplHost;
#endif

    DDimImplHost m_host;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    DDimImplDevice m_device_on_host;
#endif

public:
    template <class... Args>
    explicit DualDiscretization(Args&&... args)
        : m_host(std::forward<Args>(args)...)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
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
#if defined(KOKKOS_ENABLE_CUDA)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
            return m_device_on_host;
        }
#elif defined(KOKKOS_ENABLE_HIP)
        else if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
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
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
        return m_device_on_host;
#else
        return m_host;
#endif
    }
};

} // namespace ddc::detail
