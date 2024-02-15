// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>

namespace ddc {

/// Serial execution on the host
struct serial_host_policy
{
};

/// Parallel execution on the default device
struct parallel_host_policy
{
};

/// Kokkos parallel execution using MDRange policy
struct parallel_device_policy
{
};

using default_policy = serial_host_policy;

namespace policies {

inline constexpr serial_host_policy serial_host;
inline constexpr parallel_host_policy parallel_host;
inline constexpr parallel_device_policy parallel_device;

inline serial_host_policy const& policy(Kokkos::Serial)
{
    return serial_host;
}

#ifdef KOKKOS_ENABLE_CUDA
inline parallel_device_policy const& policy(Kokkos::Cuda)
{
    return parallel_device;
}
#endif

#ifdef KOKKOS_ENABLE_HIP
inline parallel_device_policy const& policy(Kokkos::HIP)
{
    return parallel_device;
}
#endif

#ifdef KOKKOS_ENABLE_OPENMP
inline parallel_host_policy const& policy(Kokkos::OpenMP)
{
    return parallel_host;
}
#endif

} // namespace policies

} // namespace ddc
