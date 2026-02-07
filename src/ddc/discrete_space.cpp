// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <functional>
#include <map>
#include <optional>
#include <ostream>
#include <string>

#include <Kokkos_Core.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#    include <sstream>

#    include <cuda.h>
#elif defined(KOKKOS_ENABLE_HIP)
#    include <sstream>

#    include <hip/hip_runtime.h>
#endif

namespace ddc::detail {

#if defined(KOKKOS_ENABLE_CUDA)
void device_throw_on_error(
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
void device_throw_on_error(
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

// Global CPU variable storing resetters. Required to correctly free data.
std::optional<std::map<std::string, std::function<void()>>> g_discretization_store;

void display_discretization_store(std::ostream& os)
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

} // namespace ddc::detail
