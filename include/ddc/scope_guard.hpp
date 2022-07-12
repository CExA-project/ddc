#pragma once

#include <map>
#include <memory>

#include <Kokkos_Core.hpp>

#include "discrete_space.hpp"

class ScopeGuard
{
    Kokkos::ScopeGuard m_kokkos_scope_guard;

    void discretization_store_initialization() const
    {
        detail::g_host_discretization_store = std::make_unique<std::map<std::string, std::any>>();
#if defined(__CUDACC__)
        detail::g_cuda_discretization_store = std::make_unique<std::map<std::string, void*>>();
#elif defined(__HIPCC__)
        detail::g_hip_discretization_store = std::make_unique<std::map<std::string, void*>>();
#endif
    }

public:
    ScopeGuard()
    {
        discretization_store_initialization();
    }

    ScopeGuard(int argc, char**& argv) : m_kokkos_scope_guard(argc, argv)
    {
        discretization_store_initialization();
    }

    ScopeGuard(ScopeGuard const& x) = delete;

    ScopeGuard(ScopeGuard&& x) noexcept = delete;

    ~ScopeGuard() noexcept
    {
        detail::g_host_discretization_store = nullptr;
#if defined(__CUDACC__)
        for (auto const& [key, value] : *detail::g_cuda_discretization_store) {
            cudaFree(value);
        }
        detail::g_cuda_discretization_store = nullptr;
#elif defined(__HIPCC__)
        for (auto const& [key, value] : *detail::g_hip_discretization_store) {
            hipFree(value);
        }
        detail::g_hip_discretization_store = nullptr;
#endif
    }

    ScopeGuard& operator=(ScopeGuard const& x) = delete;

    ScopeGuard& operator=(ScopeGuard&& x) noexcept = delete;
};
