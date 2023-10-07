#pragma once

#include <map>
#include <memory>
#include <optional>

#include <Kokkos_Core.hpp>
#include <signal.h>

#include "discrete_space.hpp"

namespace ddc {

class ScopeGuard
{
    Kokkos::ScopeGuard m_kokkos_scope_guard;

    void discretization_store_initialization() const
    {
        detail::g_discretization_store
                = std::make_optional<std::map<std::string, std::function<void()>>>();
    }
    static void sigintHandler(int signum)
    {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
        signal(SIGINT, SIG_DFL);
        raise(SIGINT);
    }

public:
    ScopeGuard()
    {
        discretization_store_initialization();
    }

    ScopeGuard(int argc, char**& argv) : m_kokkos_scope_guard(argc, argv)
    {
        discretization_store_initialization();
        signal(SIGINT, sigintHandler);
    }

    ScopeGuard(ScopeGuard const& x) = delete;

    ScopeGuard(ScopeGuard&& x) noexcept = delete;

    ~ScopeGuard() noexcept
    {
        for (auto const& [name, fn] : *detail::g_discretization_store) {
            fn();
        }
        detail::g_discretization_store.reset();
    }

    ScopeGuard& operator=(ScopeGuard const& x) = delete;

    ScopeGuard& operator=(ScopeGuard&& x) noexcept = delete;
};

} // namespace ddc
