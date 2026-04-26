// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "discrete_space.hpp"
#include "scope_guard.hpp"

namespace {

void discretization_store_initialization()
{
    ddc::detail::g_discretization_store
            = std::make_optional<std::map<std::string, std::function<void()>>>();
}

} // namespace

namespace ddc {

ScopeGuard::ScopeGuard()
{
    discretization_store_initialization();
}

ScopeGuard::ScopeGuard(int /*argc*/, char**& /*argv*/) : ScopeGuard() {}

ScopeGuard::~ScopeGuard() noexcept
{
    for (auto const& [name, fn] : *detail::g_discretization_store) {
        fn();
    }
    detail::g_discretization_store.reset();
}

} // namespace ddc
