// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>

namespace ddc {

class ScopeGuard
{
public:
    ScopeGuard() {}

    ScopeGuard([[maybe_unused]] int argc, [[maybe_unused]] char**& argv) {}

    ScopeGuard(ScopeGuard const& x) = delete;

    ScopeGuard(ScopeGuard&& x) noexcept = delete;

    ~ScopeGuard() noexcept {}

    ScopeGuard& operator=(ScopeGuard const& x) = delete;

    ScopeGuard& operator=(ScopeGuard&& x) noexcept = delete;
};

} // namespace ddc
