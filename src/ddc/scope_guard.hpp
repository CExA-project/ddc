// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

namespace ddc {

class ScopeGuard
{
public:
    ScopeGuard();

    ScopeGuard(int argc, char**& argv);

    ScopeGuard(ScopeGuard const& x) = delete;

    ScopeGuard(ScopeGuard&& x) noexcept = delete;

    ~ScopeGuard() noexcept;

    ScopeGuard& operator=(ScopeGuard const& x) = delete;

    ScopeGuard& operator=(ScopeGuard&& x) noexcept = delete;
};

} // namespace ddc
