// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <pdi.h>

#include "pdi.hpp"

namespace ddc {

char const* PdiEvent::store_name(std::string&& name)
{
    return m_names.emplace_back(std::move(name)).c_str();
}

char const* PdiEvent::store_name(std::string const& name)
{
    return m_names.emplace_back(name).c_str();
}

PdiEvent::PdiEvent(std::string const& event_name) : m_event_name(event_name) {}

PdiEvent::~PdiEvent() noexcept
{
    PDI_event(m_event_name.c_str());
    for (std::string const& one_name : m_names) {
        PDI_reclaim(one_name.c_str());
    }
}

PdiEvent& PdiEvent::with(std::string const& name, char const* const c_string)
{
    PDI_share(store_name(name + "_size"), store_scalar(std::strlen(c_string)), PDI_OUT);
    PDI_share(store_name(name), c_string, PDI_OUT);
    return *this;
}

} // namespace ddc
