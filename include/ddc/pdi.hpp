// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <any>
#include <list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pdi.h>

#include "ddc/chunk_traits.hpp"
#include "ddc/discrete_vector.hpp"

namespace ddc {

template <class T>
static constexpr PDI_inout_t default_access_v
        = (std::is_lvalue_reference_v<T> && !std::is_const_v<std::remove_reference_t<T>>)
                  ? PDI_INOUT
                  : PDI_OUT;

template <class T>
static constexpr PDI_inout_t chunk_default_access_v = is_writable_chunk_v<T> ? PDI_INOUT : PDI_OUT;

class PdiEvent
{
    std::string m_event_name;

    std::vector<std::string> m_names;

    std::list<std::any> m_metadata;

    char const* store_name(std::string&& name)
    {
        return m_names.emplace_back(std::move(name)).c_str();
    }

    char const* store_name(std::string const& name)
    {
        return m_names.emplace_back(name).c_str();
    }

    template <class T>
    T* store_scalar(T t)
    {
        std::any& ref = m_metadata.emplace_back(std::in_place_type<T>, std::move(t));
        return std::any_cast<T>(&ref);
    }

    template <class T>
    T* store_array(std::vector<T> v)
    {
        std::any& ref = m_metadata.emplace_back(std::in_place_type<std::vector<T>>, std::move(v));
        return std::any_cast<std::vector<T>>(&ref)->data();
    }

public:
    explicit PdiEvent(std::string const& event_name) : m_event_name(event_name) {}

    PdiEvent(PdiEvent const& rhs) = delete;

    PdiEvent(PdiEvent&& rhs) noexcept = delete;

    ~PdiEvent() noexcept
    {
        PDI_event(m_event_name.c_str());
        for (std::string const& one_name : m_names) {
            PDI_reclaim(one_name.c_str());
        }
    }

    PdiEvent& operator=(PdiEvent const& rhs) = delete;

    PdiEvent& operator=(PdiEvent&& rhs) noexcept = delete;

    /// @{
    /// API with access argument

    template <
            PDI_inout_t access,
            class BorrowedChunk,
            std::enable_if_t<is_borrowed_chunk_v<BorrowedChunk>, int> = 0>
    PdiEvent& with(std::string const& name, BorrowedChunk&& data)
    {
        static_assert(
                !(access & PDI_IN) || (chunk_default_access_v<BorrowedChunk> & PDI_IN),
                "Invalid access for constant data");
        std::array const extents = detail::array(data.domain().extents());
        PDI_share(store_name(name + "_rank"), store_scalar(extents.size()), PDI_OUT);
        PDI_share(
                store_name(name + "_extents"),
                store_array(std::vector<std::size_t>(extents.begin(), extents.end())),
                PDI_OUT);
        PDI_share(
                store_name(name),
                const_cast<chunk_value_t<BorrowedChunk>*>(data.data_handle()),
                access);
        return *this;
    }

    template <
            PDI_inout_t access,
            class Arithmetic,
            std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<Arithmetic>>, int> = 0>
    PdiEvent& with(std::string const& name, Arithmetic&& data)
    {
        static_assert(
                !(access & PDI_IN) || (default_access_v<Arithmetic> & PDI_IN),
                "Invalid access for constant data");
        using value_type = std::remove_cv_t<std::remove_reference_t<Arithmetic>>;
        value_type* data_ptr = const_cast<value_type*>(&data);
        // for read-only data, we share a copy instead of the data itself in case we received a ref on a temporary,
        if constexpr (!(access & PDI_IN)) {
            data_ptr = store_scalar(data);
        }
        PDI_share(store_name(name), data_ptr, access);
        return *this;
    }

    /// @}
    /// API with access deduction
    /// @{

    /// Borrowed chunk overload (Chunk (const)& or ChunkSpan&& or ChunkSpan (const)&)
    template <class BorrowedChunk, std::enable_if_t<is_borrowed_chunk_v<BorrowedChunk>, int> = 0>
    PdiEvent& with(std::string const& name, BorrowedChunk&& data)
    {
        return with<chunk_default_access_v<BorrowedChunk>>(name, std::forward<BorrowedChunk>(data));
    }

    /// Arithmetic overload
    template <
            class Arithmetic,
            std::enable_if_t<std::is_arithmetic_v<std::remove_reference_t<Arithmetic>>, int> = 0>
    PdiEvent& with(std::string const& name, Arithmetic&& data)
    {
        return with<default_access_v<Arithmetic>>(name, std::forward<Arithmetic>(data));
    }

    /// @}
};

template <PDI_inout_t access, class DataType>
void expose_to_pdi(std::string const& name, DataType&& data)
{
    PdiEvent(name).with<access>(name, std::forward<DataType>(data));
}

template <class DataType>
void expose_to_pdi(std::string const& name, DataType&& data)
{
    PdiEvent(name).with(name, std::forward<DataType>(data));
}

} // namespace ddc
