// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "chunk_span.hpp"
#include "discrete_vector.hpp"

#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
#    include <cxxabi.h>
#endif

namespace ddc {
namespace detail {
class ChunkPrinter
{
    static constexpr int const s_threshold = 10;
    // If this ever becomes modifiable by the user, we need to ensure that
    // s_edgeitems < (s_threshold / 2) stays true.
    static constexpr int const s_edgeitems = 3;

    std::stringstream m_ss;

    static std::ostream& alignment(std::ostream& os, int level)
    {
        for (int i = 0; i <= level; ++i) {
            os << ' ';
        }
        return os;
    }

    template <class T>
    std::size_t get_element_width(T const& elem)
    {
        m_ss.seekp(0);
        m_ss << elem;
        return m_ss.tellp();
    }

    template <class T>
    void display_aligned_element(std::ostream& os, T const& elem, std::size_t largest_element)
    {
        std::size_t const elem_width = get_element_width(elem);

        for (std::size_t i = 0; i < largest_element - elem_width; ++i) {
            os << " ";
        }
        os << elem;
    }

    template <class ElementType, class Extents, class Layout, class Accessor>
    std::ostream& base_case_display(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& s,
            std::size_t largest_element,
            std::size_t beginning,
            std::size_t end,
            std::size_t extent)
    {
        for (std::size_t i0 = beginning; i0 < end; ++i0) {
            display_aligned_element(os, s[i0], largest_element);
            if (i0 < extent - 1) {
                os << " ";
            }
        }
        return os;
    }

    template <class ElementType, class Extents, class Layout, class Accessor, std::size_t... Is>
    std::ostream& recursive_display(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& s,
            int level,
            std::size_t largest_element,
            std::size_t beginning,
            std::size_t end,
            std::index_sequence<Is...>)
    {
        for (std::size_t i0 = beginning; i0 < end; ++i0) {
            print_impl(
                    os,
                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                    level + 1,
                    largest_element,
                    std::make_index_sequence<sizeof...(Is)>());
            if (i0 < end - 1) {
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << '\n';
                }
                alignment(os, level);
            }
        }

        return os;
    }

public:
    // 0D chunk span
    template <class ElementType, class Extents, class Layout, class Accessor>
    std::ostream& print_impl(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& s,
            int /*level*/,
            std::size_t /*largest_element*/,
            std::index_sequence<>)
    {
        return os << *s.data_handle();
    }

    // Recursively parse the chunk to print it
    template <
            class ElementType,
            class Extents,
            class Layout,
            class Accessor,
            std::size_t I0,
            std::size_t... Is>
    std::ostream& print_impl(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& s,
            int level,
            std::size_t largest_element,
            std::index_sequence<I0, Is...>)
    {
        auto extent = s.extent(I0);
        if constexpr (sizeof...(Is) > 0) {
            os << '[';
            if (extent < s_threshold) {
                recursive_display(
                        os,
                        s,
                        level,
                        largest_element,
                        0,
                        extent,
                        std::make_index_sequence<sizeof...(Is)>());
            } else {
                recursive_display(
                        os,
                        s,
                        level,
                        largest_element,
                        0,
                        s_edgeitems,
                        std::make_index_sequence<sizeof...(Is)>());
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << '\n';
                }
                alignment(os, level);
                os << "...";
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << '\n';
                }
                alignment(os, level);
                recursive_display(
                        os,
                        s,
                        level,
                        largest_element,
                        extent - s_edgeitems,
                        extent,
                        std::make_index_sequence<sizeof...(Is)>());
            }
            os << "]";
        } else {
            os << "[";
            if (extent < s_threshold) {
                base_case_display(os, s, largest_element, 0, extent, extent);
            } else {
                base_case_display(os, s, largest_element, 0, s_edgeitems, extent);
                os << "... ";
                base_case_display(os, s, largest_element, extent - s_edgeitems, extent, extent);
            }
            os << "]";
        }

        return os;
    }

    // 0D, we don't need the element size in this case so the actual returned value can be anything.
    template <class ElementType, class Extents, class Layout, class Accessor>
    std::size_t find_largest_displayed_element(
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const&,
            std::index_sequence<>)
    {
        return 0;
    }

    // Find the largest element we have to print to allow alignment (it ignore
    // element that will be elided).
    template <
            class ElementType,
            class Extents,
            class Layout,
            class Accessor,
            std::size_t I0,
            std::size_t... Is>
    std::size_t find_largest_displayed_element(
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& s,
            std::index_sequence<I0, Is...>)
    {
        std::size_t ret = 0;
        auto extent = s.extent(I0);
        if constexpr (sizeof...(Is) > 0) {
            if (extent < s_threshold) {
                for (std::size_t i0 = 0; i0 < extent; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
            } else {
                for (std::size_t i0 = 0; i0 < s_edgeitems; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
                for (std::size_t i0 = extent - s_edgeitems; i0 < extent; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
            }
        } else {
            if (extent < s_threshold) {
                for (std::size_t i0 = 0; i0 < extent; ++i0) {
                    ret = std::max(ret, get_element_width(s[i0]));
                }
            } else {
                for (std::size_t i0 = 0; i0 < s_edgeitems; ++i0) {
                    ret = std::max(ret, get_element_width(s[i0]));
                }
                for (std::size_t i0 = extent - s_edgeitems; i0 < extent; ++i0) {
                    ret = std::max(ret, get_element_width(s[i0]));
                }
            }
        }

        return ret;
    }

    explicit ChunkPrinter(std::ostream const& os)
    {
        m_ss.copyfmt(os);
    }
};

inline void print_demangled_type_name(std::ostream& os, char const* const mangled_name)
{
#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
    int status;

    std::unique_ptr<char, decltype(std::free)*> const
            demangled_name(abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status), std::free);
    if (status != 0) {
        os << "Error demangling dimension name: " << status;
        return;
    }

    os << demangled_name.get();
#else
    os << mangled_name;
#endif
}

inline void print_single_dim_name(
        std::ostream& os,
        std::type_info const& dim,
        DiscreteVectorElement const size)
{
    print_demangled_type_name(os, dim.name());
    os << '(' << size << ')';
}

inline void print_dim_name(std::ostream& os, DiscreteVector<> const&)
{
    os << "Scalar";
}

template <class Dim0, class... Dims>
void print_dim_name(std::ostream& os, DiscreteVector<Dim0, Dims...> const& dd)
{
    print_single_dim_name(os, typeid(Dim0), get<Dim0>(dd));
    ((os << "×", print_single_dim_name(os, typeid(Dims), get<Dims>(dd))), ...);
}

} // namespace detail

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print_content(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    auto h_chunk_span = create_mirror_view_and_copy(Kokkos::HostSpace(), chunk_span);

    using chunkspan_type = std::remove_cv_t<std::remove_reference_t<decltype(h_chunk_span)>>;
    using mdspan_type = typename chunkspan_type::allocation_mdspan_type;
    using extents = typename mdspan_type::extents_type;

    mdspan_type const allocated_mdspan = h_chunk_span.allocation_mdspan();

    ddc::detail::ChunkPrinter printer(os);
    std::size_t const largest_element = printer.find_largest_displayed_element(
            allocated_mdspan,
            std::make_index_sequence<extents::rank()>());

    printer.print_impl(
            os,
            allocated_mdspan,
            0,
            largest_element,
            std::make_index_sequence<extents::rank()>());

    return os;
}

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print_type_info(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    ddc::detail::print_dim_name(os, chunk_span.extents());
    os << '\n';
    ddc::detail::print_demangled_type_name(os, typeid(chunk_span).name());
    os << '\n';

    return os;
}

template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    print_type_info(os, chunk_span);
    print_content(os, chunk_span);

    return os;
}


template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& operator<<(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    return print(os, chunk_span);
}

} // namespace ddc
