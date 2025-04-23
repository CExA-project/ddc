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
#include <utility>

#include "ddc/chunk_span.hpp"
#include "ddc/discrete_domain.hpp"

#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
#    include <cxxabi.h>
#endif

namespace ddc {
namespace detail {
class ChunkPrinter
{
    static constexpr int const threshold = 10;
    // If this ever becomes modifiable by the user, we need to ensure that
    // edgeitems < (threshold / 2) stays true.
    static constexpr int const edgeitems = 3;

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
            if (extent < threshold) {
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
                        edgeitems,
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
                        extent - edgeitems,
                        extent,
                        std::make_index_sequence<sizeof...(Is)>());
            }
            os << "]";
        } else {
            os << "[";
            if (extent < threshold) {
                base_case_display(os, s, largest_element, 0, extent, extent);
            } else {
                base_case_display(os, s, largest_element, 0, edgeitems, extent);
                os << "... ";
                base_case_display(os, s, largest_element, extent - edgeitems, extent, extent);
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

    // Find the largest element we have to print to allow alignement (it ignore
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
            if (extent < threshold) {
                for (std::size_t i0 = 0; i0 < extent; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
            } else {
                for (std::size_t i0 = 0; i0 < edgeitems; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
                for (std::size_t i0 = extent - edgeitems; i0 < extent; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(s, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
            }
        } else {
            if (extent < threshold) {
                for (std::size_t i0 = 0; i0 < extent; ++i0) {
                    ret = std::max(ret, get_element_width(s[i0]));
                }
            } else {
                for (std::size_t i0 = 0; i0 < edgeitems; ++i0) {
                    ret = std::max(ret, get_element_width(s[i0]));
                }
                for (std::size_t i0 = extent - edgeitems; i0 < extent; ++i0) {
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

#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
template <class Type>
void print_demangled_type_name(std::ostream& os)
{
    int status;

    std::unique_ptr<char, decltype(std::free)*> const demangled_name(
            abi::__cxa_demangle(typeid(Type).name(), nullptr, nullptr, &status),
            std::free);
    if (status != 0) {
        os << "Error demangling dimension name: " << status;
        return;
    }

    os << demangled_name.get();
}
#else
template <class Type>
void print_demangled_type_name(std::ostream& os)
{
    os << typeid(Type).name();
}
#endif

inline void print_dim_name(std::ostream& os, DiscreteDomain<> const)
{
    os << "Scalar";
}

template <class Dim>
void print_dim_name(std::ostream& os, DiscreteDomain<Dim> const dd)
{
    print_demangled_type_name<Dim>(os);
    os << '(' << dd.size() << ')';
}

template <class Dim0, class Dim1, class... Dims>
void print_dim_name(std::ostream& os, DiscreteDomain<Dim0, Dim1, Dims...> const dd)
{
    print_demangled_type_name<Dim0>(os);
    DiscreteDomain<Dim1, Dims...> const smaller_dd(dd);
    os << '(' << dd.size() / smaller_dd.size() << ")×";
    print_dim_name(os, smaller_dd);
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
    ddc::detail::print_dim_name(os, chunk_span.domain());
    os << '\n';
    ddc::detail::print_demangled_type_name<decltype(chunk_span)>(os);
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
