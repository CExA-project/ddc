// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <ostream>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <Kokkos_Core.hpp>

#include "chunk_span.hpp"
#include "discrete_vector.hpp"

namespace ddc {

struct PrinterOptions
{
    std::size_t threshold {10};
    std::size_t edgeitems {3};

    bool operator==(PrinterOptions const& rhs) const noexcept;
};

namespace detail {

/**
 * This class is a singleton, as it contains global printing option
 */
struct ChunkPrinter
{
    /*
     * We use a global lock because we do not care about performance for
     * printing and it ensure that output doesn't get mangled if trying to
     * print from different threads.
     */
    std::recursive_mutex m_global_lock;

    PrinterOptions m_options;

    // Copy of the stream format, used to compute how much space each element of the mdspan will take when printed
    std::stringstream m_ss;

    ChunkPrinter(ChunkPrinter& rhs) = delete;

    ChunkPrinter(ChunkPrinter&& rhs) = delete;

    ~ChunkPrinter();

    ChunkPrinter& operator=(ChunkPrinter&& rhs) = delete;

    ChunkPrinter& operator=(ChunkPrinter& rhs) = delete;

private:
    ChunkPrinter();

    /**
     * Print the spaces needed to align value to os
     */
    static std::ostream& align(std::ostream& os, int level);

    /**
     * Returns the size of the elem that needs to be printed
     */
    template <class T>
    std::size_t get_element_width(T const& elem)
    {
        m_ss.seekp(0);
        m_ss << elem;
        return m_ss.tellp();
    }

    /**
     * Print an element with enough leading spaces to ensure it is aligned with the others
     */
    template <class T>
    void display_aligned_element(std::ostream& os, T const& elem, std::size_t largest_element)
    {
        std::size_t const elem_width = get_element_width(elem);

        for (std::size_t i = 0; i < largest_element - elem_width; ++i) {
            os << " ";
        }
        os << elem;
    }

    /**
     * Displays the elements from the smallest dimension of a chunk span
     */
    template <class ElementType, class Extents, class Layout, class Accessor>
    std::ostream& base_case_display(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& span,
            std::size_t largest_element,
            std::size_t beginning,
            std::size_t end,
            std::size_t extent)
    {
        for (std::size_t i0 = beginning; i0 < end; ++i0) {
            display_aligned_element(os, span[i0], largest_element);
            if (i0 < extent - 1) {
                os << " ";
            }
        }
        return os;
    }

    /**
     * Recursively print the highest dimensions information of a chunk span (mostly '[' and ']'), and call base_case_display to print the smallest dimension
     */
    template <class ElementType, class Extents, class Layout, class Accessor, std::size_t... Is>
    std::ostream& recursive_display(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& span,
            int level,
            std::size_t largest_element,
            std::size_t beginning,
            std::size_t end,
            std::index_sequence<Is...>)
    {
        for (std::size_t i0 = beginning; i0 < end; ++i0) {
            print_impl(
                    os,
                    Kokkos::submdspan(span, i0, ((void)Is, Kokkos::full_extent)...),
                    level + 1,
                    largest_element,
                    std::make_index_sequence<sizeof...(Is)>());
            if (i0 < end - 1) {
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << '\n';
                }
                align(os, level);
            }
        }

        return os;
    }

public:
    static ChunkPrinter& get_instance();

    // 0D chunk span
    template <class ElementType, class Extents, class Layout, class Accessor>
    std::ostream& print_impl(
            std::ostream& os,
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& span,
            int /*level*/,
            std::size_t /*largest_element*/,
            std::index_sequence<>)
    {
        return os << *span.data_handle();
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
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& span,
            int level,
            std::size_t largest_element,
            std::index_sequence<I0, Is...>)
    {
        auto extent = span.extent(I0);
        if constexpr (sizeof...(Is) > 0) {
            os << '[';
            if (extent < m_options.threshold) {
                recursive_display(
                        os,
                        span,
                        level,
                        largest_element,
                        0,
                        extent,
                        std::make_index_sequence<sizeof...(Is)>());
            } else {
                recursive_display(
                        os,
                        span,
                        level,
                        largest_element,
                        0,
                        m_options.edgeitems,
                        std::make_index_sequence<sizeof...(Is)>());
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << '\n';
                }
                align(os, level);
                os << "...";
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << '\n';
                }
                align(os, level);
                recursive_display(
                        os,
                        span,
                        level,
                        largest_element,
                        extent - m_options.edgeitems,
                        extent,
                        std::make_index_sequence<sizeof...(Is)>());
            }
            os << "]";
        } else {
            os << "[";
            if (extent < m_options.threshold) {
                base_case_display(os, span, largest_element, 0, extent, extent);
            } else {
                base_case_display(os, span, largest_element, 0, m_options.edgeitems, extent);
                os << "... ";
                base_case_display(
                        os,
                        span,
                        largest_element,
                        extent - m_options.edgeitems,
                        extent,
                        extent);
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
            Kokkos::mdspan<ElementType, Extents, Layout, Accessor> const& span,
            std::index_sequence<I0, Is...>)
    {
        std::size_t ret = 0;
        auto extent = span.extent(I0);
        if constexpr (sizeof...(Is) > 0) {
            if (extent < m_options.threshold) {
                for (std::size_t i0 = 0; i0 < extent; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(span, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
            } else {
                for (std::size_t i0 = 0; i0 < m_options.edgeitems; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(span, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
                for (std::size_t i0 = extent - m_options.edgeitems; i0 < extent; ++i0) {
                    ret = std::max(
                            ret,
                            find_largest_displayed_element(
                                    Kokkos::submdspan(span, i0, ((void)Is, Kokkos::full_extent)...),
                                    std::make_index_sequence<sizeof...(Is)>()));
                }
            }
        } else {
            if (extent < m_options.threshold) {
                for (std::size_t i0 = 0; i0 < extent; ++i0) {
                    ret = std::max(ret, get_element_width(span[i0]));
                }
            } else {
                for (std::size_t i0 = 0; i0 < m_options.edgeitems; ++i0) {
                    ret = std::max(ret, get_element_width(span[i0]));
                }
                for (std::size_t i0 = extent - m_options.edgeitems; i0 < extent; ++i0) {
                    ret = std::max(ret, get_element_width(span[i0]));
                }
            }
        }

        return ret;
    }

    /*
     * Save the format that will be used in order to measure the size of the element we need to display
     */
    void saveformat(std::ostream& os);
};

/*
 * Print the demangled name of a type into a standard ostream.
 */
void print_demangled_type_name(std::ostream& os, char const* mangled_name);

void print_dim_name(
        std::ostream& os,
        char const* const* dims,
        DiscreteVectorElement const* sizes,
        std::size_t n);

template <class... Dims>
void print_dim_name(std::ostream& os, DiscreteVector<Dims...> const& dd)
{
    std::array<char const*, sizeof...(Dims)> const names {typeid(Dims).name()...};
    std::array const std_dd = detail::array(dd);
    print_dim_name(os, names.data(), std_dd.data(), std_dd.size());
}

} // namespace detail

/**
 * Try to set the options for the printer, returns the old format (or the
 * current format if the option passed are invalid).
 * option is invalid if m_edgeitems >= (m_threshold / 2), in this case the
 * format isn't changed.
 */
PrinterOptions set_print_options(PrinterOptions options = PrinterOptions());

/**
 * Return the currently used format options
 */
PrinterOptions get_print_options();

/**
 * Print the content of a ChunkSpan
 */
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print_content(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    auto h_chunk_span = create_mirror_view_and_copy(Kokkos::HostSpace(), chunk_span);

    using chunkspan_type = std::remove_cv_t<std::remove_reference_t<decltype(h_chunk_span)>>;
    using mdspan_type = chunkspan_type::allocation_mdspan_type;
    using extents = mdspan_type::extents_type;

    mdspan_type const allocated_mdspan = h_chunk_span.allocation_mdspan();

    ddc::detail::ChunkPrinter& printer = ddc::detail::ChunkPrinter::get_instance();
    std::scoped_lock const lock(printer.m_global_lock);

    printer.saveformat(os);

    std::size_t const largest_element = printer.find_largest_displayed_element(
            allocated_mdspan,
            std::make_index_sequence<extents::rank()>());

    printer.print_impl(
            os,
            allocated_mdspan,
            0 /*level*/,
            largest_element,
            std::make_index_sequence<extents::rank()>());

    return os;
}

/**
 * Print the metadata of a ChunkSpan (size, dimension name, ...)
 */
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print_type_info(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    ddc::detail::ChunkPrinter& printer = ddc::detail::ChunkPrinter::get_instance();
    std::scoped_lock const lock(printer.m_global_lock);

    ddc::detail::print_dim_name(os, chunk_span.extents());
    os << '\n';
    ddc::detail::print_demangled_type_name(os, typeid(chunk_span).name());
    os << '\n';

    return os;
}

/**
 * Print metadata and content of a ChunkSpan
 */
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    ddc::detail::ChunkPrinter& printer = ddc::detail::ChunkPrinter::get_instance();
    std::scoped_lock const lock(printer.m_global_lock);

    print_type_info(os, chunk_span);
    print_content(os, chunk_span);

    return os;
}

/**
 * Print metadata and content of a ChunkSpan
 * Always print without elision, regardless of options set
 */
template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print_full(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    ddc::detail::ChunkPrinter& printer = ddc::detail::ChunkPrinter::get_instance();
    std::scoped_lock const lock(printer.m_global_lock);

    PrinterOptions const old_options
            = set_print_options({.threshold = std::numeric_limits<size_t>::max(), .edgeitems = 1});

    print_type_info(os, chunk_span);
    print_content(os, chunk_span);

    set_print_options(old_options);

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
