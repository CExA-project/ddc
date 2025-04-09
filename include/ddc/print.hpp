// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>

#ifdef KOKKOS_COMPILER_GNU
#    include <cxxabi.h>
#endif

namespace ddc {
struct ChunkPrinter
{
    const int threshold = 10;
    const int edgeitems = 3;

    std::stringstream ss;
    ChunkPrinter(const std::ostream& os)
    {
        ss.copyfmt(os);
    }

    std::ostream& alignment(std::ostream& os, int level)
    {
        for (int i = 0; i <= level; ++i) {
            os << ' ';
        }
        return os;
    }

    template <class T>
    size_t get_element_width(T& elem)
    {
        ss.seekp(0);
        ss << elem;
        return ss.tellp();
    }

    template <class T>
    void display_aligned_element(std::ostream& os, T& elem, size_t largest_element)
    {
        size_t elem_width = get_element_width(elem);

        for (int i = 0; i < largest_element - elem_width; ++i) {
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
            display_aligned_element(os, s(i0), largest_element);
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
            std::size_t end)
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
                    os << "\n";
                }
                alignment(os, level);
            }
        }

        return os;
    }

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
                recursive_display<
                        ElementType,
                        Extents,
                        Layout,
                        Accessor,
                        Is...>(os, s, level, largest_element, 0, extent);
            } else {
                // TODO: fixme, find other ways to test for this, not crashing on wrong parameters
                assert(edgeitems < extent && edgeitems < threshold / 2);
                recursive_display<
                        ElementType,
                        Extents,
                        Layout,
                        Accessor,
                        Is...>(os, s, level, largest_element, 0, edgeitems);
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << "\n";
                }
                alignment(os, level);
                os << "...";
                for (int ndims = 0; ndims < sizeof...(Is); ++ndims) {
                    os << "\n";
                }
                alignment(os, level);
                recursive_display<
                        ElementType,
                        Extents,
                        Layout,
                        Accessor,
                        Is...>(os, s, level, largest_element, extent - edgeitems, extent);
            }
            os << "]";
        } else {
            os << "[";
            if (extent < threshold) {
                base_case_display(os, s, largest_element, 0, extent, extent);
            } else {
                // TODO: fixme, find other ways to test for this, not crashing on wrong parameters
                assert(edgeitems < extent && edgeitems < threshold / 2);

                base_case_display(os, s, largest_element, 0, edgeitems, extent);
                os << "... ";
                base_case_display(os, s, largest_element, extent - edgeitems, extent, extent);
            }
            os << "]";
        }

        return os;
    }

    template <
            class ElementType,
            class Extents,
            class Layout,
            class Accessor,
            std::size_t I0,
            std::size_t... Is>
    size_t find_largest_displayed_element(
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
                    ret = std::max(ret, get_element_width(s(i0)));
                }
            } else {
                for (std::size_t i0 = 0; i0 < edgeitems; ++i0) {
                    ret = std::max(ret, get_element_width(s(i0)));
                }
                for (std::size_t i0 = extent - edgeitems; i0 < extent; ++i0) {
                    ret = std::max(ret, get_element_width(s(i0)));
                }
            }
        }

        return ret;
    }
};

#ifdef KOKKOS_COMPILER_GNU

template <class Type>
void print_demangled_type_name(std::ostream& os)
{
    char* demangled_name;
    int status;

    demangled_name = abi::__cxa_demangle(typeid(Type).name(), nullptr, 0, &status);
    if (status != 0) {
        os << "Error demangling dimension name:" << status;
        return;
    }

    os << demangled_name;
    free(demangled_name);
}

#else
template <class Type>
void print_demangled_type_name(std::ostream& os)
{
    os << typeid(Type).name();
}
#endif

template <class Dim>
void print_dim_name(std::ostream& os, const DiscreteDomain<Dim>)
{
    print_demangled_type_name<Dim>(os);
    os << " ";
}

template <class Dim0, class... Dims>
void print_dim_name(std::ostream& os, const DiscreteDomain<Dim0, Dims...>)
{
    print_demangled_type_name<Dim0>(os);
    os << " ";
    print_dim_name(os, DiscreteDomain<Dims...> {});
}


template <class ElementType, class SupportType, class LayoutStridedPolicy, class MemorySpace>
std::ostream& print(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    auto h_chunk_span = create_mirror_view_and_copy(Kokkos::HostSpace(), chunk_span);
    using chunkspan_type = std::remove_cv_t<std::remove_reference_t<decltype(h_chunk_span)>>;
    using mdspan_type = typename chunkspan_type::allocation_mdspan_type;
    using extents = typename mdspan_type::extents_type;

    mdspan_type allocated_mdspan = chunk_span.allocation_mdspan();

    ChunkPrinter printer(os);
    std::size_t largest_element = printer.find_largest_displayed_element(
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
std::ostream& print_chunk_info(
        std::ostream& os,
        ChunkSpan<ElementType, SupportType, LayoutStridedPolicy, MemorySpace> const& chunk_span)
{
    os << "\n";
    print_dim_name(os, chunk_span.domain());
    os << "\n";
    print_demangled_type_name<decltype(chunk_span)>(os);
    os << "\n";
    print_demangled_type_name<SupportType>(os);

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
