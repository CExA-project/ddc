// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>

#include <Kokkos_Macros.hpp>

#include "discrete_vector.hpp"
#include "print.hpp"

#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
#    include <cxxabi.h>
#endif

namespace ddc {
namespace detail {

void print_demangled_type_name(std::ostream& os, char const* const mangled_name)
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

void print_single_dim_name(
        std::ostream& os,
        char const* const dim,
        DiscreteVectorElement const size)
{
    print_demangled_type_name(os, dim);
    os << '(' << size << ')';
}

void print_dim_name(
        std::ostream& os,
        char const* const* const dims,
        DiscreteVectorElement const* const sizes,
        std::size_t const n)
{
    if (n == 0) {
        os << "Scalar";
    } else {
        print_single_dim_name(os, dims[0], sizes[0]);
        for (std::size_t i = 1; i < n; ++i) {
            os << "×";
            print_single_dim_name(os, dims[i], sizes[i]);
        }
    }
}


} // namespace detail

void set_print_options(std::size_t edgeitems, std::size_t threshold)
{
    // Ensure that m_edgeitems < (m_threshold / 2) stays true.
    ddc::detail::ChunkPrinter& printer = ddc::detail::ChunkPrinter::getInstance();
    printer.m_global_lock.lock();

    if (edgeitems < threshold / 2) {
        printer.m_edgeitems = edgeitems;
        printer.m_threshold = threshold;
    } else {
        std::cerr << "DDC Printer: invalid values " << edgeitems << " for edgeitems and "
                  << threshold << " for threshold have been ignored\n"
                  << "threshold needs to be at least twice as big as edgeitems\n";
    }

    printer.m_global_lock.unlock();
}

} // namespace ddc
