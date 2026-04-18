// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>

#include <Kokkos_Macros.hpp> // IWYU pragma: keep (for KOKKOS_COMPILER_CLANG)

#include "discrete_vector.hpp"
#include "print.hpp"

#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
#    include <cxxabi.h>
#endif

namespace ddc {

// Workaround for nvcc, it should be `= default` but this creates a missing symbol at link stage.
bool PrinterOptions::operator==(PrinterOptions const& rhs) const noexcept
{
    return threshold == rhs.threshold && edgeitems == rhs.edgeitems;
}

namespace detail {

ChunkPrinter::~ChunkPrinter() = default;

ChunkPrinter::ChunkPrinter() = default;

void ChunkPrinter::saveformat(std::ostream& os)
{
    m_ss.copyfmt(os);
}

std::ostream& ChunkPrinter::align(std::ostream& os, int const level)
{
    for (int i = 0; i <= level; ++i) {
        os << ' ';
    }
    return os;
}

ChunkPrinter& ChunkPrinter::get_instance()
{
    static ChunkPrinter instance;
    return instance;
}

void print_demangled_type_name(std::ostream& os, char const* const mangled_name)
{
#if defined(KOKKOS_COMPILER_GNU) || defined(KOKKOS_COMPILER_CLANG)
    int status;

    std::unique_ptr<char, decltype(std::free)*> const
            demangled_name(abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status), std::free);
    if (status != 0) {
        std::cerr << "Error demangling dimension name: " << status << '\n' << std::flush;
        os << mangled_name;
    } else {
        os << demangled_name.get();
    }

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

PrinterOptions set_print_options(PrinterOptions const options)
{
    ddc::detail::ChunkPrinter& printer = ddc::detail::ChunkPrinter::get_instance();

    PrinterOptions old_options = printer.m_options;

    // Ensure options are not modified while an other thread is printing
    std::scoped_lock const lock(printer.m_global_lock);

    // Ensure that m_edgeitems < (m_threshold / 2) stays true.
    if (options.edgeitems < options.threshold / 2) {
        printer.m_options = options;
    } else {
        std::cerr << "DDC Printer: invalid values " << options.edgeitems << " for edgeitems and "
                  << options.threshold << " for threshold have been ignored\n"
                  << "threshold needs to be at least twice as big as edgeitems\n";
    }

    return old_options;
}

PrinterOptions get_print_options()
{
    ddc::detail::ChunkPrinter const& printer = ddc::detail::ChunkPrinter::get_instance();
    return printer.m_options;
}

} // namespace ddc
