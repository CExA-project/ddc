// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <ostream>
#include <span>

#include <Kokkos_Core.hpp>

#include "discrete_element.hpp"
#include "discrete_vector.hpp"
#include "sparse_discrete_domain.hpp"

namespace ddc::detail {

namespace {

/// Print the ids of the elements of one dimension, eliding the middle ones if there are too many
void print_sparse_discrete_domain_dimension(
        std::ostream& os,
        Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace> const& view)
{
    /// Number of elements displayed at each end of a dimension of a SparseDiscreteDomain
    std::size_t const print_edge_items = 3;
    std::size_t const n = view.size();
    os << '[';
    if (n <= 2 * print_edge_items + 1) {
        for (std::size_t i = 0; i < n; ++i) {
            os << (i == 0 ? "" : ", ") << view(i);
        }
    } else {
        for (std::size_t i = 0; i < print_edge_items; ++i) {
            os << view(i) << ", ";
        }
        os << "...";
        for (std::size_t i = n - print_edge_items; i < n; ++i) {
            os << ", " << view(i);
        }
    }
    os << ']';
}

} // namespace

void print_sparse_discrete_domain(
        std::ostream& os,
        bool const empty,
        std::span<DiscreteVectorElement const> const extents,
        std::span<Kokkos::View<DiscreteElementType*, Kokkos::SharedSpace> const> const elements)
{
    os << "SparseDiscreteDomain(";
    if (empty) {
        os << "empty";
    } else {
        os << "extents=";
        print_discrete_vector(os, extents);
        os << ", elements=[";
        for (std::size_t i = 0; i < elements.size(); ++i) {
            os << (i == 0 ? "" : ", ");
            print_sparse_discrete_domain_dimension(os, elements[i]);
        }
        os << ']';
    }
    os << ')';
}

} // namespace ddc::detail
