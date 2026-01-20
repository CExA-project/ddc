// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <iomanip>
#include <ostream>

#include <Kokkos_Core.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

SplinesLinearProblem::SplinesLinearProblem(std::size_t const size) : m_size(size) {}

/// @brief Destruct
SplinesLinearProblem::~SplinesLinearProblem() = default;

std::size_t SplinesLinearProblem::size() const
{
    return m_size;
}

std::size_t SplinesLinearProblem::required_number_of_rhs_rows() const
{
    std::size_t const nrows = impl_required_number_of_rhs_rows();
    assert(nrows >= size());
    return nrows;
}

std::size_t SplinesLinearProblem::impl_required_number_of_rhs_rows() const
{
    return m_size;
}

std::ostream& operator<<(std::ostream& os, SplinesLinearProblem const& linear_problem)
{
    std::size_t const n = linear_problem.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            os << std::fixed << std::setprecision(3) << std::setw(10)
               << linear_problem.get_element(i, j);
        }
        os << "\n";
    }
    return os;
}

} // namespace ddc::detail
