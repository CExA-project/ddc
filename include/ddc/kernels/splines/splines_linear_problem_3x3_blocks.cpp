// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

#include <Kokkos_Core.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_2x2_blocks.hpp"
#include "splines_linear_problem_3x3_blocks.hpp"

namespace ddc::detail {

SplinesLinearProblem3x3Blocks::SplinesLinearProblem3x3Blocks(
        std::size_t const mat_size,
        std::size_t const top_size,
        std::unique_ptr<SplinesLinearProblem> center_block)
    : SplinesLinearProblem2x2Blocks(mat_size, std::move(center_block))
    , m_top_size(top_size)
{
}

SplinesLinearProblem3x3Blocks::~SplinesLinearProblem3x3Blocks() = default;

void SplinesLinearProblem3x3Blocks::adjust_indices(std::size_t& i, std::size_t& j) const
{
    std::size_t const nq = m_top_left_block->size(); // size of the center block

    if (i < m_top_size) {
        i += nq;
    } else if (i < m_top_size + nq) {
        i -= m_top_size;
    }

    if (j < m_top_size) {
        j += nq;
    } else if (j < m_top_size + nq) {
        j -= m_top_size;
    }
}

double SplinesLinearProblem3x3Blocks::get_element(std::size_t i, std::size_t j) const
{
    adjust_indices(i, j);
    return SplinesLinearProblem2x2Blocks::get_element(i, j);
}

void SplinesLinearProblem3x3Blocks::set_element(std::size_t i, std::size_t j, double const aij)
{
    adjust_indices(i, j);
    SplinesLinearProblem2x2Blocks::set_element(i, j, aij);
}

void SplinesLinearProblem3x3Blocks::interchange_rows_from_3_to_2_blocks_rhs(MultiRHS const b) const
{
    std::size_t const nq = m_top_left_block->size(); // size of the center block

    MultiRHS const b_top
            = Kokkos::subview(b, std::pair<std::size_t, std::size_t> {0, m_top_size}, Kokkos::ALL);
    MultiRHS const b_bottom = Kokkos::
            subview(b, std::pair<std::size_t, std::size_t> {m_top_size + nq, size()}, Kokkos::ALL);

    MultiRHS const b_top_dst = Kokkos::
            subview(b,
                    std::pair<std::size_t, std::size_t> {m_top_size + nq, 2 * m_top_size + nq},
                    Kokkos::ALL);
    MultiRHS const b_bottom_dst = Kokkos::
            subview(b,
                    std::pair<std::size_t, std::size_t> {2 * m_top_size + nq, m_top_size + size()},
                    Kokkos::ALL);

    if (b_bottom.extent(0) > b_top.extent(0)) {
        // Need a buffer to prevent overlapping
        MultiRHS const buffer = Kokkos::create_mirror(Kokkos::Serial(), b_bottom);

        Kokkos::deep_copy(buffer, b_bottom);
        Kokkos::deep_copy(b_bottom_dst, buffer);
    } else {
        Kokkos::deep_copy(b_bottom_dst, b_bottom);
    }
    Kokkos::deep_copy(b_top_dst, b_top);
}

void SplinesLinearProblem3x3Blocks::interchange_rows_from_2_to_3_blocks_rhs(MultiRHS const b) const
{
    std::size_t const nq = m_top_left_block->size(); // size of the center block

    MultiRHS const b_top
            = Kokkos::subview(b, std::pair<std::size_t, std::size_t> {0, m_top_size}, Kokkos::ALL);
    MultiRHS const b_bottom = Kokkos::
            subview(b, std::pair<std::size_t, std::size_t> {m_top_size + nq, size()}, Kokkos::ALL);

    MultiRHS const b_top_src = Kokkos::
            subview(b,
                    std::pair<std::size_t, std::size_t> {m_top_size + nq, 2 * m_top_size + nq},
                    Kokkos::ALL);
    MultiRHS const b_bottom_src = Kokkos::
            subview(b,
                    std::pair<std::size_t, std::size_t> {2 * m_top_size + nq, m_top_size + size()},
                    Kokkos::ALL);

    Kokkos::deep_copy(b_top, b_top_src);
    if (b_bottom.extent(0) > b_top.extent(0)) {
        // Need a buffer to prevent overlapping
        MultiRHS const buffer = Kokkos::create_mirror(Kokkos::Serial(), b_bottom);

        Kokkos::deep_copy(buffer, b_bottom_src);
        Kokkos::deep_copy(b_bottom, buffer);
    } else {
        Kokkos::deep_copy(b_bottom, b_bottom_src);
    }
}

void SplinesLinearProblem3x3Blocks::solve(MultiRHS const b, bool const transpose) const
{
    assert(b.extent(0) == size() + m_top_size);

    interchange_rows_from_3_to_2_blocks_rhs(b);
    SplinesLinearProblem2x2Blocks::
            solve(Kokkos::
                          subview(b,
                                  std::pair<
                                          std::size_t,
                                          std::size_t> {m_top_size, m_top_size + size()},
                                  Kokkos::ALL),
                  transpose);
    interchange_rows_from_2_to_3_blocks_rhs(b);
}

std::size_t SplinesLinearProblem3x3Blocks::impl_required_number_of_rhs_rows() const
{
    return size() + m_top_size;
}

} // namespace ddc::detail
