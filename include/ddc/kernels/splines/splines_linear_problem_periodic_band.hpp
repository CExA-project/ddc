// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <string>

#include <Kokkos_DualView.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_dense.hpp"

namespace ddc::detail {

/**
 * @brief A periodic-band linear problem dedicated to the computation of a spline approximation (taking in account boundary conditions),
 * with all blocks except top-left one being stored in dense format.
 *
 * A = |   Q    | gamma |
 *     | lambda | delta |
 *
 * The storage format is dense row-major for top-left, top-right and bottom-left blocks, and determined by 
 * its type for the top-left block.
 *
 * This class implements a Schur complement method to perform a block-LU factorization and solve,
 * calling top-left block and bottom-right block setup_solver() and solve() methods for internal operations.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
template <class ExecSpace>
class SplinesLinearProblemPeriodicBand : public SplinesLinearProblem2x2Blocks<ExecSpace>
{
public:
    using typename SplinesLinearProblem2x2Blocks<ExecSpace>::MultiRHS;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::size;

protected:
    std::size_t m_kl; // no. of subdiagonals
    std::size_t m_ku; // no. of superdiagonals

public:
    /**
     * @brief SplinesLinearProblem2x2Blocks constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param q A pointer toward the top-left SplinesLinearProblem.
     */
    explicit SplinesLinearProblemPeriodicBand(
            std::size_t const mat_size,
            std::size_t const kl,
            std::size_t const ku,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> top_left_block)
        : SplinesLinearProblem2x2Blocks<ExecSpace>(mat_size, std::move(top_left_block))
    {
    }

    double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        std::size_t const ndelta = m_bottom_right_block->size();
        if (i >= nq && j < nq) {
            std::size_t const d = j - i;
            if (d > size() / 2)
                d -= size();
            if (d < -size() / 2)
                d += size();

            if (d < -m_kl || d > m_ku)
                return 0.0;
            if (d > 0) {
                return m_bottom_left_block(i - nq, j);
            } else {
                return m_bottom_left_block(i - nq, j - nq + ndelta + 1);
            }
        } else {
            return MatrixCornerBlock<ExecSpace>::get_element(i, j);
        }
    }

    void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        std::size_t const ndelta = m_bottom_right_block->size();
        if (i >= nq && j < nq) {
            int d = j - i;
            if (d > size() / 2)
                d -= size();
            if (d < -size() / 2)
                d += size();

            if (d < -m_kl || d > m_ku) {
                assert(std::fabs(aij) < 1e-20);
                return;
            }

            if (d > 0) {
                m_bottom_left_block(i - nq, j) = aij;
            } else {
                m_bottom_left_block(i - nq, j - nq + ndelta + 1) = aij;
            }
        } else {
            MatrixCornerBlock<ExecSpace>::set_element(i, j, aij);
        }
    }

private:
    // @brief Compute the Schur complement delta - lambda*Q^-1*gamma.
    void compute_schur_complement()
    {
        Kokkos::parallel_for(
                "compute_schur_complement",
                Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
                        {0, 0},
                        {m_bottom_right_block->size(), m_bottom_right_block->size()}),
                [&](const int i, const int j) {
                    double val = 0.0;
                    // Upper diagonals in lambda, lower diagonals in gamma
                    for (int l = 0; l < i + 1; ++l) {
                        val += m_bottom_left_block.h_view(i, l) * m_top_right_block.h_view(l, j);
                    }
                    // Lower diagonals in lambda, upper diagonals in gamma
                    for (int l = i + 1; l < m_bottom_right_block->size() + 1; ++l) {
                        int const l_full
                                = m_top_left_block->size() - 1 - m_bottom_right_block->size() + l;
                        val += m_bottom_left_block.h_view(i, l)
                               * m_top_right_block.h_view(l_full, j);
                    }
                    m_bottom_right_block
                            ->set_element(i, j, m_bottom_right_block->get_element(i, j) - val);
                });
    }

    /**
     * @brief Compute y <- y - LinOp*x or y <- y - LinOp^t*x.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * @param x
     * @param y
     * @param LinOp
     * @param transpose
     */
    void gemv_minus1_1(
            MultiRHS const x,
            MultiRHS const y,
            Kokkos::View<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space> const
                    LinOp,
            bool const transpose = false) const
    {
        assert(!transpose && LinOp.extent(0) == y.extent(0)
               || transpose && LinOp.extent(1) == y.extent(0));
        assert(!transpose && LinOp.extent(1) == x.extent(0)
               || transpose && LinOp.extent(0) == x.extent(0));
        assert(x.extent(1) == y.extent(1));

        Kokkos::parallel_for(
                "gemv_minus1_1",
                Kokkos::TeamPolicy<ExecSpace>(y.extent(0) * y.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int i = teamMember.league_rank() / y.extent(1);
                    const int j = teamMember.league_rank() % y.extent(1);

                    double LinOpTimesX = 0.;
                    Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(teamMember, !transpose ? i + 1 : i),
                            [&](const int l, double& LinOpTimesX_tmp) {
                                if (!transpose) {
                                    LinOpTimesX_tmp += LinOp(i, l) * x(l, j);
                                } else {
                                    LinOpTimesX_tmp += LinOp(l, i) * x(l, j);
                                }
                            },
                            LinOpTimesX);
                    teamMember.team_barrier();
                    double LinOpTimesX2 = 0.;
                    Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(
                                    teamMember,
                                    !transpose ? i + 1 : i,
                                    x.extent(0)),
                            [&](const int l, double& LinOpTimesX_tmp) {
                                int const l_full = m_top_left_block->size() - 1
                                                   - m_bottom_right_block->size() + l;
                                if (!transpose) {
                                    LinOpTimesX_tmp += LinOp(i, l) * x(l_full, j);
                                } else {
                                    LinOpTimesX_tmp += LinOp(l, i) * x(l_full, j);
                                }
                            },
                            LinOpTimesX2);
                    if (teamMember.team_rank() == 0) {
                        y(i, j) -= LinOpTimesX + LinOpTimesX2;
                    }
                });
    }
};

} // namespace ddc::detail
