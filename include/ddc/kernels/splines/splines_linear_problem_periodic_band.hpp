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
    using SplinesLinearProblem2x2Blocks<ExecSpace>::m_top_left_block;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::m_top_right_block;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::m_bottom_left_block;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::m_bottom_right_block;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::gemv_minus1_1;

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
        : SplinesLinearProblem2x2Blocks<ExecSpace>(
                mat_size,
                std::move(top_left_block),
                std::max(kl, ku),
                std::max(kl, ku) + 1)
        , m_kl(kl)
        , m_ku(ku)
    {
    }

    double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        std::size_t const ndelta = m_bottom_right_block->size();
        if (i >= nq && j < nq) {
            std::ptrdiff_t d = j - i;
            if (d > (std::ptrdiff_t)(size() / 2))
                d -= size();
            if (d < -(std::ptrdiff_t)(size() / 2))
                d += size();

            if (d < -(std::ptrdiff_t)m_kl || d > (std::ptrdiff_t)m_ku)
                return 0.0;
            if (d > (std::ptrdiff_t)0) {
                return m_bottom_left_block.h_view(i - nq, j);
            } else {
                return m_bottom_left_block.h_view(i - nq, j - nq + ndelta + 1);
            }
        } else {
            return SplinesLinearProblem2x2Blocks<ExecSpace>::get_element(i, j);
        }
    }

    void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        std::size_t const ndelta = m_bottom_right_block->size();
        if (i >= nq && j < nq) {
            std::ptrdiff_t d = j - i;
            if (d > (std::ptrdiff_t)(size() / 2))
                d -= size();
            if (d < -(std::ptrdiff_t)(size() / 2))
                d += size();

            if (d < -(std::ptrdiff_t)m_kl || d > (std::ptrdiff_t)m_ku) {
                assert(std::fabs(aij) < 1e-20);
                return;
            }
            if (d > (std::ptrdiff_t)0) {
                m_bottom_left_block.h_view(i - nq, j) = aij;
            } else {
                m_bottom_left_block.h_view(i - nq, j - nq + ndelta + 1) = aij;
            }
        } else {
            SplinesLinearProblem2x2Blocks<ExecSpace>::set_element(i, j, aij);
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

public:
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
    void per_gemv_minus1_1(
            MultiRHS const x,
            MultiRHS const y,
            Kokkos::View<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space> const
                    LinOp,
            bool const transpose = false) const
    {
        /*
        assert(!transpose && LinOp.extent(0) == y.extent(0)
               || transpose && LinOp.extent(1) == y.extent(0));
        assert(!transpose && LinOp.extent(1) == x.extent(0)
               || transpose && LinOp.extent(0) == x.extent(0));
		*/
        assert(x.extent(1) == y.extent(1));

        std::size_t const nq = m_top_left_block->size();
        std::size_t const ndelta = m_bottom_right_block->size();
        Kokkos::parallel_for(
                "per_gemv_minus1_1",
                Kokkos::TeamPolicy<
                        ExecSpace>((y.extent(0) + transpose) * y.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int i = teamMember.league_rank() / y.extent(1);
                    const int j = teamMember.league_rank() % y.extent(1);

                    if (!transpose) {
                        double LinOpTimesX = 0.;
                        Kokkos::parallel_reduce(
                                Kokkos::TeamThreadRange(teamMember, i + 1),
                                [&](const int l, double& LinOpTimesX_tmp) {
                                    LinOpTimesX_tmp += LinOp(i, l) * x(l, j);
                                },
                                LinOpTimesX);
                        teamMember.team_barrier();
                        double LinOpTimesX2 = 0.;
                        Kokkos::parallel_reduce(
                                Kokkos::TeamThreadRange(teamMember, i + 1, ndelta),
                                [&](const int l, double& LinOpTimesX_tmp) {
                                    int const l_full = nq - 1 - ndelta + l;
                                    LinOpTimesX_tmp += LinOp(i, l) * x(l_full, j);
                                },
                                LinOpTimesX2);
                        if (teamMember.team_rank() == 0) {
                            y(i, j) -= LinOpTimesX + LinOpTimesX2;
                        }
                    } else {
                        // Lower diagonals in lambda
                        for (int l = 0; l < i; ++l) {
                            if (teamMember.team_rank() == 0) {
                                y(nq - 1 - ndelta + i, j) -= LinOp(l, i) * x(l, j);
                            }
                        }
                        /// Upper diagonals in lambda
                        for (int l = i; l < ndelta; ++l) {
                            if (teamMember.team_rank() == 0) {
                                y(i, j) -= LinOp(l, i) * x(l, j);
                            }
                        }
                    }
                });
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is the one known as Schur complement method. It can be summarized as follow,
     * starting with the pre-computed elements of the matrix:
     *
     * |   Q    |         Q^-1*gamma        |
     * | lambda | delta - lambda*Q^-1*gamma |
     *
     * For the non-transposed case:
     * - Solve inplace Q * x'1 = b1 (using the solver internal to Q).
     * - Compute inplace b'2 = b2 - lambda*x'1.
     * - Solve inplace (delta - lambda*Q^-1*gamma) * x2 = b'2. 
     * - Compute inplace x1 = x'1 - (delta - lambda*Q^-1*gamma)*x2. 
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        MultiRHS b1 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(0, m_top_left_block->size()),
                        Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(m_top_left_block->size(), b.extent(0)),
                        Kokkos::ALL);
        if (!transpose) {
            m_top_left_block->solve(b1);
            per_gemv_minus1_1(b1, b2, m_bottom_left_block.d_view);
            m_bottom_right_block->solve(b2);
            gemv_minus1_1(b2, b1, m_top_right_block.d_view);
        } else {
            gemv_minus1_1(b1, b2, m_top_right_block.d_view, true);
            m_bottom_right_block->solve(b2, true);
            per_gemv_minus1_1(b2, b1, m_bottom_left_block.d_view, true);
            m_top_left_block->solve(b1, true);
        }
    }
};

} // namespace ddc::detail
