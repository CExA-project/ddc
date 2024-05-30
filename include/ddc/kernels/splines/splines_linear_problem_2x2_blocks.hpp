// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <string>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_dense.hpp"

namespace ddc::detail {

/**
 * @brief A 2x2-blocks linear problem dedicated to the computation of a spline approximation (taking in account boundary conditions),
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
class SplinesLinearProblem2x2Blocks : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using SplinesLinearProblem<ExecSpace>::size;

protected:
    std::shared_ptr<SplinesLinearProblem<ExecSpace>> m_top_left_block;
    Kokkos::View<double**, Kokkos::HostSpace> m_top_right_block;
    Kokkos::View<double**, Kokkos::HostSpace> m_bottom_left_block;
    std::shared_ptr<SplinesLinearProblem<ExecSpace>> m_bottom_right_block;

public:
    /**
     * @brief SplinesLinearProblem2x2Blocks constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param q A pointer toward the top-left SplinesLinearProblem.
     */
    explicit SplinesLinearProblem2x2Blocks(
            std::size_t const mat_size,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> q)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_top_left_block(std::move(q))
        , m_top_right_block(
                  "top_right_block",
                  m_top_left_block->size(),
                  mat_size - m_top_left_block->size())
        , m_bottom_left_block(
                  "bottom_left_block",
                  mat_size - m_top_left_block->size(),
                  m_top_left_block->size())
        , m_bottom_right_block(
                  new SplinesLinearProblemDense<ExecSpace>(mat_size - m_top_left_block->size()))
    {
        assert(m_top_left_block->size() <= mat_size);

        Kokkos::deep_copy(m_top_right_block, 0.);
        Kokkos::deep_copy(m_bottom_left_block, 0.);
    }

    virtual double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        if (i < nq && j < nq) {
            return m_top_left_block->get_element(i, j);
        } else if (i >= nq && j >= nq) {
            return m_bottom_right_block->get_element(i - nq, j - nq);
        } else if (j >= nq) {
            return m_top_right_block(i, j - nq);
        } else {
            return m_bottom_left_block(i - nq, j);
        }
    }

    virtual void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        if (i < nq && j < nq) {
            m_top_left_block->set_element(i, j, aij);
        } else if (i >= nq && j >= nq) {
            m_bottom_right_block->set_element(i - nq, j - nq, aij);
        } else if (j >= nq) {
            m_top_right_block(i, j - nq) = aij;
        } else {
            m_bottom_left_block(i - nq, j) = aij;
        }
    }

private:
    // @brief Compute the Schur complement delta - lambda*Q^-1*gamma.
    virtual void compute_schur_complement()
    {
        Kokkos::parallel_for(
                "compute_schur_complement",
                Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
                        {0, 0},
                        {m_bottom_right_block->size(), m_bottom_right_block->size()}),
                [&](const int i, const int j) {
                    double val = 0.0;
                    for (int l = 0; l < m_top_left_block->size(); ++l) {
                        val += m_bottom_left_block(i, l) * m_top_right_block(l, j);
                    }
                    m_bottom_right_block
                            ->set_element(i, j, m_bottom_right_block->get_element(i, j) - val);
                });
    }

public:
    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * Block-LU factorize the matrix A according to the Schur complement method. The block-LU factorization is:
     *
     * A = |   Q    |             0             |  | I | Q^-1*gamma |
     *     | lambda | delta - lambda*Q^-1*gamma |  | 0 |      I     |
     *
     * So we perform the factorization inplace to store only the relevant blocks in the matrix (while factorizing
     * the blocks themselves if necessary):
     *
     * |   Q    |         Q^-1*gamma        |
     * | lambda | delta - lambda*Q^-1*gamma |
     */
    void setup_solver() override
    {
        m_top_left_block->setup_solver();

        // Compute Q^-1*gamma
        auto top_right_block_device = create_mirror_view_and_copy(ExecSpace(), m_top_right_block);
        m_top_left_block->solve(top_right_block_device);
        deep_copy(m_top_right_block, top_right_block_device);

        // Compute delta - lambda*Q^-1*gamma
        compute_schur_complement();

        m_bottom_right_block->setup_solver();
    }

    /**
     * @brief Compute v <- v - lambda*u.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * @param u
     * @param v
     */
    virtual void solve_bottom_left_block_section(MultiRHS const u, MultiRHS v) const
    {
        auto bottom_left_block_device
                = create_mirror_view_and_copy(ExecSpace(), m_bottom_left_block);
        Kokkos::parallel_for(
                "solve_bottom_left_block_section",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, v.extent(0)),
                            [&](const int i) {
                                for (int l = 0; l < u.extent(0); ++l) {
                                    v(i, j) -= bottom_left_block_device(i, l) * u(l, j);
                                }
                            });
                });
    }

    /**
     * @brief Compute u <- u - lambda^t*v.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * @param u
     * @param v
     */
    virtual void solve_bottom_left_block_section_transpose(MultiRHS u, MultiRHS const v) const
    {
        auto bottom_left_block_device
                = create_mirror_view_and_copy(ExecSpace(), m_bottom_left_block);
        Kokkos::parallel_for(
                "solve_bottom_left_block_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(u.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, u.extent(0)),
                            [&](const int i) {
                                for (int l = 0; l < v.extent(0); ++l) {
                                    u(i, j) -= bottom_left_block_device(l, i) * v(l, j);
                                }
                            });
                });
    }

    /**
     * @brief Compute u <- u - gamma*v.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * @param u
     * @param v
     */
    virtual void solve_top_right_block_section(MultiRHS u, MultiRHS const v) const
    {
        auto top_right_block_device = create_mirror_view_and_copy(ExecSpace(), m_top_right_block);
        Kokkos::parallel_for(
                "solve_top_right_block_section",
                Kokkos::TeamPolicy<ExecSpace>(u.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, u.extent(0)),
                            [&](const int i) {
                                for (int l = 0; l < v.extent(0); ++l) {
                                    u(i, j) -= top_right_block_device(i, l) * v(l, j);
                                }
                            });
                });
    }

    /**
     * @brief Compute v <- v - gamma^t*u.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * @param u
     * @param v
     */
    virtual void solve_top_right_block_section_transpose(MultiRHS const u, MultiRHS v) const
    {
        auto top_right_block_device = create_mirror_view_and_copy(ExecSpace(), m_top_right_block);
        Kokkos::parallel_for(
                "solve_top_right_block_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, v.extent(0)),
                            [&](const int i) {
                                for (int l = 0; l < u.extent(0); ++l) {
                                    v(i, j) -= top_right_block_device(l, i) * u(l, j);
                                }
                            });
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
            solve_bottom_left_block_section(b1, b2);
            m_bottom_right_block->solve(b2);
            solve_top_right_block_section(b1, b2);
        } else {
            solve_top_right_block_section_transpose(b1, b2);
            m_bottom_right_block->solve(b2, true);
            solve_bottom_left_block_section_transpose(b1, b2);
            m_top_left_block->solve(b1, true);
        }
    }
};

} // namespace ddc::detail
