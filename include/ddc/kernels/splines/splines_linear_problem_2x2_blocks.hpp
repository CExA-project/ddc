// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <string>

#if __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_dense.hpp"

namespace ddc::detail {

/**
 * @brief A 2x2-blocks linear problem dedicated to the computation of a spline approximation, with all blocks except top-left being stored in dense format.
 *
 * The storage format is dense row-major for top-left and bottom-right blocks, the one of SplinesLinearProblemDense (which is also dense row-major in practice) for bottom-right block and undefined for the top-left one (determined by the type of top_left_block).
 *
 * This class implements a Schur complement method to perform a block-LU factorization and solve, calling tl_block and br_block setup_solver() and solve() methods for internal operations.
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
    //-------------------------------------
    //
    //    q = | top_left_block | top_right_block |
    //        |  bottom_left_block | bottom_right_block |
    //
    //-------------------------------------
    std::shared_ptr<SplinesLinearProblem<ExecSpace>> m_top_left_block;
    std::shared_ptr<SplinesLinearProblem<ExecSpace>> m_bottom_right_block;
    Kokkos::View<double**, Kokkos::HostSpace> m_top_right_block;
    Kokkos::View<double**, Kokkos::HostSpace> m_bottom_left_block;

public:
    /**
     * @brief SplinesLinearProblem2x2Blocks constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblem2x2Blocks(
            std::size_t const mat_size,
            std::size_t const k,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> q)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_top_left_block(std::move(q))
        , m_bottom_right_block(new SplinesLinearProblemDense<ExecSpace>(k))
        , m_top_right_block(
                  "top_right_block",
                  m_top_left_block->size(),
                  mat_size - m_top_left_block->size())
        , m_bottom_left_block(
                  "bottom_left_block",
                  mat_size - m_top_left_block->size(),
                  m_top_left_block->size())
    {
        assert(m_top_left_block->size() <= mat_size);

        Kokkos::deep_copy(m_top_right_block, 0.);
        Kokkos::deep_copy(m_bottom_left_block, 0.);
    }

protected:
    explicit SplinesLinearProblem2x2Blocks(
            std::size_t const mat_size,
            std::size_t const k,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> q,
            std::size_t const bottom_left_block_size1,
            std::size_t const bottom_left_block_size2)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_top_left_block(std::move(q))
        , m_bottom_right_block(new SplinesLinearProblemDense<ExecSpace>(k))
        , m_top_right_block(
                  "top_right_block",
                  m_top_left_block->size(),
                  mat_size - m_top_left_block->size())
        , m_bottom_left_block("bottom_left_block", bottom_left_block_size1, bottom_left_block_size2)
    {
        assert(m_top_left_block->size() <= mat_size);

        Kokkos::deep_copy(m_top_right_block, 0.);
        Kokkos::deep_copy(m_bottom_left_block, 0.);
    }

public:
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
    virtual void calculate_bottom_right_block_to_factorize()
    {
        Kokkos::parallel_for(
                "calculate_bottom_right_block_to_factorize",
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
     * Block-LU factorize the matrix A according to the Schur complement method.
     */
    void setup_solver() override
    {
        m_top_left_block->setup_solver();
        auto top_right_block_device = create_mirror_view_and_copy(ExecSpace(), m_top_right_block);
        m_top_left_block->solve(top_right_block_device);
        deep_copy(m_top_right_block, top_right_block_device);
        calculate_bottom_right_block_to_factorize();
        m_bottom_right_block->setup_solver();
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_bottom_left_block_section(MultiRHS const v, MultiRHS const u) const
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
                                // Upper diagonals in bottom_left_block
                                for (int l = 0; l < u.extent(0); ++l) {
                                    v(i, j) -= bottom_left_block_device(i, l) * u(l, j);
                                }
                            });
                });
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_bottom_left_block_section_transpose(MultiRHS const u, MultiRHS const v) const
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
                                // Upper diagonals in bottom_left_block
                                for (int l = 0; l < v.extent(0); ++l) {
                                    u(i, j) -= bottom_left_block_device(l, i) * v(l, j);
                                }
                            });
                });
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_top_right_block_section(MultiRHS const u, MultiRHS const v) const
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
                                // Upper diagonals in bottom_left_block
                                for (int l = 0; l < v.extent(0); ++l) {
                                    u(i, j) -= top_right_block_device(i, l) * v(l, j);
                                }
                            });
                });
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_top_right_block_section_transpose(MultiRHS const v, MultiRHS const u) const
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
                                // Upper diagonals in bottom_left_block
                                for (int l = 0; l < u.extent(0); ++l) {
                                    v(i, j) -= top_right_block_device(l, i) * u(l, j);
                                }
                            });
                });
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgbtrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        MultiRHS u = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(0, m_top_left_block->size()),
                        Kokkos::ALL);
        MultiRHS v = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(m_top_left_block->size(), b.extent(0)),
                        Kokkos::ALL);
        if (!transpose) {
            m_top_left_block->solve(u);
            solve_bottom_left_block_section(v, u);
            m_bottom_right_block->solve(v);
            solve_top_right_block_section(u, v);
        } else {
            solve_top_right_block_section_transpose(v, u);
            m_bottom_right_block->solve(v, true);
            solve_bottom_left_block_section_transpose(u, v);
            m_top_left_block->solve(u, true);
        }
    }
};

} // namespace ddc::detail
