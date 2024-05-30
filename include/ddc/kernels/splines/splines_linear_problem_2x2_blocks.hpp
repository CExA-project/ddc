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
 * The storage format is dense row-major for top-left and bottom-right blocks, the one of SplinesLinearProblemDense (which is also dense row-major in practice) for bottom-right block and undefined for the top-left one (determined by the type of q_block).
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
    std::size_t const m_k; // small block size
    std::size_t const m_nb; // main block matrix size
    //-------------------------------------
    //
    //    q = | q_block | gamma |
    //        |  lambda | delta |
    //
    //-------------------------------------
    std::shared_ptr<SplinesLinearProblem<ExecSpace>> m_q_block;
    std::shared_ptr<SplinesLinearProblem<ExecSpace>> m_delta;
    Kokkos::View<double**, Kokkos::HostSpace> m_Abm_1_gamma;
    Kokkos::View<double**, Kokkos::HostSpace> m_lambda;

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
        , m_k(k)
        , m_nb(mat_size - k)
        , m_q_block(std::move(q))
        , m_delta(new SplinesLinearProblemDense<ExecSpace>(k))
        , m_Abm_1_gamma("Abm_1_gamma", m_nb, m_k)
        , m_lambda("lambda", m_k, m_nb)
    {
        assert(m_k <= mat_size);
        assert(m_nb == m_q_block->size()); // TODO: remove

        Kokkos::deep_copy(m_Abm_1_gamma, 0.);
        Kokkos::deep_copy(m_lambda, 0.);
    }

protected:
    explicit SplinesLinearProblem2x2Blocks(
            std::size_t const mat_size,
            std::size_t const k,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> q,
            std::size_t const lambda_size1,
            std::size_t const lambda_size2)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_k(k)
        , m_nb(mat_size - k)
        , m_q_block(std::move(q))
        , m_delta(new SplinesLinearProblemDense<ExecSpace>(k))
        , m_Abm_1_gamma("Abm_1_gamma", m_nb, m_k)
        , m_lambda("lambda", lambda_size1, lambda_size2)
    {
        assert(m_k <= mat_size);
        assert(m_nb == m_q_block->size()); // TODO: remove

        Kokkos::deep_copy(m_Abm_1_gamma, 0.);
        Kokkos::deep_copy(m_lambda, 0.);
    }

public:
    virtual double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());

        if (i < m_nb && j < m_nb) {
            return m_q_block->get_element(i, j);
        } else if (i >= m_nb && j >= m_nb) {
            return m_delta->get_element(i - m_nb, j - m_nb);
        } else if (j >= m_nb) {
            return m_Abm_1_gamma(i, j - m_nb);
        } else {
            return m_lambda(i - m_nb, j);
        }
    }

    virtual void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());

        if (i < m_nb && j < m_nb) {
            m_q_block->set_element(i, j, aij);
        } else if (i >= m_nb && j >= m_nb) {
            m_delta->set_element(i - m_nb, j - m_nb, aij);
        } else if (j >= m_nb) {
            m_Abm_1_gamma(i, j - m_nb) = aij;
        } else {
            m_lambda(i - m_nb, j) = aij;
        }
    }

private:
    virtual void calculate_delta_to_factorize()
    {
        Kokkos::parallel_for(
                "calculate_delta_to_factorize",
                Kokkos::MDRangePolicy<
                        Kokkos::DefaultHostExecutionSpace,
                        Kokkos::Rank<2>>({0, 0}, {m_k, m_k}),
                [&](const int i, const int j) {
                    double val = 0.0;
                    for (int l = 0; l < m_nb; ++l) {
                        val += m_lambda(i, l) * m_Abm_1_gamma(l, j);
                    }
                    m_delta->set_element(i, j, m_delta->get_element(i, j) - val);
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
        m_q_block->setup_solver();
        auto Abm_1_gamma_device = create_mirror_view_and_copy(ExecSpace(), m_Abm_1_gamma);
        m_q_block->solve(Abm_1_gamma_device);
        deep_copy(m_Abm_1_gamma, Abm_1_gamma_device);
        calculate_delta_to_factorize();
        m_delta->setup_solver();
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_lambda_section(MultiRHS const v, MultiRHS const u) const
    {
        auto lambda_device = create_mirror_view_and_copy(ExecSpace(), m_lambda);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_lambda_section",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, k_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < nb_proxy; ++l) {
                                    v(i, j) -= lambda_device(i, l) * u(l, j);
                                }
                            });
                });
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_lambda_section_transpose(MultiRHS const u, MultiRHS const v) const
    {
        auto lambda_device = create_mirror_view_and_copy(ExecSpace(), m_lambda);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_lambda_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(u.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, nb_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < k_proxy; ++l) {
                                    u(i, j) -= lambda_device(l, i) * v(l, j);
                                }
                            });
                });
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_gamma_section(MultiRHS const u, MultiRHS const v) const
    {
        auto Abm_1_gamma_device = create_mirror_view_and_copy(ExecSpace(), m_Abm_1_gamma);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_gamma_section",
                Kokkos::TeamPolicy<ExecSpace>(u.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, nb_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < k_proxy; ++l) {
                                    u(i, j) -= Abm_1_gamma_device(i, l) * v(l, j);
                                }
                            });
                });
    }

    /**
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
    virtual void solve_gamma_section_transpose(MultiRHS const v, MultiRHS const u) const
    {
        auto Abm_1_gamma_device = create_mirror_view_and_copy(ExecSpace(), m_Abm_1_gamma);
        auto nb_proxy = m_nb;
        auto k_proxy = m_k;
        Kokkos::parallel_for(
                "solve_gamma_section_transpose",
                Kokkos::TeamPolicy<ExecSpace>(v.extent(1), Kokkos::AUTO),
                KOKKOS_LAMBDA(
                        const typename Kokkos::TeamPolicy<ExecSpace>::member_type& teamMember) {
                    const int j = teamMember.league_rank();


                    Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(teamMember, k_proxy),
                            [&](const int i) {
                                // Upper diagonals in lambda
                                for (int l = 0; l < nb_proxy; ++l) {
                                    v(i, j) -= Abm_1_gamma_device(l, i) * u(l, j);
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

        MultiRHS u = Kokkos::subview(b, std::pair<std::size_t, std::size_t>(0, m_nb), Kokkos::ALL);
        MultiRHS v = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(m_nb, b.extent(0)), Kokkos::ALL);
        if (!transpose) {
            m_q_block->solve(u);
            solve_lambda_section(v, u);
            m_delta->solve(v);
            solve_gamma_section(u, v);
        } else {
            solve_gamma_section_transpose(v, u);
            m_delta->solve(v, true);
            solve_lambda_section_transpose(u, v);
            m_q_block->solve(u, true);
        }
    }
};

} // namespace ddc::detail
