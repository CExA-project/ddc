// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <string>

#include <KokkosSparse_CooMatrix.hpp>
#include <KokkosSparse_coo2crs.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_DualView.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_dense.hpp"

namespace ddc::detail {

/**
 * @brief A 2x2-blocks linear problem dedicated to the computation of a spline approximation,
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
    std::unique_ptr<SplinesLinearProblem<ExecSpace>> m_top_left_block;
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
            m_top_right_block;
    KokkosSparse::CrsMatrix<double, int, typename ExecSpace::memory_space> m_top_right_block_sp;
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
            m_bottom_left_block;
    KokkosSparse::CrsMatrix<double, int, typename ExecSpace::memory_space> m_bottom_left_block_sp;
    std::unique_ptr<SplinesLinearProblem<ExecSpace>> m_bottom_right_block;

public:
    /**
     * @brief SplinesLinearProblem2x2Blocks constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param top_left_block A pointer toward the top-left SplinesLinearProblem. `setup_solver` must not have been called on it.
     */
    explicit SplinesLinearProblem2x2Blocks(
            std::size_t const mat_size,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> top_left_block)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_top_left_block(std::move(top_left_block))
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

        Kokkos::deep_copy(m_top_right_block.h_view, 0.);
        Kokkos::deep_copy(m_bottom_left_block.h_view, 0.);
    }

    double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        if (i < nq && j < nq) {
            return m_top_left_block->get_element(i, j);
        } else if (i >= nq && j >= nq) {
            return m_bottom_right_block->get_element(i - nq, j - nq);
        } else if (j >= nq) {
            return m_top_right_block.h_view(i, j - nq);
        } else {
            return m_bottom_left_block.h_view(i - nq, j);
        }
    }

    void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());

        std::size_t const nq = m_top_left_block->size();
        if (i < nq && j < nq) {
            m_top_left_block->set_element(i, j, aij);
        } else if (i >= nq && j >= nq) {
            m_bottom_right_block->set_element(i - nq, j - nq, aij);
        } else if (j >= nq) {
            m_top_right_block.h_view(i, j - nq) = aij;
        } else {
            m_bottom_left_block.h_view(i - nq, j) = aij;
        }
    }

    /**
     * @brief Fill a CRS version of a Dense matrix (remove zeros).
     *
     * /!\ Should be private, it is public due to CUDA limitation.
     *
     * Runs on a single thread to garantee ordering.
     *
     * @param dense_matrix The dense storage matrix whose non-zeros are extracted to fill the CRS matrix.
     *
     * @return The CRS storage matrix fill with the non-zeros from dense_matrix.
     */
    KokkosSparse::CrsMatrix<double, int, typename ExecSpace::memory_space> dense2crs(
            Kokkos::View<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
                    dense_matrix)
    {
        Kokkos::DualView<std::size_t, Kokkos::LayoutRight, typename ExecSpace::memory_space>
                n_nonzeros("ddc_splines_n_nonzeros");
        Kokkos::View<int*, Kokkos::LayoutRight, typename ExecSpace::memory_space> rows_idx(
                "ddc_splines_coo_rows_idx",
                dense_matrix.extent(0) * dense_matrix.extent(1));
        Kokkos::View<int*, Kokkos::LayoutRight, typename ExecSpace::memory_space> cols_idx(
                "ddc_splines_coo_cols_idx",
                dense_matrix.extent(0) * dense_matrix.extent(1));
        Kokkos::View<double*, Kokkos::LayoutRight, typename ExecSpace::memory_space>
                values("ddc_splines_coo_values", dense_matrix.extent(0) * dense_matrix.extent(1));
        Kokkos::parallel_for(
                "dense2coo",
                Kokkos::RangePolicy(ExecSpace(), 0, 1),
                KOKKOS_LAMBDA(const int) {
                    for (int i = 0; i < dense_matrix.extent(0); i++) {
                        for (int j = 0; j < dense_matrix.extent(1); j++) {
                            double aij = dense_matrix(i, j);
                            if (Kokkos::abs(aij) >= 1e-14) {
                                rows_idx(n_nonzeros.d_view()) = i;
                                cols_idx(n_nonzeros.d_view()) = j;
                                values(n_nonzeros.d_view()++) = aij;
                            }
                        }
                    }
                });
        n_nonzeros.modify_device();
        n_nonzeros.sync_host();
        Kokkos::resize(rows_idx, n_nonzeros.h_view());
        Kokkos::resize(cols_idx, n_nonzeros.h_view());
        Kokkos::resize(values, n_nonzeros.h_view());

        KokkosSparse::CooMatrix<double, int, typename ExecSpace::memory_space> coo_matrix(
                dense_matrix.extent(0),
                dense_matrix.extent(1),
                rows_idx,
                cols_idx,
                values);
        return KokkosSparse::
                coo2crs(coo_matrix.numRows(),
                        coo_matrix.numCols(),
                        coo_matrix.row(),
                        coo_matrix.col(),
                        coo_matrix.data());
    }

private:
    /// @brief Compute the Schur complement delta - lambda*Q^-1*gamma.
    void compute_schur_complement()
    {
        Kokkos::parallel_for(
                "compute_schur_complement",
                Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
                        {0, 0},
                        {m_bottom_right_block->size(), m_bottom_right_block->size()}),
                [&](const int i, const int j) {
                    double val = 0.0;
                    for (int l = 0; l < m_top_left_block->size(); ++l) {
                        val += m_bottom_left_block.h_view(i, l) * m_top_right_block.h_view(l, j);
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
        // Setup the top-left solver
        m_top_left_block->setup_solver();

        // Compute Q^-1*gamma in top-right block
        m_top_right_block.modify_host();
        m_top_right_block.sync_device();
        m_top_left_block->solve(m_top_right_block.d_view);
        m_top_right_block_sp = dense2crs(m_top_right_block.d_view);
        m_top_right_block.modify_device();
        m_top_right_block.sync_host();

        // Push lambda on device in bottom-left block
        m_bottom_left_block.modify_host();
        m_bottom_left_block.sync_device();
        m_bottom_left_block_sp = dense2crs(m_bottom_left_block.d_view);

        // Compute delta - lambda*Q^-1*gamma in bottom-right block & setup the bottom-right solver
        compute_schur_complement();
        m_bottom_right_block->setup_solver();
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

        KokkosSparse::SPMVHandle<
                ExecSpace,
                KokkosSparse::CrsMatrix<double, int, typename ExecSpace::memory_space>,
                MultiRHS,
                MultiRHS>
                spmv_handle(KokkosSparse::SPMVAlgorithm::SPMV_DEFAULT);

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
            KokkosSparse::
                    spmv(ExecSpace(), &spmv_handle, "N", -1., m_bottom_left_block_sp, b1, 1., b2);
            m_bottom_right_block->solve(b2);
            KokkosSparse::
                    spmv(ExecSpace(), &spmv_handle, "N", -1., m_top_right_block_sp, b2, 1., b1);
        } else {
            KokkosSparse::
                    spmv(ExecSpace(), &spmv_handle, "T", -1., m_top_right_block_sp, b1, 1., b2);
            m_bottom_right_block->solve(b2, true);
            KokkosSparse::
                    spmv(ExecSpace(), &spmv_handle, "T", -1., m_bottom_left_block_sp, b2, 1., b1);
            m_top_left_block->solve(b1, true);
        }
    }
};

} // namespace ddc::detail
