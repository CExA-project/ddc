// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include "splines_linear_problem.hpp"

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
 * @tparam Kokkos::Serial The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
class SplinesLinearProblem2x2Blocks : public SplinesLinearProblem
{
public:
    using SplinesLinearProblem::size;
    using typename SplinesLinearProblem::MultiRHS;

    struct Coo;

protected:
    std::unique_ptr<SplinesLinearProblem> m_top_left_block;
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            m_top_right_block;
    std::unique_ptr<Coo> m_top_right_block_coo;
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            m_bottom_left_block;
    std::unique_ptr<Coo> m_bottom_left_block_coo;
    std::unique_ptr<SplinesLinearProblem> m_bottom_right_block;

public:
    /**
     * @brief SplinesLinearProblem2x2Blocks constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param top_left_block A pointer toward the top-left SplinesLinearProblem. `setup_solver` must not have been called on it.
     */
    explicit SplinesLinearProblem2x2Blocks(
            std::size_t mat_size,
            std::unique_ptr<SplinesLinearProblem> top_left_block);

    ~SplinesLinearProblem2x2Blocks() override;

    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

    /**
     * @brief Fill a COO version of a Dense matrix (remove zeros).
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * Runs on a single thread to guarantee ordering.
     *
     * @param[in] dense_matrix The dense storage matrix whose non-zeros are extracted to fill the COO matrix.
     * @param[in] tol The tolerancy applied to filter the non-zeros.
     *
     * @return The COO storage matrix filled with the non-zeros from dense_matrix.
     */
    static std::unique_ptr<Coo> dense2coo(
            Kokkos::View<double const**, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
                    dense_matrix,
            double tol = 1e-14);

private:
    /// @brief Compute the Schur complement delta - lambda*Q^-1*gamma.
    void compute_schur_complement();

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
    void setup_solver() override;

    /**
     * @brief Compute y <- y - LinOp*x or y <- y - LinOp^t*x with a sparse LinOp.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     *
     * Perform a spdm operation (sparse-dense matrix multiplication) with parameters alpha=-1 and beta=1 between
     * a sparse matrix stored in COO format and a dense matrix x.
     *
     * @param[in] LinOp The sparse matrix, left side of the matrix multiplication.
     * @param[in] x The dense matrix, right side of the matrix multiplication.
     * @param[inout] y The dense matrix to be altered by the operation.
     * @param transpose A flag to indicate if the direct or transposed version of the operation is performed.
     */
    static void spdm_minus1_1(Coo* LinOp, MultiRHS x, MultiRHS y, bool transpose = false);

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
    void solve(MultiRHS b, bool transpose) const override;
};

} // namespace ddc::detail
