// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <optional>

<<<<<<< HEAD
#include "splines_linear_problem_band.hpp"
=======
>>>>>>> main
        #include "splines_linear_problem_dense.hpp"
#include "splines_linear_problem_sparse.hpp"

        namespace ddc::detail
{
    /**
 * @brief A static factory for SplinesLinearProblem classes.
 */
    class SplinesLinearProblemMaker
    {
    public:
        /**
     * @brief Construct a dense matrix
     *
     * @tparam the Kokkos::ExecutionSpace on which matrix-related operation will be performed.
     * @param n The size of one of the dimensions of the square matrix.
     *
     * @return The SplinesLinearProblem instance.
     */
        template <typename ExecSpace>
        static std::unique_ptr<SplinesLinearProblem<ExecSpace>> make_new_dense(int const n)
        {
            return std::make_unique<SplinesLinearProblemDense<ExecSpace>>(n);
        }

        /**
     * @brief Construct a band matrix
     *
     * @tparam the Kokkos::ExecutionSpace on which matrix-related operation will be performed.
     * @param n The size of one of the dimensions of the square matrix.
     * @param kl The number of subdiagonals.
     * @param ku The number of superdiagonals.
     * @param pds A boolean indicating if the matrix is positive-definite symetric or not.
     *
     * @return The SplinesLinearProblem instance.
     */
        template <typename ExecSpace>
        static std::unique_ptr<SplinesLinearProblem<ExecSpace>> make_new_band(
                int const n,
                [[maybe_unused]] int const kl,
                [[maybe_unused]] int const ku,
                [[maybe_unused]] bool const pds)
        {
            if (2 * kl + 1 + ku >= n) {
                return std::make_unique<SplinesLinearProblemDense<ExecSpace>>(n);
            } else {
                return std::make_unique<SplinesLinearProblemBand<ExecSpace>>(n, kl, ku);
            }
        }

        /**
     * @brief Construct a sparse matrix
     *
     * @tparam the Kokkos::ExecutionSpace on which matrix-related operation will be performed.
     * @param n The size of one of the dimensions of the square matrix.
     * @param cols_per_chunk A parameter used by the slicer (internal to the solver) to define the size
     * of a chunk of right-hand sides of the linear problem to be computed in parallel (chunks are treated
     * by the linear solver one-after-the-other).
     * This value is optional. If no value is provided then the default value is chosen by the requested solver.
     *
     * @param preconditionner_max_block_size A parameter used by the slicer (internal to the solver) to
     * define the size of a block used by the Block-Jacobi preconditioner.
     * This value is optional. If no value is provided then the default value is chosen by the requested solver.
     *
     * @return The SplinesLinearProblem instance.
     */
        template <typename ExecSpace>
        static std::unique_ptr<SplinesLinearProblem<ExecSpace>> make_new_sparse(
                int const n,
                std::optional<std::size_t> cols_per_chunk = std::nullopt,
                std::optional<unsigned int> preconditionner_max_block_size = std::nullopt)
        {
            return std::make_unique<SplinesLinearProblemSparse<
                    ExecSpace>>(n, cols_per_chunk, preconditionner_max_block_size);
        }
    };

} // namespace ddc::detail
