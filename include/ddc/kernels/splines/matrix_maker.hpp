// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <optional>

#include "matrix_sparse.hpp"

namespace ddc::detail {

/**
 * @brief A static factory for SplinesLinearProblem classes.
 */
class SplinesLinearProblemMaker
{
public:
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
