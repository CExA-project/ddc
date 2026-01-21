// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>

#include <Kokkos_Core.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_2x2_blocks.hpp"

namespace ddc::detail {

/**
 * @brief A 3x3-blocks linear problem dedicated to the computation of a spline approximation,
 * with all blocks except center one being stored in dense format.
 *
 * A = | a | b | c |
 *     | d | Q | e |
 *     | f | g | h |
 *
 * The storage format is dense for all blocks except center one, whose storage format is determined by its type.
 *
 * The matrix itself and blocks a, Q and h are square (which fully determines the dimensions of the others).
 *
 * This class implements row & columns interchanges of the matrix and of multiple right-hand sides to restructure the
 * 3x3-blocks linear problem into a 2x2-blocks linear problem, relying then on the operations implemented in SplinesLinearProblem2x2Blocks.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
template <class ExecSpace>
class SplinesLinearProblem3x3Blocks : public SplinesLinearProblem2x2Blocks<ExecSpace>
{
public:
    using typename SplinesLinearProblem2x2Blocks<ExecSpace>::MultiRHS;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::size;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::solve;
    using SplinesLinearProblem2x2Blocks<ExecSpace>::m_top_left_block;

protected:
    std::size_t m_top_size;

public:
    /**
     * @brief SplinesLinearProblem3x3Blocks constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param top_size The size of one of the dimensions of the top-left square block.
     * @param center_block A pointer toward the center SplinesLinearProblem. `setup_solver` must not have been called on it.
     */
    explicit SplinesLinearProblem3x3Blocks(
            std::size_t mat_size,
            std::size_t top_size,
            std::unique_ptr<SplinesLinearProblem<ExecSpace>> center_block);

    SplinesLinearProblem3x3Blocks(SplinesLinearProblem3x3Blocks const& rhs) = delete;

    SplinesLinearProblem3x3Blocks(SplinesLinearProblem3x3Blocks&& rhs) = delete;

    ~SplinesLinearProblem3x3Blocks() override;

    SplinesLinearProblem3x3Blocks& operator=(SplinesLinearProblem3x3Blocks const& rhs) = delete;

    SplinesLinearProblem3x3Blocks& operator=(SplinesLinearProblem3x3Blocks&& rhs) = delete;

private:
    /// @brief Adjust indices, governs the row & columns interchanges to restructure the 3x3-blocks matrix into a 2x2-blocks matrix.
    void adjust_indices(std::size_t& i, std::size_t& j) const;

public:
    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

private:
    /**
     * @brief Perform row interchanges on multiple right-hand sides to get a 2-blocks structure (matching the requirements
     * of the SplinesLinearProblem2x2Blocks solver).
     *
     * |  b_top   |    |    -     |
     * | b_center | -> | b_center |
     * | b_bottom |    |  b_top   | -- Considered as a
     * |    -     |    | b_bottom | -- single bottom block
     *
     * @param b The multiple right-hand sides.
     */
    void interchange_rows_from_3_to_2_blocks_rhs(MultiRHS b) const;

    /**
     * @brief Perform row interchanges on multiple right-hand sides to restore its 3-blocks structure.
     *
     * |    -     |    |  b_top   |
     * | b_center | -> | b_center |
     * |  b_top   |    | b_bottom |
     * | b_bottom |    |    -     |
     *
     * @param b The multiple right-hand sides.
     */
    void interchange_rows_from_2_to_3_blocks_rhs(MultiRHS b) const;

public:
    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * Perform row interchanges on multiple right-hand sides to obtain a 2x2-blocks linear problem and call the SplinesLinearProblem2x2Blocks solver.
     *
     * This class requires an additional allocation corresponding to top_size rows for internal operation.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides (+ additional garbage allocation) of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool transpose) const override;

private:
    std::size_t impl_required_number_of_rhs_rows() const override;
};

#if defined(KOKKOS_ENABLE_SERIAL)
extern template class SplinesLinearProblem3x3Blocks<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
extern template class SplinesLinearProblem3x3Blocks<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
extern template class SplinesLinearProblem3x3Blocks<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
extern template class SplinesLinearProblem3x3Blocks<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
extern template class SplinesLinearProblem3x3Blocks<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
