// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A dense linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense row-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * @tparam Kokkos::Serial The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
class SplinesLinearProblemDense : public SplinesLinearProblem
{
public:
    using SplinesLinearProblem::size;
    using typename SplinesLinearProblem::MultiRHS;

protected:
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> m_a;
    Kokkos::DualView<int*, typename Kokkos::Serial::memory_space> m_ipiv;

public:
    /**
     * @brief SplinesLinearProblemDense constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemDense(std::size_t mat_size);

    ~SplinesLinearProblemDense() override;

    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dgetrf() implementation.
     */
    void setup_solver() override;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgetrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool transpose) const override;
};

} // namespace ddc::detail
