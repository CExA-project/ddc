// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "view.hpp"

namespace ddc::detail {

/**
 * @brief The parent class for a linear problem dedicated to compute a spline approximation.
 *
 * It represents a square Matrix and provides methods to solve a multiple right-hand sides linear problem.
 * Implementations may have different storage formats, filling methods and multiple right-hand sides linear solvers.
 */
class SplinesLinearProblem
{
public:
    /// @brief The type of a Kokkos::View storing multiple right-hand sides.
    using MultiRHS = Kokkos::View<double**, typename ExecSpace::memory_space, Kokkos::LayoutRight>;

private:
    std::size_t m_size;

protected:
    explicit SplinesLinearProblem(const std::size_t size) : m_size(size) {}

public:
    /// @brief Destruct
    virtual ~SplinesLinearProblem() = default;

    /**
     * @brief Get an element of the matrix at indexes i, j. It must not be called after `factorize`.
     *
     * @param i The row index of the desired element.
     * @param j The column index of the desired element.
     *
     * @return The value of the element of the matrix.
     */
    virtual double get_element(int i, int j) const = 0;

    /**
     * @brief Set an element of the matrix at indexes i, j. It must not be called after `factorize`.
     *
     * @param i The row index of the setted element.
     * @param j The column index of the setted element.
     * @param aij The value to set in the element of the matrix.
     */
    virtual void set_element(int i, int j, double aij) = 0;

    /**
     * @brief Performs a pre-process operation on the Matrix.
     *
     * Note: this function should be renamed in the future because the pre-
     * process operation is not necessarily a factorization.
     */
    virtual void finished_filling()
    {
        int const info = factorize_method();

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        } else if (info > 0) {
            std::cerr << "U(" << info << "," << info << ") is exactly zero.";
            std::cerr << " The factorization has been completed, but the factor";
            std::cerr << " U is exactly singular, and division by zero will occur "
                         "if "
                         "it is used to";
            std::cerr << " solve a system of equations.";
            // TODO: Add LOG_FATAL_ERROR
        }
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b inplace.
     *
     * @param[in, out] bx A 2D mdpsan storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * Important note: the convention is the reverse of the common matrix one, row number is second index and column number
     * the first one. This means when solving `A x_i = b_i`,  element `(b_i)_j` is stored in `b(j, i)`.
     *
     * @return bx
     */
    virtual ddc::DSpan2D solve_multiple_inplace(ddc::DSpan2D const bx) const
    {
        assert(int(bx.extent(1)) == m_n);
        int const info = solve_inplace_method(bx.data_handle(), 'N', bx.extent(0));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    /**
     * @brief Solve the transposed multiple right-hand sides linear problem A^tx=b inplace.
     *
     * @param[in, out] bx A 2D mdspan storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * Important note: the convention is the reverse of the common matrix one, row number is second index and column number
     * the first one. It should be changed in the future.
     *
     * @return bx
     */
    virtual ddc::DSpan2D solve_multiple_transpose_inplace(ddc::DSpan2D const bx) const
    {
        assert(int(bx.extent(1)) == m_n);
        int const info = solve_inplace_method(bx.data_handle(), 'T', bx.extent(0));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    /**
     * @brief Get the size of the square matrix in one of its dimensions.
     *
     * @return The size of the matrix in one of its dimensions.
     */
    int size() const
    {
        return m_n;
    }

    /**
     * @brief Prints a Matrix in a std::ostream. It must not be called after `factorize`.
     *
     * @param out The stream in which the matrix is printed.
     *
     * @return The stream in which the matrix is printed.
     **/
    std::ostream& operator<<(std::ostream& os)
    {
        int const n = get_size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                os << std::fixed << std::setprecision(3) << std::setw(10) << get_element(i, j);
            }
            os << std::endl;
        }
        return os;
    }

protected:
    /**
     * @brief A function called by factorize() to actually perform the pre-process operation.
     *
     * @return The error code of the function.
     */
    virtual int factorize_method() = 0;

    /**
     * @brief A function called by solve_inplace() and similar functions to actually perform the linear solve operation.
     *
     * @param b A double* to a contiguous array containing the (eventually multiple) right-hand-sides. The memory layout is right.
     * @param transpose A character identifying if the normal ('N') or transposed ('T') linear system is solved.
     * @param n_equations The number of multiple-right-hand-sides (number of columns of b).
     * @return The error code of the function.
     */
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const = 0;
};

} // namespace ddc::detail
