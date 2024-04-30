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

class Matrix
{
    int m_n;

public:
    explicit Matrix(const int mat_size) : m_n(mat_size) {}

    /// @brief Destruct
    virtual ~Matrix() = default;

    /**
     * @brief Get an element of the matrix at indexes i, j.
     *
     * @param i The row index of the desired element.
     * @param j The columns index of the desired element.
     *
     * @return The value of the element of the matrix.
     */
    virtual double get_element(int i, int j) const = 0;

    /**
     * @brief Set an element of the matrix at indexes i, j.
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
    virtual void factorize()
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
     * @brief Solve the linear problem Ax=b inplace.
     *
     * @param[in, out] b A 1D mdpsan storing a right-hand-side of the problem and receiving the corresponding solution.
     */
    virtual ddc::DSpan1D solve_inplace(ddc::DSpan1D const b) const
    {
        assert(int(b.extent(0)) == m_n);
        int const info = solve_inplace_method(b.data_handle(), 'N', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }

    /**
     * @brief Solve the transposed linear problem A^tx=b inplace.
     *
     * @param[in, out] b A 1D mdpsan storing a right-hand-side of the problem and receiving the corresponding solution.
     *
     * @return b
     */
    virtual ddc::DSpan1D solve_transpose_inplace(ddc::DSpan1D const b) const
    {
        assert(int(b.extent(0)) == m_n);
        int const info = solve_inplace_method(b.data_handle(), 'T', 1);

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return b;
    }

    /**
     * @brief Solve the multiple-right-hand-side linear problem Ax=b inplace.
     *
     * @param[in, out] bx A 2D mdpsan storing the multiple right-hand-sides of the problem and receiving the corresponding solution. Important note: the convention is the reverse of the common matrix one,
     * row number is second index and column number the first one. It should be changed in the future.
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
     * @brief Solve the transposed multiple-right-hand-side linear problem A^tx=b inplace.
     *
     * @param[in, out] bx A 2D mdpsan storing the multiple right-hand-sides of the problem and receiving the corresponding solution. Important note: the convention is the reverse of the common matrix one,
     * row number is second index and column number the first one. It should be changed in the future.
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
     * @brief Solve the multiple-right-hand-side linear problem Ax=b inplace.
     *
     * @param[in, out] bx A 2D Kokkos::View storing the multiple right-hand-sides of the problem and receiving the corresponding solution. Important note: the convention is the reverse of the common matrix one,
     * row number is second index and column number the first one. It should be changed in the future.
     *
     * @return bx
     */
    template <class... Args>
    Kokkos::View<double**, Args...> solve_batch_inplace(
            Kokkos::View<double**, Args...> const bx) const
    {
        assert(int(bx.extent(0)) == m_n);
        int const info = solve_inplace_method(bx.data(), 'N', bx.extent(1));

        if (info < 0) {
            std::cerr << -info << "-th argument had an illegal value" << std::endl;
            // TODO: Add LOG_FATAL_ERROR
        }
        return bx;
    }

    /**
     * @brief Get the size of the matrix (which is necessarily squared) in one of its dimensions.
     *
     * @return The size of the matrix in one of its dimensions.
     */
    int get_size() const
    {
        return m_n;
    }

    /**
     * @brief Prints a Matrix in a std::ostream. It will segfault is the Matrix is on GPU.
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
     * @brief A function called by factorize() to actually perform the pre-process operation.
     *
     * @param b A double* to a contiguous array containing the (eventually multiple) right-hand-sides. Memory layout depends on the
     * derivated class (and the underlying algorithm).
     * @param transpose A character identifying if the normal ('N') or transposed ('T') linear system is solved.
     * @param n_equations The number of multiple-right-hand-sides (number of columns of b).
     * @return The error code of the function.
     */
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const = 0;
};

} // namespace ddc::detail
