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
 * @brief The parent class for a linear problem dedicated to the computation of a spline approximation.
 *
 * It stores a square matrix and provides methods to solve a multiple right-hand sides linear problem.
 * Implementations may have different storage formats, filling methods and multiple right-hand sides linear solvers.
 */
template <class ExecSpace>
class SplinesLinearProblem
{
public:
    /// @brief The type of a Kokkos::View storing multiple right-hand sides.
    using MultiRHS = Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>;

private:
    std::size_t m_size;

protected:
    explicit SplinesLinearProblem(const std::size_t size) : m_size(size) {}

public:
    /// @brief Destruct
    virtual ~SplinesLinearProblem() = default;

    /**
     * @brief Get an element of the matrix at indexes i, j. It must not be called after `setup_solver`.
     *
     * @param i The row index of the desired element.
     * @param j The column index of the desired element.
     *
     * @return The value of the element of the matrix.
     */
    virtual double get_element(int i, int j) const = 0;

    /**
     * @brief Set an element of the matrix at indexes i, j. It must not be called after `setup_solver`.
     *
     * @param i The row index of the setted element.
     * @param j The column index of the setted element.
     * @param aij The value to set in the element of the matrix.
     */
    virtual void set_element(int i, int j, double aij) = 0;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     */
    virtual void setup_solver() = 0;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * @param[in, out] multi_rhs A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    virtual void solve(MultiRHS b, bool transpose = false) const = 0;

    /**
     * @brief Get the size of the square matrix in one of its dimensions.
     *
     * @return The size of the matrix in one of its dimensions.
     */
    int size() const
    {
        return m_size;
    }

    /**
     * @brief Prints a Matrix in a std::ostream. It must not be called after `setup_solver`.
     *
     * @param out The stream in which the matrix is printed.
     *
     * @return The stream in which the matrix is printed.
     **/
    std::ostream& operator<<(std::ostream& os)
    {
        int const n = size();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                os << std::fixed << std::setprecision(3) << std::setw(10) << get_element(i, j);
            }
            os << std::endl;
        }
        return os;
    }
};

} // namespace ddc::detail
