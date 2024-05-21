// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>

#include <lapacke.h>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A dense linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense column-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed. Note: atm this is a placeholder for futur developments on GPU.
 */
template <class ExecSpace>
class SplinesLinearProblemDense : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using SplinesLinearProblem<ExecSpace>::size;
    using SplinesLinearProblem<ExecSpace>::operator<<;

protected:
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> m_a;
    Kokkos::View<int*, Kokkos::HostSpace> m_ipiv;

public:
    /**
     * @brief SplinesLinearProblemDense constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemDense(const std::size_t mat_size)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_a("a", mat_size, mat_size)
        , m_ipiv("ipiv", mat_size)
    {
        assert(mat_size > 0);

        Kokkos::deep_copy(m_a, 0.);
    }

    virtual double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());
        return m_a(i, j);
    }

    virtual void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        m_a(i, j) = aij;
    }

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dgetrf() implementation.
     */
    void setup_solver() override
    {
        int const info = LAPACKE_dgetrf(
                LAPACK_ROW_MAJOR,
                size(),
                size(),
                m_a.data(),
                size(),
                m_ipiv.data());
        if (info != 0) {
            throw std::runtime_error("LAPACK failed with error code " + info);
        }
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgetrs.
     *
     * @param[in, out] multi_rhs A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        auto b_host = create_mirror_view(Kokkos::DefaultHostExecutionSpace(), b);
        Kokkos::deep_copy(b_host, b);
        int const info = LAPACKE_dgetrs(
                LAPACK_ROW_MAJOR,
                transpose ? 'T' : 'N',
                b_host.extent(0),
                b_host.extent(1),
                m_a.data(),
                b_host.extent(0),
                m_ipiv.data(),
                b_host.data(),
                b_host.stride(0));
        if (info != 0) {
            throw std::runtime_error("LAPACK failed with error code " + info);
        }
        Kokkos::deep_copy(b, b_host);
    }
};

} // namespace ddc::detail
