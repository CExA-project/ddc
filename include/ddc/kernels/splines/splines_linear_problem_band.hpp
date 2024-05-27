// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <string>

#if __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A band linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense row-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed. Note: atm this is a placeholder for futur developments on GPU.
 */
template <class ExecSpace>
class SplinesLinearProblemBand : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using SplinesLinearProblem<ExecSpace>::size;

protected:
    const std::size_t m_kl; // no. of subdiagonals
    const std::size_t m_ku; // no. of superdiagonals
    const std::size_t m_c; // no. of columns in q
    Kokkos::View<int*, Kokkos::HostSpace> m_ipiv; // pivot indices
    Kokkos::View<double*, Kokkos::HostSpace> m_q; // banded matrix representation

public:
    /**
     * @brief SplinesLinearProblemBand constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemBand(
            const std::size_t mat_size,
            const std::size_t kl,
            const std::size_t ku)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_kl(kl)
        , m_ku(ku)
        , m_c(2 * kl + ku + 1)
        , m_ipiv("ipiv", mat_size)
        , m_q("q", m_c * mat_size)
    {
        assert(m_kl <= mat_size);
        assert(m_ku <= mat_size);

        /*
         * Given the linear system A*x=b, we assume that A is a square (n by n)
         * with ku super-diagonals and kl sub-diagonals.
         * All non-zero elements of A are stored in the rectangular matrix q, using
         * the format required by DGBTRF (LAPACK): diagonals of A are rows of q.
         * q has 2*kl rows for the subdiagonals, 1 row for the diagonal, and ku rows
         * for the superdiagonals. (The kl additional rows are needed for pivoting.)
         * The term A(i,j) of the full matrix is stored in q(i-j+2*kl+1,j).
         */
        Kokkos::deep_copy(m_q, 0.);
    }

    virtual double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());
        if ((std::ptrdiff_t)i >= (std::ptrdiff_t)j - (std::ptrdiff_t)m_ku && i < j + m_kl + 1) {
            return m_q(j * m_c + m_kl + m_ku + i - j);
        } else {
            return 0.0;
        }
    }

    virtual void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());
        if ((std::ptrdiff_t)i >= (std::ptrdiff_t)j - (std::ptrdiff_t)m_ku && i < j + m_kl + 1) {
            m_q(j * m_c + m_kl + m_ku + i - j) = aij;
        } else {
            assert(std::fabs(aij) < 1e-20);
        }
    }

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dgbtrf() implementation.
     */
    void setup_solver() override
    {
        int const info = LAPACKE_dgbtrf(
                LAPACK_COL_MAJOR,
                size(),
                size(),
                m_kl,
                m_ku,
                m_q.data(),
                m_c,
                m_ipiv.data());
        if (info != 0) {
            throw std::runtime_error("LAPACKE_dgbtrf failed with error code " + std::to_string(info));
        }
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgbtrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        auto b_host = create_mirror_view(Kokkos::DefaultHostExecutionSpace(), b);
        Kokkos::deep_copy(b_host, b);
        int const info = LAPACKE_dgbtrs(
                LAPACK_ROW_MAJOR,
                transpose ? 'T' : 'N',
                b_host.extent(0),
                m_kl,
                m_ku,
                b_host.extent(1),
                m_q.data(),
                m_c,
                m_ipiv.data(),
                b_host.data(),
                b_host.stride(0));
        if (info != 0) {
            throw std::runtime_error("LAPACKE_dgbtrs failed with error code " + std::to_string(info));
        }
        Kokkos::deep_copy(b, b_host);
    }
};

} // namespace ddc::detail
