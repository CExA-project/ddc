// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <string>

#include <Kokkos_DualView.hpp>

#if __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

#include <KokkosBatched_Pbtrs.hpp>
#include <KokkosBatched_Util.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A positive-definite symmetric band linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense row-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * Given the linear system A*x=b, we assume that A is a square (n by n)
 * with kd sub and superdiagonals.
 * All non-zero elements of A are stored in the rectangular matrix q, using
 * the format required by DPBTRF (LAPACK): (super-)diagonals of A are rows of q.
 * q has 1 row for the diagonal and kd rows for the superdiagonals.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
template <class ExecSpace>
class SplinesLinearProblemPDSBand : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using SplinesLinearProblem<ExecSpace>::size;

protected:
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
            m_q; // pds band matrix representation

public:
    /**
     * @brief SplinesLinearProblemPDSBand constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param kd The number of sub/superdiagonals of the matrix.
     */
    explicit SplinesLinearProblemPDSBand(std::size_t const mat_size, std::size_t const kd)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_q("q", kd + 1, mat_size)
    {
        assert(m_q.extent(0) <= mat_size);

        Kokkos::deep_copy(m_q.h_view, 0.);
    }

    double get_element(std::size_t i, std::size_t j) const override
    {
        assert(i < size());
        assert(j < size());

        // Indices are swapped for an element on subdiagonal
        if (i > j) {
            std::swap(i, j);
        }
        if (j - i < m_q.extent(0)) {
            return m_q.h_view(j - i, i);
        } else {
            return 0.0;
        }
    }

    void set_element(std::size_t i, std::size_t j, double const aij) override
    {
        assert(i < size());
        assert(j < size());

        // Indices are swapped for an element on subdiagonal
        if (i > j) {
            std::swap(i, j);
        }
        if (j - i < m_q.extent(0)) {
            m_q.h_view(j - i, i) = aij;
        } else {
            assert(std::fabs(aij) < 1e-20);
        }
    }

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dpbtrf() implementation.
     */
    void setup_solver() override
    {
        int const info = LAPACKE_dpbtrf(
                LAPACK_ROW_MAJOR,
                'L',
                size(),
                m_q.extent(0) - 1,
                m_q.h_view.data(),
                m_q.h_view.stride(
                        0) // m_q.h_view.stride(0) if LAPACK_ROW_MAJOR, m_q.h_view.stride(1) if LAPACK_COL_MAJOR
        );
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dpbtrf failed with error code " + std::to_string(info));
        }

        // Push on device
        m_q.modify_host();
        m_q.sync_device();
    }

private:
    KOKKOS_FUNCTION int tbsv(
            [[maybe_unused]] char const uplo,
            [[maybe_unused]] char const trans,
            [[maybe_unused]] char const diag,
            int const n,
            int const k,
            Kokkos::View<double**, Kokkos::LayoutStride, typename ExecSpace::memory_space> const a,
            [[maybe_unused]] int const lda,
            Kokkos::View<double*, Kokkos::LayoutStride, typename ExecSpace::memory_space> const x,
            [[maybe_unused]] int const incx) const
    {
        if (trans == 'N') {
            for (int j = 0; j < n; ++j) {
                if (x(j) != 0) {
                    x(j) /= a(0, j);
                    for (int i = j + 1; i <= Kokkos::min(n, j + k); ++i) {
                        x(i) -= a(i - j, j) * x(j);
                    }
                }
            }
        } else if (trans == 'T') {
            for (int j = n - 1; j >= 0; --j) {
                for (int i = Kokkos::min(n, j + k); i >= j + 1; --i) {
                    x(j) -= a(i - j, j) * x(i);
                }
                x(j) /= a(0, j);
            }
        }
        return 0;
    }

public:
    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dpbtrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem (unused for a symmetric problem).
     */
    void solve(MultiRHS b, bool const) const override
    {
        assert(b.extent(0) == size());

        auto q_device = m_q.d_view;
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        Kokkos::parallel_for(
                "pbtrs",
                policy,
                KOKKOS_CLASS_LAMBDA(const int i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialPbtrs<
                            KokkosBatched::Uplo::Lower,
                            KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(q_device, sub_b);
                });
    }
};

} // namespace ddc::detail
