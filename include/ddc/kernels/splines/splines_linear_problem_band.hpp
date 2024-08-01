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

#include <KokkosBatched_Gbtrs.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A band linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense row-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * Given the linear system A*x=b, we assume that A is a square (n by n)
 * with ku superdiagonals and kl subdiagonals.
 * All non-zero elements of A are stored in the rectangular matrix q, using
 * the format required by DGBTRF (LAPACK): diagonals of A are rows of q.
 * q has 2*kl rows for the subdiagonals, 1 row for the diagonal, and ku rows
 * for the superdiagonals. (The kl additional rows are needed for pivoting.)
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
template <class ExecSpace>
class SplinesLinearProblemBand : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using SplinesLinearProblem<ExecSpace>::size;

protected:
    std::size_t m_kl; // no. of subdiagonals
    std::size_t m_ku; // no. of superdiagonals
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
            m_q; // band matrix representation
    Kokkos::DualView<int*, typename ExecSpace::memory_space> m_ipiv; // pivot indices

public:
    /**
     * @brief SplinesLinearProblemBand constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param kl The number of subdiagonals of the matrix.
     * @param ku The number of superdiagonals of the matrix.
     */
    explicit SplinesLinearProblemBand(
            std::size_t const mat_size,
            std::size_t const kl,
            std::size_t const ku)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_kl(kl)
        , m_ku(ku)
        /*
         * The matrix itself stored in band format requires a (kl + ku + 1)*mat_size
         * allocation, but the LU-factorization requires an additional kl*mat_size block
         */
        , m_q("q", 2 * kl + ku + 1, mat_size)
        , m_ipiv("ipiv", mat_size)
    {
        assert(m_kl <= mat_size);
        assert(m_ku <= mat_size);

        Kokkos::deep_copy(m_q.h_view, 0.);
    }

private:
    std::size_t band_storage_row_index(std::size_t const i, std::size_t const j) const
    {
        return m_kl + m_ku + i - j;
    }

public:
    double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());
        /*
         * The "row index" of the band format storage identify the (sub/super)-diagonal
         * while the column index is actually the column index of the matrix. Two layouts
         * are supported by LAPACKE. The m_kl first lines are irrelevant for the storage of
         * the matrix itself but required for the storage of its LU factorization.
         */
        if (i >= std::
                            max(static_cast<std::ptrdiff_t>(0),
                                static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>(m_ku))
            && i < std::min(size(), j + m_kl + 1)) {
            return m_q.h_view(band_storage_row_index(i, j), j);
        } else {
            return 0.0;
        }
    }

    void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());
        /*
         * The "row index" of the band format storage identify the (sub/super)-diagonal
         * while the column index is actually the column index of the matrix. Two layouts
         * are supported by LAPACKE. The m_kl first lines are irrelevant for the storage of
         * the matrix itself but required for the storage of its LU factorization.
         */
        if (i >= std::
                            max(static_cast<std::ptrdiff_t>(0),
                                static_cast<std::ptrdiff_t>(j) - static_cast<std::ptrdiff_t>(m_ku))
            && i < std::min(size(), j + m_kl + 1)) {
            m_q.h_view(band_storage_row_index(i, j), j) = aij;
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
                LAPACK_ROW_MAJOR,
                size(),
                size(),
                m_kl,
                m_ku,
                m_q.h_view.data(),
                m_q.h_view.stride(
                        0), // m_q.h_view.stride(0) if LAPACK_ROW_MAJOR, m_q.h_view.stride(1) if LAPACK_COL_MAJOR
                m_ipiv.h_view.data());
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dgbtrf failed with error code " + std::to_string(info));
        }

        // Convert 1-based index to 0-based index
        for (int i = 0; i < size(); ++i) {
            m_ipiv.h_view(i) -= 1;
        }

        // Push on device
        m_q.modify_host();
        m_q.sync_device();
        m_ipiv.modify_host();
        m_ipiv.sync_device();
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgbtrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS const b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        std::size_t kl_proxy = m_kl;
        std::size_t ku_proxy = m_ku;
        auto q_device = m_q.d_view;
        auto ipiv_device = m_ipiv.d_view;
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        if (transpose) {
            Kokkos::parallel_for(
                    "gbtrs",
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(q_device, sub_b, ipiv_device, kl_proxy, ku_proxy);
                    });
        } else {
            Kokkos::parallel_for(
                    "gbtrs",
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(q_device, sub_b, ipiv_device, kl_proxy, ku_proxy);
                    });
        }
    }
};

} // namespace ddc::detail
