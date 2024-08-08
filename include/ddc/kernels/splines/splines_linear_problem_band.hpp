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

// FIXME cannot find appropriate header file for Serial Gemv
#include <KokkosBatched_Gbtrs.hpp>
#include <KokkosBlas2_serial_gemv_impl.hpp>
#include <KokkosBlas2_serial_gemv_internal.hpp>

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
    using typename SplinesLinearProblem<ExecSpace>::Coo;
    using typename SplinesLinearProblem<ExecSpace>::AViewType;
    using typename SplinesLinearProblem<ExecSpace>::PivViewType;
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
    {
        assert(m_kl <= mat_size);
        assert(m_ku <= mat_size);
        this->m_a = AViewType("a", 2 * kl + ku + 1, mat_size);
        this->m_ipiv = PivViewType("ipiv", mat_size);

        Kokkos::deep_copy(this->m_a.h_view, 0.);
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
            return this->m_a.h_view(band_storage_row_index(i, j), j);
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
            this->m_a.h_view(band_storage_row_index(i, j), j) = aij;
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
                this->m_a.h_view.data(),
                this->m_a.h_view.stride(
                        0), // m_q.h_view.stride(0) if LAPACK_ROW_MAJOR, m_q.h_view.stride(1) if LAPACK_COL_MAJOR
                this->m_ipiv.h_view.data());
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dgbtrf failed with error code " + std::to_string(info));
        }

        // Convert 1-based index to 0-based index
        for (int i = 0; i < size(); ++i) {
            this->m_ipiv.h_view(i) -= 1;
        }

        // Push on device
        this->m_a.modify_host();
        this->m_a.sync_device();
        this->m_ipiv.modify_host();
        this->m_ipiv.sync_device();
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
        auto a_device = this->m_a.d_view;
        auto ipiv_device = this->m_ipiv.d_view;

        std::string name = "KokkosBatched::SerialGbtrs";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        if (transpose) {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(a_device, sub_b, ipiv_device, kl_proxy, ku_proxy);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(a_device, sub_b, ipiv_device, kl_proxy, ku_proxy);
                    });
        }
    }

    void solve(
            typename AViewType::t_dev top_right_block,
            typename AViewType::t_dev bottom_left_block,
            typename AViewType::t_dev bottom_right_block,
            typename PivViewType::t_dev bottom_right_piv,
            MultiRHS b,
            bool const transpose) const override
    {
        std::size_t kl_proxy = m_kl;
        std::size_t ku_proxy = m_ku;
        auto Q = this->m_a.d_view;
        auto piv = this->m_ipiv.d_view;

        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);

        std::string name = "KokkosBatched::SerialGbtrs-Gemv";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        if (transpose) {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        KokkosBlas::SerialGemv<
                                KokkosBlas::Trans::Transpose,
                                KokkosBlas::Algo::Gemv::Unblocked>::
                                invoke(-1.0, top_right_block, sub_b1, 1.0, sub_b2);

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(bottom_right_block, bottom_right_piv, sub_b2);

                        KokkosBlas::SerialGemv<
                                KokkosBlas::Trans::Transpose,
                                KokkosBlas::Algo::Gemv::Unblocked>::
                                invoke(-1.0, bottom_left_block, sub_b2, 1.0, sub_b1);

                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(Q, sub_b1, piv, kl_proxy, ku_proxy);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(Q, sub_b1, piv, kl_proxy, ku_proxy);

                        KokkosBlas::SerialGemv<
                                KokkosBlas::Trans::NoTranspose,
                                KokkosBlas::Algo::Gemv::Unblocked>::
                                invoke(-1.0, bottom_left_block, sub_b1, 1.0, sub_b2);

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(bottom_right_block, bottom_right_piv, sub_b2);

                        KokkosBlas::SerialGemv<
                                KokkosBlas::Trans::NoTranspose,
                                KokkosBlas::Algo::Gemv::Unblocked>::
                                invoke(-1.0, top_right_block, sub_b2, 1.0, sub_b1);
                    });
        }
    }

    void solve(
            Coo top_right_block,
            Coo bottom_left_block,
            typename AViewType::t_dev bottom_right_block,
            typename PivViewType::t_dev bottom_right_piv,
            MultiRHS b,
            bool const transpose) const override
    {
        std::size_t kl_proxy = m_kl;
        std::size_t ku_proxy = m_ku;
        auto Q = this->m_a.d_view;
        auto piv = this->m_ipiv.d_view;

        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);

        std::string name = "KokkosBatched::SerialGbtrs-Spmv";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        if (transpose) {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        for (int nz_idx = 0; nz_idx < top_right_block.nnz(); ++nz_idx) {
                            const int r = top_right_block.rows_idx()(nz_idx);
                            const int c = top_right_block.cols_idx()(nz_idx);
                            sub_b2(c) -= top_right_block.values()(nz_idx) * sub_b1(r);
                        }

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(bottom_right_block, bottom_right_piv, sub_b2);

                        for (int nz_idx = 0; nz_idx < bottom_left_block.nnz(); ++nz_idx) {
                            const int r = bottom_left_block.rows_idx()(nz_idx);
                            const int c = bottom_left_block.cols_idx()(nz_idx);
                            sub_b1(c) -= bottom_left_block.values()(nz_idx) * sub_b2(r);
                        }

                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(Q, sub_b1, piv, kl_proxy, ku_proxy);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        KokkosBatched::SerialGbtrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::
                                invoke(Q, sub_b1, piv, kl_proxy, ku_proxy);

                        for (int nz_idx = 0; nz_idx < bottom_left_block.nnz(); ++nz_idx) {
                            const int r = bottom_left_block.rows_idx()(nz_idx);
                            const int c = bottom_left_block.cols_idx()(nz_idx);
                            sub_b2(r) -= bottom_left_block.values()(nz_idx) * sub_b1(c);
                        }

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(bottom_right_block, bottom_right_piv, sub_b2);

                        for (int nz_idx = 0; nz_idx < top_right_block.nnz(); ++nz_idx) {
                            const int r = top_right_block.rows_idx()(nz_idx);
                            const int c = top_right_block.cols_idx()(nz_idx);
                            sub_b1(r) -= top_right_block.values()(nz_idx) * sub_b2(c);
                        }
                    });
        }
    }
};

} // namespace ddc::detail
