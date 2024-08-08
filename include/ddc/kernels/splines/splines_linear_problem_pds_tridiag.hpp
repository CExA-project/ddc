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
#include <KokkosBatched_Pttrs.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBlas2_serial_gemv_impl.hpp>
#include <KokkosBlas2_serial_gemv_internal.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A positive-definite symmetric tridiagonal linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense row-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * Given the linear system A*x=b, we assume that A is a square (n by n)
 * with 1 subdiagonal and 1 superdiagonal.
 * All non-zero elements of A are stored in the rectangular matrix q, using
 * the format required by DPTTRF (LAPACK): diagonal and superdiagonal of A are rows of q.
 * q has 1 row for the diagonal and 1 row for the superdiagonal.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
template <class ExecSpace>
class SplinesLinearProblemPDSTridiag : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using typename SplinesLinearProblem<ExecSpace>::Coo;
    using typename SplinesLinearProblem<ExecSpace>::AViewType;
    using typename SplinesLinearProblem<ExecSpace>::PivViewType;
    using SplinesLinearProblem<ExecSpace>::size;

public:
    /**
     * @brief SplinesLinearProblemPDSTridiag constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemPDSTridiag(std::size_t const mat_size)
        : SplinesLinearProblem<ExecSpace>(mat_size)
    {
        this->m_a = AViewType("a", 2, mat_size);
        Kokkos::deep_copy(this->m_a.h_view, 0.);
    }

    double get_element(std::size_t i, std::size_t j) const override
    {
        assert(i < size());
        assert(j < size());

        // Indices are swapped for an element on subdiagonal
        if (i > j) {
            std::swap(i, j);
        }
        if (j - i < 2) {
            return this->m_a.h_view(j - i, i);
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
        if (j - i < 2) {
            this->m_a.h_view(j - i, i) = aij;
        } else {
            assert(std::fabs(aij) < 1e-20);
        }
    }

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dpttrf() implementation.
     */
    void setup_solver() override
    {
        int const info = LAPACKE_dpttrf(
                size(),
                this->m_a.h_view.data(),
                this->m_a.h_view.data() + this->m_a.h_view.stride(0));
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dpttrf failed with error code " + std::to_string(info));
        }

        // Push on device
        this->m_a.modify_host();
        this->m_a.sync_device();
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dpttrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem (unused for a symmetric problem).
     */
    void solve(MultiRHS const b, bool const) const override
    {
        assert(b.extent(0) == size());
        auto a_device = this->m_a.d_view;
        auto d = Kokkos::subview(a_device, 0, Kokkos::ALL);
        auto e = Kokkos::subview(a_device, 1, Kokkos::pair<int, int>(0, a_device.extent(1) - 1));
        std::string name = "KokkosBatched::SerialPttrs";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        Kokkos::parallel_for(
                name,
                policy,
                KOKKOS_LAMBDA(const int i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialPttrs<
                            KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b);
                });
    }

    void solve(
            typename AViewType::t_dev top_right_block,
            typename AViewType::t_dev bottom_left_block,
            typename AViewType::t_dev bottom_right_block,
            typename PivViewType::t_dev bottom_right_piv,
            MultiRHS b,
            bool const transpose) const override
    {
        auto Q = this->m_a.d_view;
        auto d = Kokkos::subview(Q, 0, Kokkos::ALL);
        auto e = Kokkos::subview(Q, 1, Kokkos::pair<int, int>(0, Q.extent(1) - 1));
        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);

        std::string name = "KokkosBatched::SerialPttrs-Gemv";
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

                        KokkosBatched::SerialPttrs<
                                KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b1);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);
                        KokkosBatched::SerialPttrs<
                                KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b1);

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
        auto Q = this->m_a.d_view;
        auto d = Kokkos::subview(Q, 0, Kokkos::ALL);
        auto e = Kokkos::subview(Q, 1, Kokkos::pair<int, int>(0, Q.extent(1) - 1));
        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);

        std::string name = "KokkosBatched::SerialPttrs-Spmv";
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

                        KokkosBatched::SerialPttrs<
                                KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b1);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);
                        KokkosBatched::SerialPttrs<
                                KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b1);

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
