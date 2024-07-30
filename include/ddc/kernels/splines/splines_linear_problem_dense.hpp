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
#include <KokkosBatched_Getrs.hpp>
#include <KokkosBatched_Util.hpp>
#include <KokkosBlas2_serial_gemv_impl.hpp>
#include <KokkosBlas2_serial_gemv_internal.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A dense linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is dense row-major. Lapack is used to perform every matrix and linear solver-related operations.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are supposed to be performed.
 */
template <class ExecSpace>
class SplinesLinearProblemDense : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using typename SplinesLinearProblem<ExecSpace>::Coo;
    using typename SplinesLinearProblem<ExecSpace>::AViewType;
    using typename SplinesLinearProblem<ExecSpace>::PivViewType;
    using SplinesLinearProblem<ExecSpace>::size;

public:
    /**
     * @brief SplinesLinearProblemDense constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemDense(std::size_t const mat_size)
        : SplinesLinearProblem<ExecSpace>(mat_size)
    {
        this->m_a = AViewType("a", mat_size, mat_size);
        this->m_ipiv = PivViewType("ipiv", mat_size);
        Kokkos::deep_copy(this->m_a.h_view, 0.);
    }

    double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());
        return this->m_a.h_view(i, j);
    }

    void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());
        this->m_a.h_view(i, j) = aij;
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
                this->m_a.h_view.data(),
                size(),
                this->m_ipiv.h_view.data());
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dgetrf failed with error code " + std::to_string(info));
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
     * The solver method is gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgetrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS const b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        // For order 1 splines, size() can be 0 then we bypass the solver call.
        if (size() == 0)
            return;

        auto a_device = this->m_a.d_view;
        auto ipiv_device = this->m_ipiv.d_view;

        std::string name = "KokkosBatched::SerialGetrs";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        if (transpose) {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(a_device, ipiv_device, sub_b);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(a_device, ipiv_device, sub_b);
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
        auto Q = this->m_a.d_view;
        auto piv = this->m_ipiv.d_view;
        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);
        std::string name = "KokkosBatched::SerialGetrs-Gemm";
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

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Getrs::Unblocked>::invoke(Q, piv, sub_b1);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::invoke(Q, piv, sub_b1);

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
        auto piv = this->m_ipiv.d_view;
        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);
        std::string name = "KokkosBatched::SerialGetrs-Spdm";
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

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Getrs::Unblocked>::invoke(Q, piv, sub_b1);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gbtrs::Unblocked>::invoke(Q, piv, sub_b1);

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
