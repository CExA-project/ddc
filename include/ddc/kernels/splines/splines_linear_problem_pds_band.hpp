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
    using typename SplinesLinearProblem<ExecSpace>::AViewType;
    using typename SplinesLinearProblem<ExecSpace>::PivViewType;
    using SplinesLinearProblem<ExecSpace>::size;

    //protected:
    //    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
    //            m_q; // pds band matrix representation

public:
    /**
     * @brief SplinesLinearProblemPDSBand constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param kd The number of sub/superdiagonals of the matrix.
     */
    explicit SplinesLinearProblemPDSBand(std::size_t const mat_size, std::size_t const kd)
        : SplinesLinearProblem<ExecSpace>(mat_size)
    {
        assert(this->m_a.extent(0) <= mat_size);

        this->m_a = AViewType("a", kd + 1, mat_size);
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
        if (j - i < this->m_a.extent(0)) {
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
        if (j - i < this->m_a.extent(0)) {
            this->m_a.h_view(j - i, i) = aij;
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
                this->m_a.extent(0) - 1,
                this->m_a.h_view.data(),
                this->m_a.h_view.stride(
                        0) // m_q.h_view.stride(0) if LAPACK_ROW_MAJOR, m_q.h_view.stride(1) if LAPACK_COL_MAJOR
        );
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dpbtrf failed with error code " + std::to_string(info));
        }

        // Push on device
        this->m_a.modify_host();
        this->m_a.sync_device();
    }

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

        auto a_device = this->m_a.d_view;
        std::string name = "pbtrs";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        Kokkos::Profiling::pushRegion(name);
        Kokkos::parallel_for(
                name,
                policy,
                KOKKOS_CLASS_LAMBDA(const int i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialPbtrs<
                            KokkosBatched::Uplo::Lower,
                            KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(a_device, sub_b);
                });
        Kokkos::Profiling::popRegion();
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
        MultiRHS b1 = Kokkos::
                subview(b, std::pair<std::size_t, std::size_t>(0, this->size()), Kokkos::ALL);
        MultiRHS b2 = Kokkos::
                subview(b,
                        std::pair<std::size_t, std::size_t>(this->size(), b.extent(0)),
                        Kokkos::ALL);
        std::string name = "pbtrs";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        Kokkos::Profiling::pushRegion(name);
        if (transpose) {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_CLASS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);

                        KokkosBatched::SerialGemm<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gemm::Unblocked>::
                                invoke(-1.0, top_right_block, sub_b1, 1.0, sub_b2);

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(bottom_right_block, bottom_right_piv, sub_b2);

                        KokkosBatched::SerialGemm<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gemm::Unblocked>::
                                invoke(-1.0, bottom_left_block, sub_b2, 1.0, sub_b1);

                        KokkosBatched::SerialPbtrs<
                                KokkosBatched::Uplo::Lower,
                                KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(Q, sub_b1);
                    });
        } else {
            Kokkos::parallel_for(
                    name,
                    policy,
                    KOKKOS_CLASS_LAMBDA(const int i) {
                        auto sub_b1 = Kokkos::subview(b1, Kokkos::ALL, i);
                        auto sub_b2 = Kokkos::subview(b2, Kokkos::ALL, i);
                        KokkosBatched::SerialPbtrs<
                                KokkosBatched::Uplo::Lower,
                                KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(Q, sub_b1);

                        KokkosBatched::SerialGemm<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gemm::Unblocked>::
                                invoke(-1.0, bottom_left_block, sub_b1, 1.0, sub_b2);

                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Getrs::Unblocked>::
                                invoke(bottom_right_block, bottom_right_piv, sub_b2);

                        KokkosBatched::SerialGemm<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Gemm::Unblocked>::
                                invoke(-1.0, top_right_block, sub_b2, 1.0, sub_b1);
                    });
        }
        Kokkos::Profiling::popRegion();
    }
};

} // namespace ddc::detail
