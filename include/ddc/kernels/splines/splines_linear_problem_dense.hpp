// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#if __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif

#include <KokkosBatched_Util.hpp>

#include "kokkos-kernels-ext/KokkosBatched_Getrs.hpp"

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
    using SplinesLinearProblem<ExecSpace>::size;

protected:
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space> m_a;
    Kokkos::DualView<int*, typename ExecSpace::memory_space> m_ipiv;

public:
    /**
     * @brief SplinesLinearProblemDense constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemDense(std::size_t const mat_size)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_a("a", mat_size, mat_size)
        , m_ipiv("ipiv", mat_size)
    {
        Kokkos::deep_copy(m_a.view_host(), 0.);
    }

    double get_element(std::size_t const i, std::size_t const j) const override
    {
        assert(i < size());
        assert(j < size());
        return m_a.view_host()(i, j);
    }

    void set_element(std::size_t const i, std::size_t const j, double const aij) override
    {
        assert(i < size());
        assert(j < size());
        m_a.view_host()(i, j) = aij;
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
                m_a.view_host().data(),
                size(),
                m_ipiv.view_host().data());
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dgetrf failed with error code " + std::to_string(info));
        }

        // Convert 1-based index to 0-based index
        for (int i = 0; i < size(); ++i) {
            m_ipiv.view_host()(i) -= 1;
        }

        // Push on device
        m_a.modify_host();
        m_a.sync_device();
        m_ipiv.modify_host();
        m_ipiv.sync_device();
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
        if (size() == 0) {
            return;
        }

        auto a_device = m_a.view_device();
        auto ipiv_device = m_ipiv.view_device();

        Kokkos::RangePolicy<ExecSpace> const policy(0, b.extent(1));

        if (transpose) {
            Kokkos::parallel_for(
                    "gerts",
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::Transpose,
                                KokkosBatched::Algo::Level3::Unblocked>::
                                invoke(a_device, ipiv_device, sub_b);
                    });
        } else {
            Kokkos::parallel_for(
                    "gerts",
                    policy,
                    KOKKOS_LAMBDA(const int i) {
                        auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                        KokkosBatched::SerialGetrs<
                                KokkosBatched::Trans::NoTranspose,
                                KokkosBatched::Algo::Level3::Unblocked>::
                                invoke(a_device, ipiv_device, sub_b);
                    });
        }
    }
};

} // namespace ddc::detail
