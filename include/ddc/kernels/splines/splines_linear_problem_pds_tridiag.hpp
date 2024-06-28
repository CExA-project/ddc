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

#include <KokkosBatched_Pttrs.hpp>
#include <KokkosBatched_Util.hpp>

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
    using SplinesLinearProblem<ExecSpace>::size;

protected:
    Kokkos::DualView<double**, Kokkos::LayoutRight, typename ExecSpace::memory_space>
            m_q; // pds tridiagonal matrix representation

public:
    /**
     * @brief SplinesLinearProblemPDSTridiag constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     */
    explicit SplinesLinearProblemPDSTridiag(std::size_t const mat_size)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_q("q", 2, mat_size)
    {
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
        if (j - i < 2) {
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
        if (j - i < 2) {
            m_q.h_view(j - i, i) = aij;
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
                m_q.h_view.data(),
                m_q.h_view.data() + m_q.h_view.stride(0));
        if (info != 0) {
            throw std::runtime_error(
                    "LAPACKE_dpttrf failed with error code " + std::to_string(info));
        }

        // Push on device
        m_q.modify_host();
        m_q.sync_device();
    }

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dpttrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem (unused for a symmetric problem).
     */
    void solve(MultiRHS b, bool const) const override
    {
        assert(b.extent(0) == size());
        auto q_device = m_q.d_view;
        auto d = Kokkos::subview(q_device, 0, Kokkos::ALL);
        auto e = Kokkos::subview(q_device, 1, Kokkos::pair<int, int>(0, q_device.extent(1) - 1));
        std::string name = "pttrs";
        Kokkos::RangePolicy<ExecSpace> policy(0, b.extent(1));
        Kokkos::Profiling::pushRegion(name);
        Kokkos::parallel_for(
                name,
                policy,
                KOKKOS_CLASS_LAMBDA(const int i) {
                    auto sub_b = Kokkos::subview(b, Kokkos::ALL, i);
                    KokkosBatched::SerialPttrs<
                            KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, sub_b);
                });
        Kokkos::Profiling::popRegion();
    }
};

} // namespace ddc::detail
