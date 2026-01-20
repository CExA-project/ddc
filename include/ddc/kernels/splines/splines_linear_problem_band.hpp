// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#if !defined(NDEBUG)
#    include <cmath>
#endif
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#if __has_include(<mkl_lapacke.h>)
#    include <mkl_lapacke.h>
#else
#    include <lapacke.h>
#endif

#include <KokkosBatched_Util.hpp>

#include "kokkos-kernels-ext/KokkosBatched_Gbtrs.hpp"

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
    explicit SplinesLinearProblemBand(std::size_t mat_size, std::size_t kl, std::size_t ku);

    SplinesLinearProblemBand(SplinesLinearProblemBand const& rhs) = delete;

    SplinesLinearProblemBand(SplinesLinearProblemBand&& rhs) = delete;

    ~SplinesLinearProblemBand() override;

    SplinesLinearProblemBand& operator=(SplinesLinearProblemBand const& rhs) = delete;

    SplinesLinearProblemBand& operator=(SplinesLinearProblemBand&& rhs) = delete;

private:
    std::size_t band_storage_row_index(std::size_t i, std::size_t j) const;

public:
    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dgbtrf() implementation.
     */
    void setup_solver() override;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgbtrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool transpose) const override;
};

#if defined(KOKKOS_ENABLE_SERIAL)
extern template class SplinesLinearProblemBand<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
extern template class SplinesLinearProblemBand<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
extern template class SplinesLinearProblemBand<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
extern template class SplinesLinearProblemBand<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
extern template class SplinesLinearProblemBand<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
