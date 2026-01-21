// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

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
    explicit SplinesLinearProblemPDSTridiag(std::size_t mat_size);

    SplinesLinearProblemPDSTridiag(SplinesLinearProblemPDSTridiag const& rhs) = delete;

    SplinesLinearProblemPDSTridiag(SplinesLinearProblemPDSTridiag&& rhs) = delete;

    ~SplinesLinearProblemPDSTridiag() override;

    SplinesLinearProblemPDSTridiag& operator=(SplinesLinearProblemPDSTridiag const& rhs) = delete;

    SplinesLinearProblemPDSTridiag& operator=(SplinesLinearProblemPDSTridiag&& rhs) = delete;

    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dpttrf() implementation.
     */
    void setup_solver() override;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is band gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dpttrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem (unused for a symmetric problem).
     */
    void solve(MultiRHS b, bool transpose) const override;
};

#if defined(KOKKOS_ENABLE_SERIAL)
extern template class SplinesLinearProblemPDSTridiag<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
extern template class SplinesLinearProblemPDSTridiag<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
extern template class SplinesLinearProblemPDSTridiag<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
extern template class SplinesLinearProblemPDSTridiag<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
extern template class SplinesLinearProblemPDSTridiag<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
