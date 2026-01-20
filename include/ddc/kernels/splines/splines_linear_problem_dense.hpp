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
#    include <mkl_lapacke.h>
#else
#    include <lapacke.h>
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
    explicit SplinesLinearProblemDense(std::size_t mat_size);

    SplinesLinearProblemDense(SplinesLinearProblemDense const& rhs) = delete;

    SplinesLinearProblemDense(SplinesLinearProblemDense&& rhs) = delete;

    ~SplinesLinearProblemDense() override;

    SplinesLinearProblemDense& operator=(SplinesLinearProblemDense const& rhs) = delete;

    SplinesLinearProblemDense& operator=(SplinesLinearProblemDense&& rhs) = delete;

    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * LU-factorize the matrix A and store the pivots using the LAPACK dgetrf() implementation.
     */
    void setup_solver() override;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is gaussian elimination with partial pivoting using the LU-factorized matrix A. The implementation is LAPACK method dgetrs.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool transpose) const override;
};

#if defined(KOKKOS_ENABLE_SERIAL)
extern template class SplinesLinearProblemDense<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
extern template class SplinesLinearProblemDense<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
extern template class SplinesLinearProblemDense<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
extern template class SplinesLinearProblemDense<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
extern template class SplinesLinearProblemDense<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
