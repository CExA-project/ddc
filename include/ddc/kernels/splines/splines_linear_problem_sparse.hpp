// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include <Kokkos_Core.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief A sparse linear problem dedicated to the computation of a spline approximation.
 *
 * The storage format is CSR. Ginkgo is used to perform every matrix and linear solver-related operations.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace on which operations related to the matrix are performed.
 */
template <class ExecSpace>
class SplinesLinearProblemSparse : public SplinesLinearProblem<ExecSpace>
{
public:
    using typename SplinesLinearProblem<ExecSpace>::MultiRHS;
    using SplinesLinearProblem<ExecSpace>::size;

private:
    class Impl;

    std::unique_ptr<Impl> m_impl;

public:
    /**
     * @brief SplinesLinearProblemSparse constructor.
     *
     * @param mat_size The size of one of the dimensions of the square matrix.
     * @param cols_per_chunk An optional parameter used to define the number of right-hand sides to pass to
     * Ginkgo solver calls. see default_cols_per_chunk.
     * @param preconditioner_max_block_size An optional parameter used to define the maximum size of a block
     * used by the block-Jacobi preconditioner. see default_preconditioner_max_block_size.
     */
    explicit SplinesLinearProblemSparse(
            std::size_t mat_size,
            std::optional<std::size_t> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditioner_max_block_size = std::nullopt);

    SplinesLinearProblemSparse(SplinesLinearProblemSparse const& rhs) = delete;

    SplinesLinearProblemSparse(SplinesLinearProblemSparse&& rhs) = delete;

    ~SplinesLinearProblemSparse() override;

    SplinesLinearProblemSparse& operator=(SplinesLinearProblemSparse const& rhs) = delete;

    SplinesLinearProblemSparse& operator=(SplinesLinearProblemSparse&& rhs) = delete;

    double get_element(std::size_t i, std::size_t j) const override;

    void set_element(std::size_t i, std::size_t j, double aij) override;

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * Removes the zeros from the CSR object and instantiate a Ginkgo solver. It also constructs a transposed version of the solver.
     *
     * The stopping criterion is a reduction factor ||Ax-b||/||b||<1e-15 with 1000 maximum iterations.
     */
    void setup_solver() override;

    /**
     * @brief Solve the multiple right-hand sides linear problem Ax=b or its transposed version A^tx=b inplace.
     *
     * The solver method is currently Bicgstab on CPU Serial and GPU and Gmres on OMP (because of Ginkgo issue #1563).
     *
     * Multiple right-hand sides are sliced in chunks of size cols_per_chunk which are passed one-after-the-other to Ginkgo.
     *
     * @param[in, out] b A 2D Kokkos::View storing the multiple right-hand sides of the problem and receiving the corresponding solution.
     * @param transpose Choose between the direct or transposed version of the linear problem.
     */
    void solve(MultiRHS b, bool transpose) const override;
};

#if defined(KOKKOS_ENABLE_SERIAL)
extern template class SplinesLinearProblemSparse<Kokkos::Serial>;
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
extern template class SplinesLinearProblemSparse<Kokkos::OpenMP>;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
extern template class SplinesLinearProblemSparse<Kokkos::Cuda>;
#endif
#if defined(KOKKOS_ENABLE_HIP)
extern template class SplinesLinearProblemSparse<Kokkos::HIP>;
#endif
#if defined(KOKKOS_ENABLE_SYCL)
extern template class SplinesLinearProblemSparse<Kokkos::SYCL>;
#endif

} // namespace ddc::detail
