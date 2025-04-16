// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <ginkgo/extensions/kokkos.hpp>
#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>

#include "splines_linear_problem.hpp"

namespace ddc::detail {

/**
 * @brief Convert KokkosView to Ginkgo Dense matrix.
 *
 * @param[in] gko_exec A Ginkgo executor that has access to the Kokkos::View memory space
 * @param[in] view A 2-D Kokkos::View with unit stride in the second dimension
 * @return A Ginkgo Dense matrix view over the Kokkos::View data
 */
template <class KokkosViewType>
auto to_gko_dense(std::shared_ptr<const gko::Executor> const& gko_exec, KokkosViewType const& view)
{
    static_assert((Kokkos::is_view_v<KokkosViewType> && KokkosViewType::rank == 2));
    using value_type = typename KokkosViewType::traits::value_type;

    if (view.stride(1) != 1) {
        throw std::runtime_error("The view needs to be contiguous in the second dimension");
    }

    return gko::matrix::Dense<value_type>::
            create(gko_exec,
                   gko::dim<2>(view.extent(0), view.extent(1)),
                   gko::array<value_type>::view(gko_exec, view.span(), view.data()),
                   view.stride(0));
}

/**
 * @brief Return the default value of the parameter cols_per_chunk for a given Kokkos::ExecutionSpace.
 *
 * The values are hardware-specific (but they can be overridden in the constructor of SplinesLinearProblemSparse).
 * They have been tuned on the basis of ddc/benchmarks/splines.cpp results on 4xIntel 6230 + Nvidia V100.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace type.
 * @return The default value for the parameter cols_per_chunk.
 */
template <class ExecSpace>
std::size_t default_cols_per_chunk() noexcept
{
#if defined(KOKKOS_ENABLE_SERIAL)
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return 8192;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return 8192;
    }
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return 65535;
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        return 65535;
    }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    if (std::is_same_v<ExecSpace, Kokkos::SYCL>) {
        return 65535;
    }
#endif
    return 1;
}

/**
 * @brief Return the default value of the parameter preconditioner_max_block_size for a given Kokkos::ExecutionSpace.
 *
 * The values are hardware-specific (but they can be overridden in the constructor of SplinesLinearProblemSparse).
 * They have been tuned on the basis of ddc/benchmarks/splines.cpp results on 4xIntel 6230 + Nvidia V100.
 *
 * @tparam ExecSpace The Kokkos::ExecutionSpace type.
 * @return The default value for the parameter preconditioner_max_block_size.
 */
template <class ExecSpace>
unsigned int default_preconditioner_max_block_size() noexcept
{
#if defined(KOKKOS_ENABLE_SERIAL)
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return 32U;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return 1U;
    }
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return 1U;
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        return 1U;
    }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    if (std::is_same_v<ExecSpace, Kokkos::SYCL>) {
        return 1U;
    }
#endif
    return 1U;
}

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
    using matrix_sparse_type = gko::matrix::Csr<double, gko::int32>;
#if defined(KOKKOS_ENABLE_OPENMP)
    using solver_type = std::conditional_t<
            std::is_same_v<ExecSpace, Kokkos::OpenMP>,
            gko::solver::Gmres<double>,
            gko::solver::Bicgstab<double>>;
#else
    using solver_type = gko::solver::Bicgstab<double>;
#endif


private:
    std::unique_ptr<gko::matrix::Dense<double>> m_matrix_dense;

    std::shared_ptr<matrix_sparse_type> m_matrix_sparse;

    std::shared_ptr<solver_type> m_solver;
    std::shared_ptr<gko::LinOp> m_solver_tr;

    std::size_t m_cols_per_chunk; // Maximum number of columns of B to be passed to a Ginkgo solver

    unsigned int m_preconditioner_max_block_size; // Maximum size of Jacobi-block preconditioner

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
            const std::size_t mat_size,
            std::optional<std::size_t> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditioner_max_block_size = std::nullopt)
        : SplinesLinearProblem<ExecSpace>(mat_size)
        , m_cols_per_chunk(cols_per_chunk.value_or(default_cols_per_chunk<ExecSpace>()))
        , m_preconditioner_max_block_size(preconditioner_max_block_size.value_or(
                  default_preconditioner_max_block_size<ExecSpace>()))
    {
        std::shared_ptr const gko_exec = gko::ext::kokkos::create_executor(ExecSpace());
        m_matrix_dense = gko::matrix::Dense<
                double>::create(gko_exec->get_master(), gko::dim<2>(mat_size, mat_size));
        m_matrix_dense->fill(0);
        m_matrix_sparse = matrix_sparse_type::create(gko_exec, gko::dim<2>(mat_size, mat_size));
    }

    double get_element(std::size_t i, std::size_t j) const override
    {
        return m_matrix_dense->at(i, j);
    }

    void set_element(std::size_t i, std::size_t j, double aij) override
    {
        m_matrix_dense->at(i, j) = aij;
    }

    /**
     * @brief Perform a pre-process operation on the solver. Must be called after filling the matrix.
     *
     * Removes the zeros from the CSR object and instantiate a Ginkgo solver. It also constructs a transposed version of the solver.
     *
     * The stopping criterion is a reduction factor ||Ax-b||/||b||<1e-15 with 1000 maximum iterations.
     */
    void setup_solver() override
    {
        // Remove zeros
        gko::matrix_data<double> matrix_data(gko::dim<2>(size(), size()));
        m_matrix_dense->write(matrix_data);
        m_matrix_dense.reset();
        matrix_data.remove_zeros();
        m_matrix_sparse->read(matrix_data);
        std::shared_ptr const gko_exec = m_matrix_sparse->get_executor();

        // Create the solver factory
        std::shared_ptr const residual_criterion
                = gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-15).on(
                        gko_exec);

        std::shared_ptr const iterations_criterion
                = gko::stop::Iteration::build().with_max_iters(1000U).on(gko_exec);

        std::shared_ptr const preconditioner
                = gko::preconditioner::Jacobi<double>::build()
                          .with_max_block_size(m_preconditioner_max_block_size)
                          .on(gko_exec);

        std::unique_ptr const solver_factory
                = solver_type::build()
                          .with_preconditioner(preconditioner)
                          .with_criteria(residual_criterion, iterations_criterion)
                          .on(gko_exec);

        m_solver = solver_factory->generate(m_matrix_sparse);
        m_solver_tr = m_solver->transpose();
        gko_exec->synchronize();
    }

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
    void solve(MultiRHS const b, bool const transpose) const override
    {
        assert(b.extent(0) == size());

        std::shared_ptr const gko_exec = m_solver->get_executor();
        std::shared_ptr const convergence_logger = gko::log::Convergence<double>::create();

        std::size_t const main_chunk_size = std::min(m_cols_per_chunk, b.extent(1));

        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> const
                b_buffer("ddc_sparse_b_buffer", size(), main_chunk_size);
        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> const
                x("ddc_sparse_x", size(), main_chunk_size);

        std::size_t const iend = (b.extent(1) + main_chunk_size - 1) / main_chunk_size;
        for (std::size_t i = 0; i < iend; ++i) {
            std::size_t const subview_begin = i * main_chunk_size;
            std::size_t const subview_end
                    = (i + 1 == iend) ? b.extent(1) : (subview_begin + main_chunk_size);

            auto const b_chunk
                    = Kokkos::subview(b, Kokkos::ALL, Kokkos::pair(subview_begin, subview_end));
            auto const b_buffer_chunk = Kokkos::
                    subview(b_buffer,
                            Kokkos::ALL,
                            Kokkos::pair(std::size_t(0), subview_end - subview_begin));
            auto const x_chunk = Kokkos::
                    subview(x,
                            Kokkos::ALL,
                            Kokkos::pair(std::size_t(0), subview_end - subview_begin));

            Kokkos::deep_copy(b_buffer_chunk, b_chunk);
            Kokkos::deep_copy(x_chunk, b_chunk);

            if (!transpose) {
                m_solver->add_logger(convergence_logger);
                m_solver
                        ->apply(to_gko_dense(gko_exec, b_buffer_chunk),
                                to_gko_dense(gko_exec, x_chunk));
                m_solver->remove_logger(convergence_logger);
            } else {
                m_solver_tr->add_logger(convergence_logger);
                m_solver_tr
                        ->apply(to_gko_dense(gko_exec, b_buffer_chunk),
                                to_gko_dense(gko_exec, x_chunk));
                m_solver_tr->remove_logger(convergence_logger);
            }

            if (!convergence_logger->has_converged()) {
                throw std::runtime_error(
                        "Ginkgo did not converged in ddc::detail::SplinesLinearProblemSparse");
            }

            Kokkos::deep_copy(b_chunk, x_chunk);
        }
    }
};

} // namespace ddc::detail
