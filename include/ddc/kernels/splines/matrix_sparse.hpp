#pragma once
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ginkgo/core/matrix/dense.hpp"

#include "Kokkos_Core_fwd.hpp"
#include "matrix.hpp"
#include "view.hpp"

namespace ddc::detail {
// Matrix class for Csr storage and iterative solve
template <class ExecSpace>
class Matrix_Sparse : public Matrix
{
public:
    // Constructor
    Matrix_Sparse(const int mat_size)
        : Matrix(mat_size)
        , m(mat_size)
        , n(mat_size)
        , rows("rows", mat_size + 1)
        , cols("cols", mat_size * mat_size)
        , data("data", mat_size * mat_size)
    {
        // Fill the csr indexes as a dense matrix and initialize with zeros (zeros will be removed once non-zeros elements will be set)
        for (int i = 0; i < m * n; i++) {
            if (i < m + 1) {
                rows(i) = i * n; //CSR
            }
            cols(i) = i % n;
            data(i) = 0;
        }
#ifdef KOKKOS_ENABLE_SERIAL
        if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
            cols_per_par_chunk = 1;
            par_chunks_per_seq_chunk = 1;
        }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
            cols_per_par_chunk = 4096;
            // TODO: Investigate OpenMP parallelism in Ginkgo
            par_chunks_per_seq_chunk = ExecSpace::concurrency();
        }
#endif
#ifdef KOKKOS_ENABLE_CUDA
        if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
            cols_per_par_chunk = Kokkos::pow(2, 16) - 1; // TODO: call cudaMaxGridSize ?
            par_chunks_per_seq_chunk = 1;
        }
#endif
    }
    int m;
    int n;
    Kokkos::View<int*, Kokkos::HostSpace> rows;
    Kokkos::View<int*, Kokkos::HostSpace> cols;
    Kokkos::View<double*, Kokkos::HostSpace> data;

    int cols_per_par_chunk; // Maximum number of columns of B to be passed to a Ginkgo solver
    int par_chunks_per_seq_chunk; // Maximum number of Ginkgo calls to be executed in parallel

    virtual std::unique_ptr<gko::matrix::Dense<>, std::default_delete<gko::matrix::Dense<>>>
    to_gko_vec(
            double* vec_ptr,
            size_t n,
            size_t n_equations,
            std::shared_ptr<gko::Executor> gko_exec) const
    {
        auto v = gko::matrix::Dense<>::
                create(gko_exec,
                       gko::dim<2> {n, n_equations},
                       gko::array<double>::view(gko_exec, n * n_equations, vec_ptr),
                       n_equations);
        return v;
    }

    virtual std::unique_ptr<gko::matrix::Csr<>, std::default_delete<gko::matrix::Csr<>>> to_gko_mat(
            double* mat_ptr,
            size_t n_nonzero_rows,
            size_t n_nonzeros,
            std::shared_ptr<gko::Executor> gko_exec) const
    {
        auto M = gko::matrix::Csr<>::
                create(gko_exec,
                       gko::dim<2> {m, n},
                       gko::array<double>::view(gko_exec, n_nonzeros, mat_ptr),
                       gko::array<int>::view(gko_exec, n_nonzeros, cols.data()),
                       gko::array<int>::view(gko_exec, n_nonzero_rows + 1, rows.data()));
        return M;
    }

    virtual double get_element(int i, int j) const override
    {
        // Wrong, TODO: correct
        return data(i * n + j);
    }
    virtual void set_element(int i, int j, double aij) override
    {
        data(i * n + j) = aij;
    }

    virtual int factorize_method() override
    {
        std::shared_ptr<gko::Executor> gko_exec = create_gko_exec<ExecSpace>();
        // Remove zeros
        auto data_mat = gko::share(to_gko_mat(data.data(), m, m * n, gko_exec->get_master()));
        auto data_mat_ = gko::matrix_data<>(gko::dim<2> {m, n});
        data_mat->write(data_mat_);
        data_mat_.remove_zeros();
        data_mat->read(data_mat_);

        // Realloc Kokkos::Views without zeros
        Kokkos::realloc(Kokkos::WithoutInitializing, cols, data_mat_.nonzeros.size());
        Kokkos::realloc(Kokkos::WithoutInitializing, data, data_mat_.nonzeros.size());
        Kokkos::deep_copy(
                rows,
                Kokkos::View<
                        int*,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(data_mat->get_row_ptrs(), m + 1));
        Kokkos::deep_copy(
                cols,
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        data_mat->get_col_idxs(),
                        data_mat->get_num_stored_elements()));
        Kokkos::deep_copy(
                data,
                Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        data_mat->get_values(),
                        data_mat->get_num_stored_elements()));
        return 0;
    }
    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const override
    {
        std::shared_ptr<gko::Executor> gko_exec = create_gko_exec<ExecSpace>();
        auto data_mat = gko::share(
                to_gko_mat(data.data(), rows.size() - 1, cols.size(), gko_exec->get_master()));
        auto data_mat_gpu = gko::share(gko::clone(gko_exec, data_mat));
        Kokkos::View<
                double**,
                Kokkos::LayoutRight,
                ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                b_view(b, n, n_equations);

        const int n_seq_chunks = n_equations / cols_per_par_chunk / par_chunks_per_seq_chunk + 1;
        const int par_chunks_per_last_seq_chunk
                = (n_equations % (cols_per_par_chunk * par_chunks_per_seq_chunk))
                          / cols_per_par_chunk
                  + 1;
        const int cols_per_last_par_chunk
                = (n_equations % (cols_per_par_chunk * par_chunks_per_seq_chunk * n_seq_chunks))
                  % cols_per_par_chunk;

        Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace>
                b_buffer("b_buffer", par_chunks_per_seq_chunk, n, cols_per_par_chunk);
        // Last par_chunk of last seq_chunk do not have same number of columns than the others. To get proper layout (because we passe the pointers to Ginkgo), we need a dedicated allocation
        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>
                b_last_buffer("b_last_buffer", n, cols_per_last_par_chunk);

        // Sequential loop
        for (int i = 0; i < n_seq_chunks; i++) {
            int n_par_chunks_in_seq_chunk = i < n_seq_chunks - 1 ? par_chunks_per_seq_chunk
                                                                 : par_chunks_per_last_seq_chunk;
            // Parallel loop
            Kokkos::parallel_for(
                    Kokkos::RangePolicy<
                            Kokkos::DefaultHostExecutionSpace>(0, n_par_chunks_in_seq_chunk),
                    [&](int const j) {
                        int n_equations_in_par_chunk
                                = (i < n_seq_chunks - 1 || j < n_par_chunks_in_seq_chunk - 1)
                                          ? cols_per_par_chunk
                                          : cols_per_last_par_chunk;
                        // Ignore last parallel chunk if empty
                        if (n_equations_in_par_chunk != 0) {
                            // Select window of cols in the current parallel chunk
                            auto par_chunk_window = std::pair<int, int>(
                                    (i * par_chunks_per_seq_chunk + j) * cols_per_par_chunk,
                                    (i * par_chunks_per_seq_chunk + j) * cols_per_par_chunk
                                            + n_equations_in_par_chunk);
                            Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> b_par_chunk;
                            if (i < n_seq_chunks - 1 || j < n_par_chunks_in_seq_chunk - 1) {
                                b_par_chunk = Kokkos::
                                        subview(b_buffer,
                                                j,
                                                Kokkos::ALL,
                                                std::pair<int, int>(0, n_equations_in_par_chunk));
                            } else {
                                b_par_chunk = Kokkos::
                                        subview(b_last_buffer,
                                                Kokkos::ALL,
                                                std::pair<int, int>(0, n_equations_in_par_chunk));
                            }
                            // Copy data window from b to the buffer
                            Kokkos::deep_copy(
                                    b_par_chunk,
                                    Kokkos::subview(b_view, Kokkos::ALL, par_chunk_window));
                            auto b_vec_batch = to_gko_vec(
                                    b_par_chunk.data(),
                                    n,
                                    n_equations_in_par_chunk,
                                    gko_exec);
                            // Create the solver TODO: pass in constructor ?
                            std::shared_ptr<gko::log::Stream<>> stream_logger = gko::log::Stream<>::
                                    create(gko::log::Logger::all_events_mask
                                                   ^ gko::log::Logger::linop_factory_events_mask
                                                   ^ gko::log::Logger::
                                                           polymorphic_object_events_mask,
                                           std::cout);
                            std::shared_ptr<gko::stop::ResidualNorm<>::Factory> residual_criterion
                                    = gko::stop::ResidualNorm<>::build()
                                              .with_reduction_factor(1e-20)
                                              .on(gko_exec);
                            auto preconditionner = gko::preconditioner::Jacobi<>::build()
                                                           .with_max_block_size(1u)
                                                           .on(gko_exec);
                            auto preconditionner_
                                    = gko::share(preconditionner->generate(data_mat_gpu));
                            auto solver = gko::solver::Bicgstab<>::build()
                                                  .with_generated_preconditioner(preconditionner_)
                                                  .with_criteria(
                                                          residual_criterion,
                                                          gko::stop::Iteration::build()
                                                                  .with_max_iters(1000u)
                                                                  .on(gko_exec))
                                                  .on(gko_exec);
                            auto solver_ = solver->generate(data_mat_gpu);
                            // solver_->add_logger(stream_logger);

                            // Solve
                            solver_->apply(b_vec_batch, b_vec_batch); // inplace solve

                            // Copy the result from the buffer to b
                            Kokkos::deep_copy(
                                    Kokkos::subview(b_view, Kokkos::ALL, par_chunk_window),
                                    b_par_chunk);
                        }
                    });
        }
        return 1;
    }
};
} // namespace ddc::detail
