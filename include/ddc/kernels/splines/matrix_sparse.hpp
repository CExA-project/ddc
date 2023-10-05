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
// TODO : support multiple-rhs case
// Residual logger (error logged at each iteration)
template <typename ValueType>
ValueType get_first_element(const gko::matrix::Dense<ValueType>* norm)
{
    return norm->get_executor()->copy_val_to_host(norm->get_const_values());
}

template <typename ValueType>
ValueType compute_norm(const gko::matrix::Dense<ValueType>* b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
    b->compute_norm2(b_norm);
    return get_first_element(b_norm.get());
}


template <typename ValueType>
ValueType compute_residual_norm(
        const gko::LinOp* system_matrix,
        const gko::matrix::Dense<ValueType>* b,
        const gko::matrix::Dense<ValueType>* x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<gko::matrix::Dense<ValueType>>({-1.0}, exec);
    auto res = gko::clone(b);
    system_matrix->apply(one, x, neg_one, res);
    return compute_norm(res.get());
}

template <typename ValueType>
struct ResidualLogger : gko::log::Logger
{
    void on_iteration_complete(
            const gko::LinOp*,
            const gko::size_type&,
            const gko::LinOp* residual,
            const gko::LinOp* solution,
            const gko::LinOp* residual_norm) const override
    {
        if (residual_norm) {
            rec_res_norms.push_back(
                    get_first_element(gko::as<gko::matrix::Dense<ValueType>>(residual_norm)));
        } else {
            rec_res_norms.push_back(compute_norm(gko::as<gko::matrix::Dense<ValueType>>(residual)));
        }
        if (solution) {
            true_res_norms.push_back(compute_residual_norm(
                    matrix,
                    b,
                    gko::as<gko::matrix::Dense<ValueType>>(solution)));
        } else {
            true_res_norms.push_back(-1.0);
        }
    }

    ResidualLogger(const gko::LinOp* matrix, const gko::matrix::Dense<ValueType>* b)
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
        , matrix {matrix}
        , b {b}
    {
    }

    void write_data(std::ostream& ostream)
    {
        ostream << "Recurrent Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto& entry : rec_res_norms) {
            ostream << " " << entry;
        }
        ostream << "];" << std::endl;

        ostream << "True Residual Norms: " << std::endl;
        ostream << "[" << std::endl;
        for (const auto& entry : true_res_norms) {
            ostream << " " << entry;
        }
        ostream << "];" << std::endl;
    }

private:
    const gko::LinOp* matrix;
    const gko::matrix::Dense<ValueType>* b;
    mutable std::vector<ValueType> rec_res_norms;
    mutable std::vector<ValueType> true_res_norms;
};

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

		cols_per_par_chunk = 65535;
        // cols_per_par_chunk = std::is_same_v<ExecSpace, Kokkos::Cuda>
        //                              ? Kokkos::pow(2, 16) - 1
        //                             : INT_MAX; // TODO: call cudaMaxGridSize ?
        par_chunks_per_seq_chunk = 1;
    }
    int m;
    int n;
    Kokkos::View<int*, Kokkos::HostSpace> rows;
    Kokkos::View<int*, Kokkos::HostSpace> cols;
    Kokkos::View<double*, Kokkos::HostSpace> data;

    int cols_per_par_chunk; // Maximum number of columns of B to be passed to a Ginkgo solver
    int par_chunks_per_seq_chunk; // Maximum number of teams to be executed in parallel

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

        // TODO: use a last incomplete per_par_chunk
        const int n_seq_chunks = n_equations / cols_per_par_chunk / par_chunks_per_seq_chunk + 1;
        for (int i = 0; i < n_seq_chunks; i++) {
            int n_par_chunks_in_seq_chunk
                    = i < n_seq_chunks - 1
                              ? par_chunks_per_seq_chunk
                              : (n_equations % (cols_per_par_chunk * par_chunks_per_seq_chunk))
                                                / cols_per_par_chunk
                                        + 1;
            Kokkos::parallel_for(
                    Kokkos::RangePolicy<
                            Kokkos::DefaultHostExecutionSpace>(0, n_par_chunks_in_seq_chunk),
                    [&](int const j) {
                        int n_equations_in_par_chunk
                                = (i < n_seq_chunks - 1 || j < n_par_chunks_in_seq_chunk - 1)
                                          ? cols_per_par_chunk
                                          : (n_equations
                                             % (cols_per_par_chunk * par_chunks_per_seq_chunk
                                                * n_seq_chunks))
                                                    % cols_per_par_chunk;
                        if (n_equations_in_par_chunk != 0) {
                            Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>
                                    b_par_chunk("b_gpu", n, n_equations_in_par_chunk);
                            Kokkos::deep_copy(
                                    b_par_chunk,
                                    Kokkos::
                                            subview(b_view,
                                                    Kokkos::ALL,
                                                    std::pair<int, int>(
                                                            (i * par_chunks_per_seq_chunk + j)
                                                                    * cols_per_par_chunk,
                                                            (i * par_chunks_per_seq_chunk + j)
                                                                            * cols_per_par_chunk
                                                                    + n_equations_in_par_chunk)));
                            auto b_vec_batch = to_gko_vec(
                                    b_par_chunk.data(),
                                    n,
                                    n_equations_in_par_chunk,
                                    gko_exec);
                            Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>
                                    x_par_chunk("x_gpu", n, n_equations_in_par_chunk);
                            auto x_vec_batch = to_gko_vec(
                                    x_par_chunk.data(),
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
                            // gko_exec->add_logger(stream_logger);
                            std::shared_ptr<gko::stop::ResidualNorm<>::Factory> residual_criterion
                                    = gko::stop::ResidualNorm<>::build()
                                              .with_reduction_factor(1e-20)
                                              .on(gko_exec);
                            std::shared_ptr<const gko::log::Convergence<>> convergence_logger
                                    = gko::log::Convergence<>::create(gko_exec);
                            residual_criterion->add_logger(convergence_logger);
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
                            solver_->add_logger(stream_logger);
                            // auto res_logger = std::make_shared<ResidualLogger<double>>(data_mat_gpu.get(), b_vec_batch.get());
                            // solver_->add_logger(res_logger);
                            solver_->apply(b_vec_batch, x_vec_batch);
// res_logger->write_data(std::cout);

// Debug purpose
#if 0
      	for (int i=0; i<130; i++) {
        	// auto b_data = b_vec_batch->get_values();
        	auto b_data = b_par_chunk.data();
      		Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0,1),KOKKOS_LAMBDA (int j) { printf("%f ", b_data[i]); });
		}
#endif
#if 0
        // Write result
        std::cout << "-----------------------";
        write(std::cout, data_mat_gpu);
        std::cout << "-----------------------";
        write(std::cout, b_vec_batch);
        std::cout << "-----------------------";
        write(std::cout, x_vec_batch);


#endif
                            Kokkos::deep_copy(
                                    Kokkos::
                                            subview(b_view,
                                                    Kokkos::ALL,
                                                    std::pair<int, int>(
                                                            (i * par_chunks_per_seq_chunk + j)
                                                                    * cols_per_par_chunk,
                                                            (i * par_chunks_per_seq_chunk + j)
                                                                            * cols_per_par_chunk
                                                                    + n_equations_in_par_chunk)),
                                    x_par_chunk);
                        }
                    });
        }
        return 1;
    }
};
} // namespace ddc::detail
