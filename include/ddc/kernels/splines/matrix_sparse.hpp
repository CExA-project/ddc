#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>

#include "ddc/misc/ginkgo_executors.hpp"

#include "matrix.hpp"

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

/**
 * @param gko_exec[in] A Ginkgo executor that has access to the Kokkos::View memory space
 * @param view[in] A 2-D Kokkos::View with unit stride in the second dimension
 * @return A Ginkgo Dense matrix view over the Kokkos::View data
 */
template <class KokkosViewType>
auto to_gko_dense(std::shared_ptr<const gko::Executor> const& gko_exec, KokkosViewType const& view)
{
    static_assert((Kokkos::is_view_v<KokkosViewType> && KokkosViewType::rank == 2));
    using value_type = typename KokkosViewType::traits::value_type;

    if (view.stride_1() != 1) {
        throw std::runtime_error("The view needs to be contiguous in the second dimension");
    }

    return gko::matrix::Dense<value_type>::
            create(gko_exec,
                   gko::dim<2>(view.extent(0), view.extent(1)),
                   gko::array<value_type>::view(gko_exec, view.span(), view.data()),
                   view.stride_0());
}

template <class ExecSpace>
int default_cols_per_chunk() noexcept
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return 8192;
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return 8192;
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return 65535;
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        return 65535;
    }
#endif
    return 1;
}

template <class ExecSpace>
unsigned int default_preconditionner_max_block_size() noexcept
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return 32u;
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return 32u;
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return 1u;
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        return 1u;
    }
#endif
    return 1u;
}

// Matrix class for sparse storage and iterative solve
template <class ExecSpace>
class Matrix_Sparse : public Matrix
{
    using matrix_sparse_type = gko::matrix::Csr<double, int>;

private:
    std::unique_ptr<gko::matrix::Dense<double>> m_matrix_dense;

    std::shared_ptr<matrix_sparse_type> m_matrix_sparse;

    std::shared_ptr<gko::solver::Bicgstab<double>> m_solver;

    int m_cols_per_chunk; // Maximum number of columns of B to be passed to a Ginkgo solver

    unsigned int m_preconditionner_max_block_size; // Maximum size of Jacobi-block preconditionner

public:
    // Constructor
    explicit Matrix_Sparse(
            const int mat_size,
            std::optional<int> cols_per_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt)
        : Matrix(mat_size)
        , m_cols_per_chunk(cols_per_chunk.value_or(default_cols_per_chunk<ExecSpace>()))
        , m_preconditionner_max_block_size(preconditionner_max_block_size.value_or(
                  default_preconditionner_max_block_size<ExecSpace>()))
    {
        std::shared_ptr const gko_exec = create_gko_exec<ExecSpace>();
        m_matrix_dense = gko::matrix::Dense<
                double>::create(gko_exec->get_master(), gko::dim<2>(mat_size, mat_size));
        m_matrix_dense->fill(0);
        m_matrix_sparse = matrix_sparse_type::create(gko_exec, gko::dim<2>(mat_size, mat_size));
    }

    virtual double get_element([[maybe_unused]] int i, [[maybe_unused]] int j) const override
    {
        throw std::runtime_error("MatrixSparse::get_element() is not implemented because no API is "
                                 "provided by Ginkgo");
    }

    virtual void set_element(int i, int j, double aij) override
    {
        m_matrix_dense->at(i, j) = aij;
    }

    int factorize_method() override
    {
        // Remove zeros
        gko::matrix_data<double> matrix_data(gko::dim<2>(get_size(), get_size()));
        m_matrix_dense->write(matrix_data);
        m_matrix_dense.reset();
        matrix_data.remove_zeros();
        m_matrix_sparse->read(matrix_data);

        std::shared_ptr const gko_exec = m_matrix_sparse->get_executor();
        // Create the solver factory
        std::shared_ptr const residual_criterion
                = gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-19).on(
                        gko_exec);

        std::shared_ptr const iterations_criterion
                = gko::stop::Iteration::build().with_max_iters(1000u).on(gko_exec);

        std::shared_ptr const preconditioner
                = gko::preconditioner::Jacobi<double>::build()
                          .with_max_block_size(m_preconditionner_max_block_size)
                          .on(gko_exec);

        std::unique_ptr const solver_factory
                = gko::solver::Bicgstab<double>::build()
                          .with_preconditioner(preconditioner)
                          .with_criteria(residual_criterion, iterations_criterion)
                          .on(gko_exec);

        m_solver = solver_factory->generate(m_matrix_sparse);
        gko_exec->synchronize();

        return 0;
    }

    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const override
    {
        if (transpose != 'N') {
            throw std::domain_error("transpose");
        }

        std::shared_ptr const gko_exec = m_solver->get_executor();

        int const main_chunk_size = std::min(m_cols_per_chunk, n_equations);

        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> const
                b_view(b, get_size(), n_equations);
        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> const
                x_view("", get_size(), main_chunk_size);

        int const iend = (n_equations + main_chunk_size - 1) / main_chunk_size;
        for (int i = 0; i < iend; ++i) {
            int const subview_begin = i * main_chunk_size;
            int const subview_end
                    = (i + 1 == iend) ? n_equations : (subview_begin + main_chunk_size);

            auto const b_subview = Kokkos::
                    subview(b_view, Kokkos::ALL, Kokkos::pair(subview_begin, subview_end));
            auto const x_subview = Kokkos::
                    subview(x_view, Kokkos::ALL, Kokkos::pair(0, subview_end - subview_begin));

            Kokkos::deep_copy(x_subview, b_subview);

			auto res_logger = std::make_shared<ResidualLogger<double>>(m_matrix_sparse.get(), to_gko_dense(gko_exec, b_subview).get());
            // m_solver->add_logger(res_logger);
            m_solver->apply(to_gko_dense(gko_exec, b_subview), to_gko_dense(gko_exec, x_subview));
			// res_logger->write_data(std::cout);
            Kokkos::deep_copy(b_subview, x_subview);
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
        write(std::cout, data_mat_device);
        std::cout << "-----------------------";
        write(std::cout, b_vec_batch);

#endif
        }

        return 1;
    }
};

} // namespace ddc::detail
