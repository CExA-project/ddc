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
int default_cols_per_par_chunk() noexcept
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return 256;
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return 256;
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
int default_par_chunks_per_seq_chunk() noexcept
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return 1;
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return ExecSpace().concurrency();
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return 1;
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        return 1;
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

// Matrix class for Csr storage and iterative solve
template <class ExecSpace>
class Matrix_Sparse : public Matrix
{
private:
    Kokkos::View<int*, Kokkos::HostSpace> m_rows;

    Kokkos::View<int*, Kokkos::HostSpace> m_cols;

    Kokkos::View<double*, Kokkos::HostSpace> m_data;

    std::unique_ptr<gko::solver::Bicgstab<double>::Factory> m_solver_factory;

    int m_cols_per_par_chunk; // Maximum number of columns of B to be passed to a Ginkgo solver

    int m_par_chunks_per_seq_chunk; // Maximum number of teams to be executed in parallel

    unsigned int m_preconditionner_max_block_size; // Maximum size of Jacobi-block preconditionner

public:
    // Constructor
    explicit Matrix_Sparse(
            const int mat_size,
            std::optional<int> cols_per_par_chunk = std::nullopt,
            std::optional<int> par_chunks_per_seq_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt)
        : Matrix(mat_size)
        , m_rows("rows", mat_size + 1)
        , m_cols("cols", mat_size * mat_size)
        , m_data("data", mat_size * mat_size)
        , m_cols_per_par_chunk(cols_per_par_chunk.value_or(default_cols_per_par_chunk<ExecSpace>()))
        , m_par_chunks_per_seq_chunk(
                  par_chunks_per_seq_chunk.value_or(default_par_chunks_per_seq_chunk<ExecSpace>()))
        , m_preconditionner_max_block_size(preconditionner_max_block_size.value_or(
                  default_preconditionner_max_block_size<ExecSpace>()))
    {
        // Fill the csr indexes as a dense matrix and initialize with zeros (zeros will be removed once non-zeros elements will be set)
        for (int i = 0; i < get_size() * get_size(); i++) {
            if (i < get_size() + 1) {
                m_rows(i) = i * get_size(); //CSR
            }
            m_cols(i) = i % get_size();
            m_data(i) = 0;
        }

        // Create the solver factory
        std::shared_ptr<gko::Executor> gko_exec;
        if (false) {
        }
#ifdef KOKKOS_ENABLE_OPENMP
        else if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
            gko_exec = create_gko_exec<Kokkos::Serial>();
        }
#endif
        else {
            gko_exec = create_gko_exec<ExecSpace>();
        }
        std::shared_ptr<gko::stop::ResidualNorm<double>::Factory> residual_criterion
                = gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-17).on(
                        gko_exec);
        m_solver_factory
                = gko::solver::Bicgstab<double>::build()
                          .with_preconditioner(
                                  gko::preconditioner::Jacobi<double>::build()
                                          .with_max_block_size(m_preconditionner_max_block_size)
                                          .on(gko_exec))
                          .with_criteria(
                                  residual_criterion,
                                  gko::stop::Iteration::build().with_max_iters(1000u).on(gko_exec))
                          .on(gko_exec);
    }

    std::unique_ptr<gko::matrix::Csr<double, int>> to_gko_mat(
            double* mat_ptr,
            size_t n_nonzero_rows,
            size_t n_nonzeros,
            std::shared_ptr<gko::Executor> gko_exec) const
    {
        auto M = gko::matrix::Csr<double, int>::
                create(gko_exec,
                       gko::dim<2>(get_size(), get_size()),
                       gko::array<double>::view(gko_exec, n_nonzeros, mat_ptr),
                       gko::array<int>::view(gko_exec, n_nonzeros, m_cols.data()),
                       gko::array<int>::view(gko_exec, n_nonzero_rows + 1, m_rows.data()));
        return M;
    }

    virtual double get_element([[maybe_unused]] int i, [[maybe_unused]] int j) const override
    {
        throw std::runtime_error("MatrixSparse::get_element() is not implemented because no API is "
                                 "provided by Ginkgo");
    }

    virtual void set_element(int i, int j, double aij) override
    {
        m_data(i * get_size() + j) = aij;
    }

    int factorize_method() override
    {
        std::shared_ptr<gko::Executor> gko_exec = create_gko_exec<ExecSpace>();
        // Remove zeros
        auto data_mat = gko::share(to_gko_mat(
                m_data.data(),
                get_size(),
                get_size() * get_size(),
                gko_exec->get_master()));
        auto data_mat_ = gko::matrix_data<double>(gko::dim<2>(get_size(), get_size()));
        data_mat->write(data_mat_);
        data_mat_.remove_zeros();
        data_mat->read(data_mat_);

        // Realloc Kokkos::Views without zeros
        Kokkos::realloc(Kokkos::WithoutInitializing, m_cols, data_mat_.nonzeros.size());
        Kokkos::realloc(Kokkos::WithoutInitializing, m_data, data_mat_.nonzeros.size());
        Kokkos::deep_copy(
                m_rows,
                Kokkos::View<
                        int*,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<
                                Kokkos::Unmanaged>>(data_mat->get_row_ptrs(), get_size() + 1));
        Kokkos::deep_copy(
                m_cols,
                Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        data_mat->get_col_idxs(),
                        data_mat->get_num_stored_elements()));
        Kokkos::deep_copy(
                m_data,
                Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        data_mat->get_values(),
                        data_mat->get_num_stored_elements()));

        return 0;
    }

    virtual int solve_inplace_method(double* b, char transpose, int n_equations) const override
    {
        if (transpose != 'N') {
            throw std::domain_error("transpose");
        }

        std::shared_ptr<gko::Executor> gko_exec = create_gko_exec<ExecSpace>();
        auto data_mat = gko::share(to_gko_mat(
                m_data.data(),
                m_rows.size() - 1,
                m_cols.size(),
                gko_exec->get_master()));
        auto data_mat_device = gko::share(gko::clone(gko_exec, data_mat));
        Kokkos::View<
                double**,
                Kokkos::LayoutRight,
                ExecSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                b_view(b, get_size(), n_equations);

        const int n_seq_chunks
                = n_equations / m_cols_per_par_chunk / m_par_chunks_per_seq_chunk + 1;
        const int par_chunks_per_last_seq_chunk
                = (n_equations % (m_cols_per_par_chunk * m_par_chunks_per_seq_chunk))
                          / m_cols_per_par_chunk
                  + 1;
        const int cols_per_last_par_chunk
                = (n_equations % (m_cols_per_par_chunk * m_par_chunks_per_seq_chunk * n_seq_chunks))
                  % m_cols_per_par_chunk;

        Kokkos::View<double***, Kokkos::LayoutRight, ExecSpace> b_buffer(
                "b_buffer",
                std::min(n_equations / m_cols_per_par_chunk, m_par_chunks_per_seq_chunk),
                get_size(),
                std::min(n_equations, m_cols_per_par_chunk));
        // Last par_chunk of last seq_chunk do not have same number of columns than the others. To get proper layout (because we passe the pointers to Ginkgo), we need a dedicated allocation
        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace>
                b_last_buffer("b_last_buffer", get_size(), cols_per_last_par_chunk);

        for (int i = 0; i < n_seq_chunks; i++) {
            int n_par_chunks_in_seq_chunk = i < n_seq_chunks - 1 ? m_par_chunks_per_seq_chunk
                                                                 : par_chunks_per_last_seq_chunk;
            Kokkos::parallel_for(
                    Kokkos::RangePolicy<
                            Kokkos::DefaultHostExecutionSpace>(0, n_par_chunks_in_seq_chunk),
                    [&](int const j) {
                        int n_equations_in_par_chunk
                                = (i < n_seq_chunks - 1 || j < n_par_chunks_in_seq_chunk - 1)
                                          ? m_cols_per_par_chunk
                                          : cols_per_last_par_chunk;
                        if (n_equations_in_par_chunk != 0) {
                            auto solver = m_solver_factory->generate(data_mat_device);
                            std::pair<int, int> par_chunk_window(
                                    (i * m_par_chunks_per_seq_chunk + j) * m_cols_per_par_chunk,
                                    (i * m_par_chunks_per_seq_chunk + j) * m_cols_per_par_chunk
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
                            Kokkos::deep_copy(
                                    b_par_chunk,
                                    Kokkos::subview(b_view, Kokkos::ALL, par_chunk_window));
                            auto b_vec_batch = to_gko_dense(gko_exec, b_par_chunk);

                            solver->apply(b_vec_batch, b_vec_batch); // inplace solve
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
