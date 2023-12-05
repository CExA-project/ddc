#pragma once
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include <ginkgo/ginkgo.hpp>

#include <Kokkos_Core.hpp>

#include "ddc/misc/ginkgo_executors.hpp"
#include "ginkgo/core/matrix/dense.hpp"

#include "matrix.hpp"
#include "view.hpp"

namespace ddc::detail {

template <class T, class ExecSpace>
std::unique_ptr<gko::matrix::Dense<T>> to_gko_vec(
        Kokkos::View<T**, Kokkos::LayoutRight, ExecSpace> const& view)
{
    std::shared_ptr<gko::Executor> exec = create_gko_exec<ExecSpace>();
    return gko::matrix::Dense<T>::
            create(exec,
                   gko::dim<2>(view.extent_int(0), view.extent_int(1)),
                   gko::array<T>::view(exec, view.span(), view.data()),
                   view.stride_0());
}

// Matrix class for Csr storage and iterative solve
template <class ExecSpace>
class Matrix_Sparse : public Matrix
{
private:
    int m_m;

    int m_n;

    Kokkos::View<int*, Kokkos::HostSpace> m_rows;

    Kokkos::View<int*, Kokkos::HostSpace> m_cols;

    Kokkos::View<double*, Kokkos::HostSpace> m_data;

    std::unique_ptr<gko::solver::Bicgstab<>::Factory> m_solver_factory;

    int m_main_chunk_size; // Maximum number of columns of B to be passed to a Ginkgo solver
    int m_preconditionner_max_block_size; // Maximum size of Jacobi-block preconditionner

public:
    // Constructor
    explicit Matrix_Sparse(
            const int mat_size,
            std::optional<int> cols_per_par_chunk = std::nullopt,
            std::optional<int> par_chunks_per_seq_chunk = std::nullopt,
            std::optional<unsigned int> preconditionner_max_block_size = std::nullopt)
        : Matrix(mat_size)
        , m_m(mat_size)
        , m_n(mat_size)
        , m_rows("rows", mat_size + 1)
        , m_cols("cols", mat_size * mat_size)
        , m_data("data", mat_size * mat_size)
    {
        // Fill the csr indexes as a dense matrix and initialize with zeros (zeros will be removed once non-zeros elements will be set)
        for (int i = 0; i < m_m * m_n; i++) {
            if (i < m_m + 1) {
                m_rows(i) = i * m_n; //CSR
            }
            m_cols(i) = i % m_n;
            m_data(i) = 0;
        }

        if (cols_per_par_chunk.has_value()) {
            m_main_chunk_size = cols_per_par_chunk.value();
        } else {
#ifdef KOKKOS_ENABLE_SERIAL
            if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
                m_main_chunk_size = 256;
            }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
            if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
                m_main_chunk_size = 256;
            }
#endif
#ifdef KOKKOS_ENABLE_CUDA
            if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                m_main_chunk_size = 65535;
            }
#endif
#ifdef KOKKOS_ENABLE_HIP
            if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
                m_main_chunk_size = 65535;
            }
#endif
        }

        if (preconditionner_max_block_size.has_value()) {
            m_preconditionner_max_block_size = preconditionner_max_block_size.value();
        } else {
#ifdef KOKKOS_ENABLE_SERIAL
            if (std::is_same_v<ExecSpace, Kokkos::Serial>) {
                m_preconditionner_max_block_size = 32u;
            }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
            if (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
                m_preconditionner_max_block_size = 32u;
            }
#endif
#ifdef KOKKOS_ENABLE_CUDA
            if (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
                m_preconditionner_max_block_size = 1u;
            }
#endif
#ifdef KOKKOS_ENABLE_HIP
            if (std::is_same_v<ExecSpace, Kokkos::HIP>) {
                m_preconditionner_max_block_size = 1u;
            }
#endif
        }

        // Create the solver factory
        std::shared_ptr<gko::Executor> gko_exec = create_gko_exec<ExecSpace>();
        std::shared_ptr<gko::stop::ResidualNorm<>::Factory> residual_criterion
                = gko::stop::ResidualNorm<>::build().with_reduction_factor(1e-20).on(gko_exec);
        m_solver_factory
                = gko::solver::Bicgstab<>::build()
                          .with_preconditioner(
                                  gko::preconditioner::Jacobi<>::build()
                                          .with_max_block_size(m_preconditionner_max_block_size)
                                          .on(gko_exec))
                          .with_criteria(
                                  residual_criterion,
                                  gko::stop::Iteration::build().with_max_iters(1000u).on(gko_exec))
                          .on(gko_exec);
        gko_exec->synchronize();
    }

    std::unique_ptr<gko::matrix::Csr<>> to_gko_mat(
            double* mat_ptr,
            size_t n_nonzero_rows,
            size_t n_nonzeros,
            std::shared_ptr<gko::Executor> gko_exec) const
    {
        auto M = gko::matrix::Csr<>::
                create(gko_exec,
                       gko::dim<2>(m_m, m_n),
                       gko::array<double>::view(gko_exec, n_nonzeros, mat_ptr),
                       gko::array<int>::view(gko_exec, n_nonzeros, m_cols.data()),
                       gko::array<int>::view(gko_exec, n_nonzero_rows + 1, m_rows.data()));
        return M;
    }

    virtual double get_element(int i, int j) const override
    {
        throw std::runtime_error("MatrixSparse::get_element() is not implemented because no API is "
                                 "provided by Ginkgo");
    }

    virtual void set_element(int i, int j, double aij) override
    {
        m_data(i * m_n + j) = aij;
    }

    int factorize_method() override
    {
        std::shared_ptr<gko::Executor> gko_exec = create_gko_exec<ExecSpace>();
        // Remove zeros
        auto data_mat
                = gko::share(to_gko_mat(m_data.data(), m_m, m_m * m_n, gko_exec->get_master()));
        auto data_mat_ = gko::matrix_data<>(gko::dim<2>(m_m, m_n));
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
                                Kokkos::Unmanaged>>(data_mat->get_row_ptrs(), m_m + 1));
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
        std::shared_ptr<gko::Executor> const gko_exec = create_gko_exec<ExecSpace>();
        auto const data_mat = gko::share(to_gko_mat(
                m_data.data(),
                m_rows.size() - 1,
                m_cols.size(),
                gko_exec->get_master()));
        auto const data_mat_device = gko::share(gko::clone(gko_exec, data_mat));
        auto const solver = m_solver_factory->generate(data_mat_device);

        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> b_view(b, m_n, n_equations);
        Kokkos::View<double**, Kokkos::LayoutRight, ExecSpace> x_view("", m_n, m_main_chunk_size);

        int const iend = (n_equations + m_main_chunk_size - 1) / m_main_chunk_size;
        for (int i = 0; i < iend; ++i) {
            int const subview_begin = i * m_main_chunk_size;
            int const subview_end
                    = (i + 1 == iend) ? n_equations : (subview_begin + m_main_chunk_size);

            auto const b_subview = Kokkos::
                    subview(b_view, Kokkos::ALL, Kokkos::pair(subview_begin, subview_end));
            auto const x_subview = Kokkos::
                    subview(x_view, Kokkos::ALL, Kokkos::pair(0, subview_end - subview_begin));

            Kokkos::deep_copy(x_subview, b_subview);
            Kokkos::fence();

            solver
                    ->apply(to_gko_vec<double, ExecSpace>(b_subview),
                            to_gko_vec<double, ExecSpace>(x_subview)); // inplace solve
            gko_exec->synchronize();

            Kokkos::deep_copy(b_subview, x_subview);
        }

        return 1;
    }
};

} // namespace ddc::detail
