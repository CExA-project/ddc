// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

#include "splines_linear_problem.hpp"
#include "splines_linear_problem_2x2_blocks.hpp"
#include "splines_linear_problem_dense.hpp"

namespace ddc::detail {

/**
     * @brief COO storage.
     *
     * [SHOULD BE PRIVATE (GPU programming limitation)]
     */
struct SplinesLinearProblem2x2Blocks::Coo
{
    std::size_t m_nrows;
    std::size_t m_ncols;
    Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> m_rows_idx;
    Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> m_cols_idx;
    Kokkos::View<double*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> m_values;

    Coo() : m_nrows(0), m_ncols(0) {}

    Coo(std::size_t const nrows_,
        std::size_t const ncols_,
        Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> rows_idx_,
        Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> cols_idx_,
        Kokkos::View<double*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space> values_)
        : m_nrows(nrows_)
        , m_ncols(ncols_)
        , m_rows_idx(std::move(rows_idx_))
        , m_cols_idx(std::move(cols_idx_))
        , m_values(std::move(values_))
    {
        assert(m_rows_idx.extent(0) == m_cols_idx.extent(0));
        assert(m_rows_idx.extent(0) == m_values.extent(0));
    }

    KOKKOS_FUNCTION std::size_t nnz() const
    {
        return m_values.extent(0);
    }

    KOKKOS_FUNCTION std::size_t nrows() const
    {
        return m_nrows;
    }

    KOKKOS_FUNCTION std::size_t ncols() const
    {
        return m_ncols;
    }

    KOKKOS_FUNCTION Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
    rows_idx() const
    {
        return m_rows_idx;
    }

    KOKKOS_FUNCTION Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
    cols_idx() const
    {
        return m_cols_idx;
    }

    KOKKOS_FUNCTION Kokkos::
            View<double*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            values() const
    {
        return m_values;
    }
};

SplinesLinearProblem2x2Blocks::SplinesLinearProblem2x2Blocks(
        std::size_t const mat_size,
        std::unique_ptr<SplinesLinearProblem> top_left_block)
    : SplinesLinearProblem(mat_size)
    , m_top_left_block(std::move(top_left_block))
    , m_top_right_block(
              "top_right_block",
              m_top_left_block->size(),
              mat_size - m_top_left_block->size())
    , m_bottom_left_block(
              "bottom_left_block",
              mat_size - m_top_left_block->size(),
              m_top_left_block->size())
    , m_bottom_right_block(new SplinesLinearProblemDense(mat_size - m_top_left_block->size()))
{
    assert(m_top_left_block->size() <= mat_size);

    Kokkos::deep_copy(m_top_right_block.view_host(), 0.);
    Kokkos::deep_copy(m_bottom_left_block.view_host(), 0.);
}

SplinesLinearProblem2x2Blocks::~SplinesLinearProblem2x2Blocks() = default;

double SplinesLinearProblem2x2Blocks::get_element(std::size_t const i, std::size_t const j) const
{
    assert(i < size());
    assert(j < size());

    std::size_t const nq = m_top_left_block->size();
    if (i < nq && j < nq) {
        return m_top_left_block->get_element(i, j);
    }

    if (i >= nq && j >= nq) {
        return m_bottom_right_block->get_element(i - nq, j - nq);
    }

    if (j >= nq) {
        return m_top_right_block.view_host()(i, j - nq);
    }

    return m_bottom_left_block.view_host()(i - nq, j);
}

void SplinesLinearProblem2x2Blocks::set_element(
        std::size_t const i,
        std::size_t const j,
        double const aij)
{
    assert(i < size());
    assert(j < size());

    std::size_t const nq = m_top_left_block->size();
    if (i < nq && j < nq) {
        m_top_left_block->set_element(i, j, aij);
    } else if (i >= nq && j >= nq) {
        m_bottom_right_block->set_element(i - nq, j - nq, aij);
    } else if (j >= nq) {
        m_top_right_block.view_host()(i, j - nq) = aij;
    } else {
        m_bottom_left_block.view_host()(i - nq, j) = aij;
    }
}

std::unique_ptr<SplinesLinearProblem2x2Blocks::Coo> SplinesLinearProblem2x2Blocks::dense2coo(
        Kokkos::View<double const**, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
                dense_matrix,
        double const tol)
{
    Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            rows_idx("ddc_splines_coo_rows_idx", dense_matrix.extent(0) * dense_matrix.extent(1));
    Kokkos::View<int*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            cols_idx("ddc_splines_coo_cols_idx", dense_matrix.extent(0) * dense_matrix.extent(1));
    Kokkos::View<double*, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            values("ddc_splines_coo_values", dense_matrix.extent(0) * dense_matrix.extent(1));

    Kokkos::DualView<std::size_t, Kokkos::LayoutRight, typename Kokkos::Serial::memory_space>
            n_nonzeros("ddc_splines_n_nonzeros");
    n_nonzeros.view_host()() = 0;
    n_nonzeros.modify_host();
    n_nonzeros.sync_device();

    auto const n_nonzeros_device = n_nonzeros.view_device();
    Kokkos::parallel_for(
            "dense2coo",
            Kokkos::RangePolicy(Kokkos::Serial(), 0, 1),
            KOKKOS_LAMBDA(int const) {
                for (int i = 0; i < dense_matrix.extent_int(0); ++i) {
                    for (int j = 0; j < dense_matrix.extent_int(1); ++j) {
                        double const aij = dense_matrix(i, j);
                        if (Kokkos::abs(aij) >= tol) {
                            rows_idx(n_nonzeros_device()) = i;
                            cols_idx(n_nonzeros_device()) = j;
                            values(n_nonzeros_device()) = aij;
                            n_nonzeros_device()++;
                        }
                    }
                }
            });
    n_nonzeros.modify_device();
    n_nonzeros.sync_host();
    Kokkos::resize(rows_idx, n_nonzeros.view_host()());
    Kokkos::resize(cols_idx, n_nonzeros.view_host()());
    Kokkos::resize(values, n_nonzeros.view_host()());

    return std::make_unique<
            Coo>(dense_matrix.extent(0), dense_matrix.extent(1), rows_idx, cols_idx, values);
}

void SplinesLinearProblem2x2Blocks::compute_schur_complement()
{
    auto const bottom_left_block = m_bottom_left_block.view_host();
    auto const top_right_block = m_top_right_block.view_host();
    Kokkos::parallel_for(
            "compute_schur_complement",
            Kokkos::MDRangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Rank<2>>(
                    {0, 0},
                    {m_bottom_right_block->size(), m_bottom_right_block->size()}),
            [&](int const i, int const j) {
                double val = 0.0;
                for (std::size_t l = 0; l < m_top_left_block->size(); ++l) {
                    val += bottom_left_block(i, l) * top_right_block(l, j);
                }
                m_bottom_right_block
                        ->set_element(i, j, m_bottom_right_block->get_element(i, j) - val);
            });
}

void SplinesLinearProblem2x2Blocks::setup_solver()
{
    // Setup the top-left solver
    m_top_left_block->setup_solver();

    // Compute Q^-1*gamma in top-right block
    m_top_right_block.modify_host();
    m_top_right_block.sync_device();
    m_top_left_block->solve(m_top_right_block.view_device(), false);
    m_top_right_block_coo = dense2coo(m_top_right_block.view_device());
    m_top_right_block.modify_device();
    m_top_right_block.sync_host();

    // Push lambda on device in bottom-left block
    m_bottom_left_block.modify_host();
    m_bottom_left_block.sync_device();
    m_bottom_left_block_coo = dense2coo(m_bottom_left_block.view_device());

    // Compute delta - lambda*Q^-1*gamma in bottom-right block & setup the bottom-right solver
    compute_schur_complement();
    m_bottom_right_block->setup_solver();
}

void SplinesLinearProblem2x2Blocks::spdm_minus1_1(
        Coo* LinOp_,
        MultiRHS const x,
        MultiRHS const y,
        bool const transpose)
{
    assert((!transpose && LinOp.nrows() == y.extent(0))
           || (transpose && LinOp.ncols() == y.extent(0)));
    assert((!transpose && LinOp.ncols() == x.extent(0))
           || (transpose && LinOp.nrows() == x.extent(0)));
    assert(x.extent(1) == y.extent(1));

    Coo LinOp = *LinOp_;

    if (!transpose) {
        Kokkos::parallel_for(
                "ddc_splines_spdm_minus1_1",
                Kokkos::RangePolicy(Kokkos::Serial(), 0, y.extent(1)),
                KOKKOS_LAMBDA(int const j) {
                    for (std::size_t nz_idx = 0; nz_idx < LinOp.nnz(); ++nz_idx) {
                        int const i = LinOp.rows_idx()(nz_idx);
                        int const k = LinOp.cols_idx()(nz_idx);
                        y(i, j) -= LinOp.values()(nz_idx) * x(k, j);
                    }
                });
    } else {
        Kokkos::parallel_for(
                "ddc_splines_spdm_minus1_1_tr",
                Kokkos::RangePolicy(Kokkos::Serial(), 0, y.extent(1)),
                KOKKOS_LAMBDA(int const j) {
                    for (std::size_t nz_idx = 0; nz_idx < LinOp.nnz(); ++nz_idx) {
                        int const i = LinOp.rows_idx()(nz_idx);
                        int const k = LinOp.cols_idx()(nz_idx);
                        y(k, j) -= LinOp.values()(nz_idx) * x(i, j);
                    }
                });
    }
}

void SplinesLinearProblem2x2Blocks::solve(MultiRHS const b, bool const transpose) const
{
    assert(b.extent(0) == size());

    MultiRHS const b1 = Kokkos::
            subview(b,
                    std::pair<std::size_t, std::size_t>(0, m_top_left_block->size()),
                    Kokkos::ALL);
    MultiRHS const b2 = Kokkos::
            subview(b,
                    std::pair<std::size_t, std::size_t>(m_top_left_block->size(), b.extent(0)),
                    Kokkos::ALL);
    if (!transpose) {
        m_top_left_block->solve(b1, false);
        spdm_minus1_1(m_bottom_left_block_coo.get(), b1, b2);
        m_bottom_right_block->solve(b2, false);
        spdm_minus1_1(m_top_right_block_coo.get(), b2, b1);
    } else {
        spdm_minus1_1(m_top_right_block_coo.get(), b1, b2, true);
        m_bottom_right_block->solve(b2, true);
        spdm_minus1_1(m_bottom_left_block_coo.get(), b2, b1, true);
        m_top_left_block->solve(b1, true);
    }
}

} // namespace ddc::detail
