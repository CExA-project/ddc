// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <utility>

#include <string.h>

#include "matrix_corner_block.hpp"
#include "view.hpp"

namespace ddc::detail {
class Matrix;

template <class ExecSpace>
class Matrix_Center_Block : public Matrix_Corner_Block<ExecSpace>
{
    // Necessary because we inherit from a template class, otherwise we should use this-> everywhere
    using Matrix_Corner_Block<ExecSpace>::get_size;
    using Matrix_Corner_Block<ExecSpace>::solve_inplace;
    using Matrix_Corner_Block<ExecSpace>::m_q_block;
    using Matrix_Corner_Block<ExecSpace>::m_delta;
    using Matrix_Corner_Block<ExecSpace>::m_Abm_1_gamma;
    using Matrix_Corner_Block<ExecSpace>::m_lambda;

protected:
    int const m_top_block_size;
    int const m_bottom_block_size;
    int const m_bottom_block_index;

public:
    Matrix_Center_Block(
            int const n,
            int const m_top_block_size,
            int const m_bottom_block_size,
            std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block<ExecSpace>(n, m_top_block_size + m_bottom_block_size, std::move(q))
        , m_top_block_size(m_top_block_size)
        , m_bottom_block_size(m_bottom_block_size)
        , m_bottom_block_index(n - m_bottom_block_size)
    {
    }

    double get_element(int i, int j) const override
    {
        adjust_indexes(i, j);
        return Matrix_Corner_Block<ExecSpace>::get_element(i, j);
    }

    void set_element(int i, int j, double aij) override
    {
        adjust_indexes(i, j);
        Matrix_Corner_Block<ExecSpace>::set_element(i, j, aij);
    }

    ddc::DSpan2D_stride solve_inplace(ddc::DSpan2D_stride const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block<ExecSpace>::solve_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }


protected:
    void adjust_indexes(int& i, int& j) const
    {
        if (i < m_top_block_size)
            i += m_q_block->get_size();
        else if (i < m_bottom_block_index)
            i -= m_top_block_size;

        if (j < m_top_block_size)
            j += m_q_block->get_size();
        else if (j < m_bottom_block_index)
            j -= m_top_block_size;
    }
    ddc::DSpan2D_stride swap_array_to_corner(ddc::DSpan2D_stride const bx) const
    {
        auto bx_top = std::experimental::submdspan(
                bx,
                std::pair<int, int> {0, m_top_block_size},
                std::experimental::full_extent);
        auto bx_q = std::experimental::submdspan(
                bx,
                std::pair<int, int> {m_top_block_size, m_top_block_size + m_q_block->get_size()},
                std::experimental::full_extent);
        auto bx_top_dest = std::experimental::submdspan(
                bx,
                std::pair<
                        int,
                        int> {m_q_block->get_size(), m_top_block_size + m_q_block->get_size()},
                std::experimental::full_extent);
        auto bx_q_dest = std::experimental::submdspan(
                bx,
                std::pair<int, int> {0, m_q_block->get_size()},
                std::experimental::full_extent);
        // Necessary to convert to Kokkos::View to deep_copy afterward, this is inlining of code present in ddc/details/kokkos.hpp, maybe we could make a dedicated function for it
        auto bx_top_kokkos_layout = ddc::detail::build_kokkos_layout(
                bx_top.extents(),
                bx_top.mapping(),
                std::make_index_sequence<2> {});
        auto bx_q_kokkos_layout = ddc::detail::
                build_kokkos_layout(bx_q.extents(), bx_q.mapping(), std::make_index_sequence<2> {});
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_top_view(bx_top.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_q_view(bx_q.data_handle(), bx_q_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_top_dest_view(bx_top_dest.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_q_dest_view(bx_q_dest.data_handle(), bx_q_kokkos_layout);
        auto bx_q_buffer = Kokkos::create_mirror(ExecSpace(), bx_q_view);

        Kokkos::deep_copy(bx_q_buffer, bx_q_view);
        Kokkos::deep_copy(bx_top_dest_view, bx_top_view);
        Kokkos::deep_copy(bx_q_dest_view, bx_q_buffer);
        return bx;
    }
    ddc::DSpan2D_stride swap_array_to_center(ddc::DSpan2D_stride const bx) const
    {
        auto bx_top = std::experimental::submdspan(
                bx,
                std::pair<int, int> {0, m_top_block_size},
                std::experimental::full_extent);
        auto bx_q = std::experimental::submdspan(
                bx,
                std::pair<int, int> {m_top_block_size, m_top_block_size + m_q_block->get_size()},
                std::experimental::full_extent);
        auto bx_top_src = std::experimental::submdspan(
                bx,
                std::pair<
                        int,
                        int> {m_q_block->get_size(), m_top_block_size + m_q_block->get_size()},
                std::experimental::full_extent);
        auto bx_q_src = std::experimental::submdspan(
                bx,
                std::pair<int, int> {0, m_q_block->get_size()},
                std::experimental::full_extent);
        // Necessary to convert to Kokkos::View to deep_copy afterward, this is inlining of code present in ddc/details/kokkos.hpp, maybe we could make a dedicated function for it
        auto bx_top_kokkos_layout = ddc::detail::build_kokkos_layout(
                bx_top.extents(),
                bx_top.mapping(),
                std::make_index_sequence<2> {});
        auto bx_q_kokkos_layout = ddc::detail::
                build_kokkos_layout(bx_q.extents(), bx_q.mapping(), std::make_index_sequence<2> {});
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_top_view(bx_top.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_q_view(bx_q.data_handle(), bx_q_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_top_src_view(bx_top_src.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                typename ExecSpace::memory_space>
                bx_q_src_view(bx_q_src.data_handle(), bx_q_kokkos_layout);
        auto bx_q_buffer = Kokkos::create_mirror(ExecSpace(), bx_q_src_view);

        Kokkos::deep_copy(bx_q_buffer, bx_q_src_view);
        Kokkos::deep_copy(bx_top_view, bx_top_src_view);
        Kokkos::deep_copy(bx_q_view, bx_q_buffer);
        return bx;
    }
};

} // namespace ddc::detail
