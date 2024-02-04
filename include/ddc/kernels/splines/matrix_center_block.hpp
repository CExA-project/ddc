#ifndef MATRIX_CENTER_BLOCK_H
#define MATRIX_CENTER_BLOCK_H
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
protected:
	int const top_block_size;
    int const bottom_block_size;
    int const bottom_block_index;
    std::unique_ptr<double[]> swap_array;

public:
    Matrix_Center_Block(
            int const n,
            int const top_block_size,
            int const bottom_block_size,
            std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block<ExecSpace>(n, top_block_size + bottom_block_size, std::move(q))
        , top_block_size(top_block_size)
        , bottom_block_size(bottom_block_size)
        , bottom_block_index(n - bottom_block_size)
        , swap_array(std::make_unique<double[]>(q->get_size()))
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
    ddc::DSpan1D solve_inplace(ddc::DSpan1D const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block<ExecSpace>::solve_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }
    ddc::DSpan1D solve_transpose_inplace(ddc::DSpan1D const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block<ExecSpace>::solve_transpose_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }
    ddc::DSpan2D solve_multiple_inplace(ddc::DSpan2D const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block<ExecSpace>::solve_multiple_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }
    ddc::DSpan2D_stride solve_multiple_inplace2(ddc::DSpan2D_stride const bx) const override
    {
        swap_array_to_corner2(bx);
        Matrix_Corner_Block<ExecSpace>::solve_multiple_inplace2(bx);
        swap_array_to_center2(bx);
        return bx;
    }


protected:
    void adjust_indexes(int& i, int& j) const
    {
        if (i < top_block_size)
            i += m_q_block->get_size();
        else if (i < bottom_block_index)
            i -= top_block_size;

        if (j < top_block_size)
            j += m_q_block->get_size();
        else if (j < bottom_block_index)
            j -= top_block_size;
    }
    ddc::DSpan1D swap_array_to_corner(ddc::DSpan1D const bx) const
    {
        memcpy(swap_array.get(),
               bx.data_handle() + top_block_size,
               m_q_block->get_size() * sizeof(double));
        memcpy(bx.data_handle() + m_q_block->get_size(),
               bx.data_handle(),
               top_block_size * sizeof(double));
        memcpy(bx.data_handle(), swap_array.get(), m_q_block->get_size() * sizeof(double));
        return bx;
    }
    ddc::DSpan2D swap_array_to_corner(ddc::DSpan2D const bx) const
    {
        int const ncols = bx.extent(1);
        memcpy(swap_array.get(),
               bx.data_handle() + top_block_size * ncols,
               m_q_block->get_size() * ncols * sizeof(double));
        memcpy(bx.data_handle() + m_q_block->get_size() * ncols,
               bx.data_handle(),
               top_block_size * ncols * sizeof(double));
        memcpy(bx.data_handle(), swap_array.get(), m_q_block->get_size() * ncols * sizeof(double));
        return bx;
    }
    ddc::DSpan2D_stride swap_array_to_corner2(ddc::DSpan2D_stride const bx) const
    {
        auto bx_top = std::experimental::submdspan(
                bx,
                std::pair<int, int> {0, top_block_size},
                std::experimental::full_extent);
        auto bx_q = std::experimental::submdspan(
                bx,
                std::pair<int, int> {top_block_size, top_block_size + m_q_block->get_size()},
                std::experimental::full_extent);
        auto bx_top_dest = std::experimental::submdspan(
                bx,
                std::pair<int, int> {m_q_block->get_size(), top_block_size + m_q_block->get_size()},
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
                Kokkos::HostSpace>
                bx_top_view(bx_top.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                Kokkos::HostSpace>
                bx_q_view(bx_q.data_handle(), bx_q_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                Kokkos::HostSpace>
                bx_top_dest_view(bx_top_dest.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                Kokkos::HostSpace>
                bx_q_dest_view(bx_q_dest.data_handle(), bx_q_kokkos_layout);
        auto bx_q_buffer = Kokkos::create_mirror(Kokkos::DefaultHostExecutionSpace(), bx_q_view);

        Kokkos::deep_copy(bx_q_buffer, bx_q_view);
        Kokkos::deep_copy(bx_top_dest_view, bx_top_view);
        Kokkos::deep_copy(bx_q_dest_view, bx_q_buffer);
        return bx;
    }
    ddc::DSpan1D swap_array_to_center(ddc::DSpan1D const bx) const
    {
        memcpy(swap_array.get(), bx.data_handle(), m_q_block->get_size() * sizeof(double));
        memcpy(bx.data_handle(),
               bx.data_handle() + m_q_block->get_size(),
               top_block_size * sizeof(double));
        memcpy(bx.data_handle() + top_block_size,
               swap_array.get(),
               m_q_block->get_size() * sizeof(double));
        return bx;
    }
    ddc::DSpan2D swap_array_to_center(ddc::DSpan2D const bx) const
    {
        int const ncols = bx.extent(1);
        memcpy(swap_array.get(), bx.data_handle(), m_q_block->get_size() * ncols * sizeof(double));
        memcpy(bx.data_handle(),
               bx.data_handle() + m_q_block->get_size() * ncols,
               top_block_size * ncols * sizeof(double));
        memcpy(bx.data_handle() + top_block_size * ncols,
               swap_array.get(),
               m_q_block->get_size() * ncols * sizeof(double));
        return bx;
    }
    ddc::DSpan2D_stride swap_array_to_center2(ddc::DSpan2D_stride const bx) const
    {
        auto bx_top = std::experimental::submdspan(
                bx,
                std::pair<int, int> {0, top_block_size},
                std::experimental::full_extent);
        auto bx_q = std::experimental::submdspan(
                bx,
                std::pair<int, int> {top_block_size, top_block_size + m_q_block->get_size()},
                std::experimental::full_extent);
        auto bx_top_src = std::experimental::submdspan(
                bx,
                std::pair<int, int> {m_q_block->get_size(), top_block_size + m_q_block->get_size()},
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
                Kokkos::HostSpace>
                bx_top_view(bx_top.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                Kokkos::HostSpace>
                bx_q_view(bx_q.data_handle(), bx_q_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                Kokkos::HostSpace>
                bx_top_src_view(bx_top_src.data_handle(), bx_top_kokkos_layout);
        Kokkos::View<
                ddc::detail::mdspan_to_kokkos_element_t<double, 2>,
                Kokkos::LayoutStride,
                Kokkos::HostSpace>
                bx_q_src_view(bx_q_src.data_handle(), bx_q_kokkos_layout);
        auto bx_q_buffer
                = Kokkos::create_mirror(Kokkos::DefaultHostExecutionSpace(), bx_q_src_view);

        Kokkos::deep_copy(bx_q_buffer, bx_q_src_view);
        Kokkos::deep_copy(bx_top_view, bx_top_src_view);
        Kokkos::deep_copy(bx_q_view, bx_q_buffer);
        return bx;
    }
};

} // namespace ddc::detail
#endif // MATRIX_CENTER_BLOCK_H
