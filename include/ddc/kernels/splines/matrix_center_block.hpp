#ifndef MATRIX_CENTER_BLOCK_H
#define MATRIX_CENTER_BLOCK_H
#include <memory>
#include <utility>

#include <string.h>

#include "matrix_corner_block.hpp"
#include "view.hpp"

namespace ddc::detail {
class Matrix;

class Matrix_Center_Block : public Matrix_Corner_Block
{
public:
    Matrix_Center_Block(
            int const n,
            int const top_block_size,
            int const bottom_block_size,
            std::unique_ptr<Matrix> q)
        : Matrix_Corner_Block(n, top_block_size + bottom_block_size, std::move(q))
        , top_block_size(top_block_size)
        , bottom_block_size(bottom_block_size)
        , bottom_block_index(n - bottom_block_size)
        , swap_array(std::make_unique<double[]>(q_block->get_size()))
    {
    }

    void reset() const override {}


    double get_element(int i, int j) const override
    {
        adjust_indexes(i, j);
        return Matrix_Corner_Block::get_element(i, j);
    }
    void set_element(int i, int j, double a_ij) override
    {
        adjust_indexes(i, j);
        Matrix_Corner_Block::set_element(i, j, a_ij);
    }
    ddc::DSpan1D solve_inplace(ddc::DSpan1D const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block::solve_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }
    ddc::DSpan1D solve_transpose_inplace(ddc::DSpan1D const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block::solve_transpose_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }
    ddc::DSpan2D solve_multiple_inplace(ddc::DSpan2D const bx) const override
    {
        swap_array_to_corner(bx);
        Matrix_Corner_Block::solve_multiple_inplace(bx);
        swap_array_to_center(bx);
        return bx;
    }

protected:
    void adjust_indexes(int& i, int& j) const
    {
        if (i < top_block_size)
            i += q_block->get_size();
        else if (i < bottom_block_index)
            i -= top_block_size;

        if (j < top_block_size)
            j += q_block->get_size();
        else if (j < bottom_block_index)
            j -= top_block_size;
    }
    ddc::DSpan1D swap_array_to_corner(ddc::DSpan1D const bx) const
    {
        memcpy(swap_array.get(),
               bx.data_handle() + top_block_size,
               q_block->get_size() * sizeof(double));
        memcpy(bx.data_handle() + q_block->get_size(),
               bx.data_handle(),
               top_block_size * sizeof(double));
        memcpy(bx.data_handle(), swap_array.get(), q_block->get_size() * sizeof(double));
        return bx;
    }
    ddc::DSpan2D swap_array_to_corner(ddc::DSpan2D const bx) const
    {
        int const ncols = bx.extent(1);
        memcpy(swap_array.get(),
               bx.data_handle() + top_block_size * ncols,
               q_block->get_size() * ncols * sizeof(double));
        memcpy(bx.data_handle() + q_block->get_size() * ncols,
               bx.data_handle(),
               top_block_size * ncols * sizeof(double));
        memcpy(bx.data_handle(), swap_array.get(), q_block->get_size() * ncols * sizeof(double));
        return bx;
    }
    ddc::DSpan1D swap_array_to_center(ddc::DSpan1D const bx) const
    {
        memcpy(swap_array.get(), bx.data_handle(), q_block->get_size() * sizeof(double));
        memcpy(bx.data_handle(),
               bx.data_handle() + q_block->get_size(),
               top_block_size * sizeof(double));
        memcpy(bx.data_handle() + top_block_size,
               swap_array.get(),
               q_block->get_size() * sizeof(double));
        return bx;
    }
    ddc::DSpan2D swap_array_to_center(ddc::DSpan2D const bx) const
    {
        int const ncols = bx.extent(1);
        memcpy(swap_array.get(), bx.data_handle(), q_block->get_size() * ncols * sizeof(double));
        memcpy(bx.data_handle(),
               bx.data_handle() + q_block->get_size() * ncols,
               top_block_size * ncols * sizeof(double));
        memcpy(bx.data_handle() + top_block_size * ncols,
               swap_array.get(),
               q_block->get_size() * ncols * sizeof(double));
        return bx;
    }
    int const top_block_size;
    int const bottom_block_size;
    int const bottom_block_index;
    std::unique_ptr<double[]> swap_array;
};

} // namespace ddc::detail
#endif // MATRIX_CENTER_BLOCK_H