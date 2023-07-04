#pragma once
#include <memory>

#include "matrix_banded.hpp"
#include "matrix_center_block.hpp"
#include "matrix_corner_block.hpp"
#include "matrix_dense.hpp"
#include "matrix_pds_tridiag.hpp"
#include "matrix_periodic_banded.hpp"



class MatrixMaker
{
public:
    static std::unique_ptr<Matrix> make_new_banded(
            int const n,
            int const kl,
            int const ku,
            bool const pds)
    {
        if (kl == ku && kl == 1 && pds) {
            return std::make_unique<Matrix_PDS_Tridiag>(n);
        } else if (2 * kl + 1 + ku >= n) {
            return std::make_unique<Matrix_Dense>(n);
        } else {
            return std::make_unique<Matrix_Banded>(n, kl, ku);
        }
    }
    static std::unique_ptr<Matrix> make_new_periodic_banded(
            int const n,
            int const kl,
            int const ku,
            bool const pds)
    {
        int const border_size = std::max(kl, ku);
        int const banded_size = n - border_size;
        std::unique_ptr<Matrix> block_mat;
        if (pds && kl == ku && kl == 1) {
            block_mat = std::make_unique<Matrix_PDS_Tridiag>(banded_size);
        } else if (
                border_size * n + border_size * (border_size + 1) + (2 * kl + 1 + ku) * banded_size
                >= n * n) {
            return std::make_unique<Matrix_Dense>(n);
        } else {
            block_mat = std::make_unique<Matrix_Banded>(banded_size, kl, ku);
        }
        return std::make_unique<Matrix_Periodic_Banded>(n, kl, ku, std::move(block_mat));
    }
    static std::unique_ptr<Matrix> make_new_block_with_banded_region(
            int const n,
            int const kl,
            int const ku,
            bool const pds,
            int const block1_size,
            int const block2_size)
    {
        int const banded_size = n - block1_size - block2_size;
        std::unique_ptr<Matrix> block_mat;
        if (pds && kl == ku && kl == 1) {
            block_mat = std::make_unique<Matrix_PDS_Tridiag>(banded_size);
        } else if (2 * kl + 1 + ku >= banded_size) {
            return std::make_unique<Matrix_Dense>(n);
        } else {
            block_mat = std::make_unique<Matrix_Banded>(banded_size, kl, ku);
        }
        if (block2_size == 0) {
            return std::make_unique<Matrix_Corner_Block>(n, block1_size, std::move(block_mat));
        } else {
            return std::make_unique<
                    Matrix_Center_Block>(n, block1_size, block2_size, std::move(block_mat));
        }
    }
};
